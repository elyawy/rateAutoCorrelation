"""
Null control: shuffle all columns of each whole alignment (destroys site correlation, keeps column
composition) for genes with low / mid / high predicted rho, then rerun the CNN on the shuffled alignments.
If the model detects site correlation, shuffled predictions should drop to low rho.
"""

import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "common"))
import common.config as config


class RhoRegressor(nn.Module):
    """CNN embedding of MSA summary stats -> MLP head -> scalar rho estimate."""

    def __init__(self, n_scalars=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + n_scalars, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, scalars):
        embedding = self.cnn(x).mean(dim=-1)
        return self.head(torch.cat([embedding, scalars], dim=1)).squeeze(-1)


extract = importlib.import_module("1_extract_orthomam_features")  # reuse the loading / feature code

N_PER_GROUP = 500
SEED = 0
MIN_LENGTH_FOR_GROUPS = 500  # only draw genes from alignments this long (predictions are noisier below); 0 = no limit
PREDICTIONS_CSV = config.RESULTS_DIR / "orthomam_predictions_amino.csv"
CNN_WEIGHTS = REPO_DIR / "rho_regressor_amino_alphaprior_w1000.pt"  # trained on 1000-site alignments
OUTPUT_NAME = "shuffle_control_amino.csv"

GROUPS = {
    "low (<0.25)": lambda r: r < 0.25,
    "mid (0.45-0.55)": lambda r: (r >= 0.45) & (r <= 0.55),
    "high (>0.75)": lambda r: r > 0.75,
}


def shuffled_alignment(filename, rng):
    """Load a gene's whole alignment and shuffle all its columns with one random permutation."""
    _, content = extract.extract_fasta_from_zip(extract.AA_DIR / f"{filename}.zip")  # adjust if zip names differ
    sequences = extract.parse_fasta_content(content)
    columns = np.array([list(seq) for seq in sequences])  # (n_sequences, n_sites)
    columns = columns[:, rng.permutation(columns.shape[1])]
    return ["".join(row) for row in columns]


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[(df["status"] == "ok") & (df["alignment_length"] >= MIN_LENGTH_FOR_GROUPS)]
    df = df.dropna(subset=["pred_rho"]).copy()

    selected = []
    for name, cond in GROUPS.items():
        pool = df[cond(df["pred_rho"])]
        print(f"{name}: {len(pool)} genes available")
        pick = pool.sample(min(N_PER_GROUP, len(pool)), random_state=SEED)
        selected.append(pick.assign(group=name))
    selected = pd.concat(selected, ignore_index=True)

    model = RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()

    # features and prediction of each shuffled alignment, one at a time since lengths differ
    shuffled_rho = np.zeros(len(selected))
    with torch.no_grad():
        for i, filename in enumerate(selected["filename"]):
            sequences = shuffled_alignment(filename, rng)
            features = extract.calculate_cnn_input(sequences, alphabet_size=extract.ALPHABET_SIZE).unsqueeze(0)
            scalars = extract.calculate_scalar_features(sequences, alphabet_size=extract.ALPHABET_SIZE).unsqueeze(0)
            shuffled_rho[i] = model(features, scalars).item()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(selected)}")
    selected["shuffled_rho"] = shuffled_rho

    selected[["filename", "group", "pred_rho", "shuffled_rho"]].to_csv(config.RESULTS_DIR / OUTPUT_NAME, index=False)

    print("\nmean rho per group (original -> shuffled)")
    for name, g in selected.groupby("group", sort=False):
        print(f"  {name} (n={len(g)}): {g['pred_rho'].mean():.3f} -> {g['shuffled_rho'].mean():.3f}")


if __name__ == "__main__":
    main()