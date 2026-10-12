"""
Null control: shuffle the columns inside each 250-site chunk (destroys site correlation, keeps column
composition) for genes with low / mid / high predicted rho, then rerun CNN and SBI on the shuffled chunks.
If the models detect site correlation, shuffled predictions should drop to low rho.
"""

import importlib
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "training"))
sys.path.insert(0, str(REPO_DIR / "common"))
import common.config as config
from cnn_embedding import CNNScalarEmbedding  # noqa: F401  (needed to unpickle the posterior)


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
N_POSTERIOR_SAMPLES = 1000
MAX_SAMPLING_TIME = 10
WINDOW = extract.WINDOW
PREDICTIONS_CSV = config.RESULTS_DIR / "orthomam_predictions_amino.csv"
CNN_WEIGHTS = REPO_DIR / f"rho_regressor_amino_alphaprior_w{WINDOW}.pt"
POSTERIOR_PATH = REPO_DIR / "posterior_amino_alphaprior.pkl"
OUTPUT_NAME = "shuffle_control_amino.csv"

# groups use the mean of the CNN and SBI gene predictions
GROUPS = {
    "low (<0.25)": lambda r: r < 0.25,
    "mid (0.45-0.55)": lambda r: (r >= 0.45) & (r <= 0.55),
    "high (>0.75)": lambda r: r > 0.75,
}


def shuffled_chunks(filename, rng):
    """Chunk a gene's alignment like step 1, shuffling the columns inside each chunk."""
    _, content = extract.extract_fasta_from_zip(extract.AA_DIR / f"{filename}.zip")  # adjust if zip names differ
    sequences = extract.parse_fasta_content(content)
    chunks = []
    for start in range(0, len(sequences[0]) - WINDOW + 1, WINDOW):
        perm = rng.permutation(WINDOW)
        chunks.append(["".join(seq[start + j] for j in perm) for seq in sequences])
    return chunks


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[df["status"] == "ok"].dropna(subset=["pred_rho", "pred_rho_sbi"]).copy()
    df["mean_rho"] = (df["pred_rho"] + df["pred_rho_sbi"]) / 2

    selected = []
    for name, cond in GROUPS.items():
        pool = df[cond(df["mean_rho"])]
        print(f"{name}: {len(pool)} genes available")
        pick = pool.sample(min(N_PER_GROUP, len(pool)), random_state=SEED)
        selected.append(pick.assign(group=name))
    selected = pd.concat(selected, ignore_index=True)

    # features of the shuffled chunks
    features, scalars, owner = [], [], []
    for i, filename in enumerate(selected["filename"]):
        for chunk in shuffled_chunks(filename, rng):
            features.append(extract.calculate_cnn_input(chunk, alphabet_size=extract.ALPHABET_SIZE))
            scalars.append(extract.calculate_scalar_features(chunk, alphabet_size=extract.ALPHABET_SIZE))
            owner.append(i)
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(selected)}")
    features, scalars, owner = torch.stack(features), torch.stack(scalars), np.array(owner)

    # CNN
    model = RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        rho_cnn = model(features, scalars).numpy()

    # SBI (median per chunk; timeouts stay NaN)
    with open(POSTERIOR_PATH, "rb") as f:
        posterior = pickle.load(f)
    x = torch.cat([features.flatten(1), scalars], dim=1)
    rho_sbi = np.full(len(x), np.nan)
    for i in range(len(x)):
        try:
            samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x[i].unsqueeze(0), show_progress_bars=False,
                                       max_sampling_time=MAX_SAMPLING_TIME)
            rho_sbi[i] = samples[:, 1].median().item()
        except RuntimeError:
            pass
        if (i + 1) % 200 == 0:
            print(f"  posterior {i + 1}/{len(x)}")

    per_gene = pd.DataFrame({"owner": owner, "cnn": rho_cnn, "sbi": rho_sbi}).groupby("owner").mean()
    selected["shuffled_rho_cnn"] = per_gene["cnn"].reindex(range(len(selected))).values
    selected["shuffled_rho_sbi"] = per_gene["sbi"].reindex(range(len(selected))).values
    selected[["filename", "group", "pred_rho", "pred_rho_sbi", "shuffled_rho_cnn", "shuffled_rho_sbi"]].to_csv(
        config.RESULTS_DIR / OUTPUT_NAME, index=False)

    print("\nmean rho per group (original -> shuffled)")
    for name, g in selected.groupby("group", sort=False):
        print(f"  {name} (n={len(g)}): CNN {g['pred_rho'].mean():.3f} -> {g['shuffled_rho_cnn'].mean():.3f} | "
              f"SBI {g['pred_rho_sbi'].mean():.3f} -> {g['shuffled_rho_sbi'].mean():.3f}")


if __name__ == "__main__":
    main()
