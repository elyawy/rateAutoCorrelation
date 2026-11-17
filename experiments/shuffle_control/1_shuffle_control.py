"""
Null control: shuffle all columns of each whole alignment (destroys site correlation, keeps column
composition) for genes with low / mid / high predicted rho, then rerun the CNN on the shuffled alignments.
If the model detects site correlation, shuffled predictions should drop to low rho.

Needs results/real_data/results_real_new.csv (pipeline/4_inference/run_cnn_real.py) and the MSAs in
config.REAL_MSA_DIR. Writes results/shuffle_control/shuffle_control_amino.csv.
"""

import pathlib

import numpy as np
import pandas as pd
import torch

import config
from common.features_calculator import calculate_cnn_input, calculate_scalar_features
from common.model import RhoRegressor

N_PER_GROUP = 500
SEED = 0
ALPHABET_SIZE = 20
OUTPUT_CSV = pathlib.Path(config.RESULTS_DIR) / "shuffle_control" / "shuffle_control_amino.csv"

GROUPS = {
    "low (<0.25)": lambda r: r < 0.25,
    "mid (0.45-0.55)": lambda r: (r >= 0.45) & (r <= 0.55),
    "high (>0.75)": lambda r: r > 0.75,
}


def read_fasta(path):
    seqs, cur = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def shuffled_alignment(gene, rng):
    """Load a gene's whole alignment and shuffle all its columns with one random permutation."""
    sequences = read_fasta(pathlib.Path(config.REAL_MSA_DIR) / f"{gene}.fasta")
    columns = np.array([list(seq) for seq in sequences])  # (n_sequences, n_sites)
    columns = columns[:, rng.permutation(columns.shape[1])]
    return ["".join(row) for row in columns]


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(config.REAL_CNN_RESULTS, dtype={"gene": str}).dropna(subset=["rho"])

    selected = []
    for name, cond in GROUPS.items():
        pool = df[cond(df["rho"])]
        print(f"{name}: {len(pool)} genes available")
        pick = pool.sample(min(N_PER_GROUP, len(pool)), random_state=SEED)
        selected.append(pick.assign(group=name))
    selected = pd.concat(selected, ignore_index=True)

    model = RhoRegressor()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
    model.eval()

    # features and prediction of each shuffled alignment, one at a time since lengths differ
    shuffled_rho = np.zeros(len(selected))
    with torch.no_grad():
        for i, gene in enumerate(selected["gene"]):
            sequences = shuffled_alignment(gene, rng)
            x = calculate_cnn_input(sequences, alphabet_size=ALPHABET_SIZE).float().unsqueeze(0)
            s = calculate_scalar_features(sequences, alphabet_size=ALPHABET_SIZE).float().unsqueeze(0)
            shuffled_rho[i] = model(x, s).item()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(selected)}")
    selected["shuffled_rho"] = shuffled_rho

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected[["gene", "group", "rho", "shuffled_rho"]].to_csv(OUTPUT_CSV, index=False)

    print("\nmean rho per group (original -> shuffled)")
    for name, g in selected.groupby("group", sort=False):
        print(f"  {name} (n={len(g)}): {g['rho'].mean():.3f} -> {g['shuffled_rho'].mean():.3f}")


if __name__ == "__main__":
    main()
