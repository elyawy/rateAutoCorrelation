"""
Run the OLD SBI posterior (random trees, 500 sites, theta = [alpha, rho, log10 tree_scale]) on real OrthoMaM
chunks for the manifest genes, then relate its rho to the real trees' divergence (mean branch, zero-branch share).
Run from the repo so that the `1_extract_orthomam_features.py` helpers import.
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

extract = importlib.import_module("1_extract_orthomam_features")  # zip / fasta helpers + calculate_cnn_input

WINDOW = 500
OLD_POSTERIOR_PATH = "training/sbi_models/posterior_500_10k.pt"
TREE_METRICS_CSV = "manifest_tree_metrics.csv"  # per-gene tree metrics (from the manifest)
ZIP_PATTERN = "{key}_NT_AL_AA.fasta.zip"
N_POSTERIOR_SAMPLES = 1000
MAX_SAMPLING_TIME = 10
LOG10_SCALE_LOW = np.log10(0.005)  # lower edge of the old tree_scale prior
OUTPUT_NAME = "old_model_on_manifest_genes.csv"


class CNN1dEmbedding(nn.Module):
    """Must be defined here (in __main__) with the same layers, so the saved posterior can be unpickled."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)


def gene_chunks(key):
    """Non-overlapping 500-site chunks of a gene's alignment (empty list if too short / DNA)."""
    zip_path = extract.AA_DIR / ZIP_PATTERN.format(key=key)
    if not zip_path.exists():
        return []
    _, content = extract.extract_fasta_from_zip(zip_path)
    if content is None:
        return []
    sequences = extract.parse_fasta_content(content)
    if not sequences or extract.looks_like_dna(sequences) or len(sequences[0]) < WINDOW:
        return []
    return [[seq[s:s + WINDOW] for seq in sequences] for s in range(0, len(sequences[0]) - WINDOW + 1, WINDOW)]


def main():
    posterior = torch.load(OLD_POSTERIOR_PATH, map_location="cpu", weights_only=False)
    tree = pd.read_csv(TREE_METRICS_CSV, dtype={"gene_key": str})

    rows = []
    for i, key in enumerate(tree["gene_key"]):
        chunks = gene_chunks(key)
        meds = []
        for chunk in chunks:
            x = extract.calculate_cnn_input(chunk, alphabet_size=extract.ALPHABET_SIZE).float().unsqueeze(0)  # (1, 2, 500)
            try:
                samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x, show_progress_bars=False,
                                           max_sampling_time=MAX_SAMPLING_TIME)
                meds.append(samples.median(dim=0).values.numpy())  # [alpha, rho, log10 tree_scale]
            except RuntimeError:
                pass
        if meds:
            m = np.mean(meds, axis=0)
            rows.append({"gene_key": key, "n_chunks": len(chunks), "n_failed": len(chunks) - len(meds),
                         "old_alpha": m[0], "old_rho": m[1], "old_log10_scale": m[2]})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(tree)} genes")

    res = pd.DataFrame(rows).merge(tree, on="gene_key")
    res["log10_mean_branch"] = np.log10(res["mean_branch"])
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.RESULTS_DIR / OUTPUT_NAME, index=False)

    print(f"\n{len(res)} genes with >= {WINDOW} sites; old rho mean={res['old_rho'].mean():.3f}")
    print(f"old tree_scale within 0.1 (log10) of the lower prior edge: "
          f"{(res['old_log10_scale'] < LOG10_SCALE_LOW + 0.1).mean():.1%} of genes")

    print("\ncorrelation of old rho with: ")
    for col in ["log10_mean_branch", "frac_zero_branches", "total_length", "old_log10_scale"]:
        print(f"  {col}: {res['old_rho'].corr(res[col]):+.3f}")

    # old rho ~ log10 mean branch + zero-branch share (standardized, so coefficients are comparable)
    X = res[["log10_mean_branch", "frac_zero_branches"]]
    Xs = (X - X.mean()) / X.std()
    A = np.column_stack([np.ones(len(Xs)), Xs.values])
    coef, *_ = np.linalg.lstsq(A, res["old_rho"].values, rcond=None)
    r2 = 1 - np.sum((res["old_rho"].values - A @ coef) ** 2) / np.sum((res["old_rho"] - res["old_rho"].mean()) ** 2)
    print(f"\nregression (standardized): log10_mean_branch {coef[1]:+.3f}, frac_zero_branches {coef[2]:+.3f}, R2={r2:.3f}")

    print("\nold rho by quartile of zero-branch share:")
    print(res.groupby(pd.qcut(res["frac_zero_branches"], 4, duplicates="drop"), observed=True)["old_rho"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
