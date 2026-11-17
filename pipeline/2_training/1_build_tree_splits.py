"""
Pick a train and a test set of trees from protein_estimates.csv, stratified by tree length.

Trees are cut into N_BINS bins by tree-length quantile (each bin = 10% of all usable trees), and the
same number of trees is drawn from every bin. So the sets mirror the overall tree-length distribution
while still covering short and long trees. Train and test do not overlap.

Only genes that also have a row in the indel-params CSV are used (the training script skips the others).

Usage: python build_tree_splits.py protein_estimates.csv orthomam_indel_params.csv [output_dir]
"""

import re
import sys
from pathlib import Path

import pandas as pd
import config

N_TRAIN = 500
N_TEST = 200
N_BINS = 10
MIN_TAXA = 20
SEED = 0


def tree_length(newick: str) -> float:
    """Sum of all branch lengths (every number after a colon)."""
    return sum(float(x) for x in re.findall(r":([0-9.eE+-]+)", newick))


def main():
    estimates_path, indel_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else config.REPO_ROOT / "pipeline" / "2_training" / "inputs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(estimates_path)
    indel_genes = set(pd.read_csv(indel_path)["name"].str.removesuffix(".fasta"))
    df = df[df["gene"].astype(str).isin(indel_genes)]
    print(f"{len(df)} genes with both a tree and an indel model")
    df["tree_len"] = df["newick"].map(tree_length)
    df = df[(df["n_taxa"] >= MIN_TAXA) & (df["tree_len"] > 0)].copy()
    print(f"{len(df)} usable trees")

    df["bin"] = pd.qcut(df["tree_len"], N_BINS, labels=False)

    train_per_bin, test_per_bin = N_TRAIN // N_BINS, N_TEST // N_BINS
    train = df.groupby("bin").sample(n=train_per_bin, random_state=SEED)
    rest = df.drop(train.index)
    test = rest.groupby("bin").sample(n=test_per_bin, random_state=SEED)

    for name, subset in [("train", train), ("test", test)]:
        path = out_dir / f"protein_estimates_{name}{len(subset)}.csv"
        subset.drop(columns="bin").to_csv(path, index=False)
        print(f"{name}: {len(subset)} trees -> {path}")
        print(subset["tree_len"].describe(percentiles=[.1, .5, .9]).round(2).to_string())


if __name__ == "__main__":
    main()
