#!/usr/bin/env python3
"""Run the trained CNN on real MSAs."""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd
import torch

from common.features_calculator import calculate_cnn_input, calculate_scalar_features
from common.model import RhoRegressor


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN on real MSAs")
    parser.add_argument("--msa-dir", type=str, default="data/no_sync/msas")
    parser.add_argument("--weights", type=str, default="models/rho_regressor_amino_indel_trees500_gap_w1000.pt")
    parser.add_argument("--out", type=str, default="data/results_real_new.csv")
    args = parser.parse_args()

    model = RhoRegressor()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    rows = []
    for msa in sorted(pathlib.Path(args.msa_dir).glob("*.fasta")):
        start = time.time()
        seqs = read_fasta(msa)
        x = calculate_cnn_input(seqs, alphabet_size=20).float().unsqueeze(0)
        s = calculate_scalar_features(seqs, alphabet_size=20).float().unsqueeze(0)
        with torch.no_grad():
            rho = model(x, s).item()
        elapsed = time.time() - start
        rows.append({"gene": msa.stem, "time": elapsed, "rho": rho})

    df = pd.DataFrame(rows, columns=["gene", "time", "rho"])
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
