"""
Compare the rho regressor with codeml's ML rho on the real OrthoMaM protein MSAs.

Reads msas/{gene}.fasta, codeml_runs_real/{gene}/mlb and protein_estimates.csv,
writes results_real.csv (gene, alpha, nn_rho, codeml_rho) and prints correlation / RMSE.
"""

import pathlib
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / "training"), str(ROOT / "common")])

from train_rho_regressor_amino_indel import RhoRegressor
from features_calculator import calculate_cnn_input, calculate_scalar_features

MODEL_PATH = ROOT / "rho_regressor_amino_indel_w1000.pt"
ALPHABET_SIZE = 20
MSA_DIR = pathlib.Path("msas")
RUNS_DIR = pathlib.Path("codeml_runs_real")
ESTIMATES_CSV = pathlib.Path("protein_estimates.csv")


def read_fasta_seqs(path):
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    seqs.append("".join(cur))
    return seqs


def parse_codeml_rho(mlb_path):
    if not mlb_path.exists():
        return np.nan
    match = re.search(r"rho[^=\n]*=\s*([-0-9.eE+]+)", mlb_path.read_text(), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def main():
    model = RhoRegressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    rows = []
    for fasta in sorted(MSA_DIR.glob("*.fasta")):
        gene = fasta.stem
        seqs = read_fasta_seqs(fasta)
        x = calculate_cnn_input(seqs, alphabet_size=ALPHABET_SIZE).unsqueeze(0)
        s = calculate_scalar_features(seqs, alphabet_size=ALPHABET_SIZE).unsqueeze(0)
        with torch.no_grad():
            nn_rho = model(x, s).item()
        rows.append({"gene": gene, "nn_rho": nn_rho,
                     "codeml_rho": parse_codeml_rho(RUNS_DIR / gene / "mlb")})

    df = pd.DataFrame(rows)
    est = pd.read_csv(ESTIMATES_CSV, usecols=["gene", "alpha"]).astype({"gene": str})
    df = df.merge(est, on="gene", how="left")
    df.to_csv("results_real.csv", index=False)

    print(f"codeml failed/unparsed: {df['codeml_rho'].isna().sum()}/{len(df)}")
    ok = df.dropna(subset=["codeml_rho"])
    diff = ok["nn_rho"] - ok["codeml_rho"]
    print(f"NN mean {ok['nn_rho'].mean():.3f} | codeml mean {ok['codeml_rho'].mean():.3f}")
    print(f"corr {np.corrcoef(ok['nn_rho'], ok['codeml_rho'])[0, 1]:.3f} | "
          f"RMSE {np.sqrt((diff ** 2).mean()):.3f} | mean diff (NN - codeml) {diff.mean():.3f}")


if __name__ == "__main__":
    main()
