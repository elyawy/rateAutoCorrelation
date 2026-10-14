"""
Step 3: Compare the rho regressor with codeml's ML rho on the simulated MSAs.

Reads truth.csv (step 1) and codeml 'mlb' outputs (step 2), runs the trained
regressor on each alignment, writes results.csv and prints RMSE / correlation.
"""

import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]  # <root>/benchmarks/codeml_benchmark/3_compare.py
sys.path.extend([str(ROOT / "training"), str(ROOT / "common")])

import config
from train_rho_regressor_amino_alphaprior import RhoRegressor
from features_calculator import calculate_cnn_input, calculate_scalar_features

MODEL_PATH = ROOT / "rho_regressor_amino_alphaprior_w1000.pt"
ALPHABET_SIZE = 20
MIN_VARIABLE_SITES = 10  # alignments with fewer variable columns are excluded from the stats


def read_phylip_sequences(path):
    lines = path.read_text().strip().split("\n")[1:]
    return [line.split()[-1] for line in lines]


def count_variable_sites(seqs):
    """Columns with more than one distinct residue (gaps ignored)."""
    return sum(len(set(col) - {"-"}) > 1 for col in zip(*seqs))


def parse_codeml_rho(mlb_path):
    if not mlb_path.exists():
        return np.nan
    match = re.search(r"rho[^=\n]*=\s*([-0-9.eE+]+)", mlb_path.read_text(), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def main():
    sim_root = pathlib.Path(config.SIMULATED_DATA_DIR)
    runs_root = pathlib.Path(config.CODEML_RUNS_DIR)
    truth = pd.read_csv(sim_root / "truth.csv")

    model = RhoRegressor()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    torch.set_num_threads(1)  # one CPU core, like a single codeml run

    reg_rho, codeml_rho, n_variable, reg_time = [], [], [], []
    for name in truth["simulation"]:
        seqs = read_phylip_sequences(sim_root / name / "alignment.phy")
        n_variable.append(count_variable_sites(seqs))
        t0 = time.perf_counter()
        x = calculate_cnn_input(seqs, alphabet_size=ALPHABET_SIZE).unsqueeze(0)
        s = calculate_scalar_features(seqs, alphabet_size=ALPHABET_SIZE).unsqueeze(0)
        with torch.no_grad():
            reg_rho.append(model(x, s).item())
        reg_time.append(time.perf_counter() - t0)
        codeml_rho.append(parse_codeml_rho(runs_root / name / "mlb"))

    truth["regressor_rho"] = reg_rho
    truth["codeml_rho"] = codeml_rho
    truth["n_variable"] = n_variable
    truth["regressor_time"] = reg_time
    truth.to_csv("results.csv", index=False)

    print(f"codeml failed/unparsed: {truth['codeml_rho'].isna().sum()}/{len(truth)}")
    ok = truth.dropna(subset=["codeml_rho"])
    n_before = len(ok)
    ok = ok[ok["n_variable"] >= MIN_VARIABLE_SITES]
    print(f"excluded (< {MIN_VARIABLE_SITES} variable sites): {n_before - len(ok)}/{n_before}")
    for scale, g in ok.groupby("scale"):
        print(f"\nscale={scale}")
        for col in ("regressor_rho", "codeml_rho"):
            rmse = np.sqrt(np.mean((g[col] - g["rho"]) ** 2))
            corr = np.corrcoef(g[col], g["rho"])[0, 1]
            print(f"  {col:14s} vs truth: RMSE={rmse:.3f} corr={corr:.3f}")
        print(g.groupby("rho")[["regressor_rho", "codeml_rho"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()