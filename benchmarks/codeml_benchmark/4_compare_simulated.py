"""
Stability check: how well do codeml and the NN recover the rho used to simulate each MSA?

Reads sim_truth.csv (step 1), sim_msas/, and codeml_runs_sim/{gene}/mlb (step 2 run with
--tree-dir sim_trees --msa-dir sim_msas --runs-dir codeml_runs_sim).
Writes results_sim.csv and prints bias / RMSE / correlation vs the true rho, overall and by alpha.
"""

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare_real", HERE / "3_compare_real.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

SIM_MSA_DIR = pathlib.Path("sim_msas")
RUNS_DIR = pathlib.Path("codeml_runs_sim")
TRUTH_CSV = pathlib.Path("sim_truth.csv")


def summarize(g):
    out = {"n": len(g)}
    for col in ("codeml_rho", "nn_rho"):
        d = g[col] - g["true_rho"]
        out[f"{col}_bias"] = d.mean()
        out[f"{col}_rmse"] = np.sqrt((d ** 2).mean())
        out[f"{col}_corr"] = np.corrcoef(g[col], g["true_rho"])[0, 1] if len(g) > 2 else np.nan
    return pd.Series(out)


def main():
    model = cr.RhoRegressor()
    model.load_state_dict(torch.load(cr.MODEL_PATH, map_location="cpu"))
    model.eval()

    truth = pd.read_csv(TRUTH_CSV, dtype={"gene": str})
    nn_rho, codeml_rho = [], []
    for gene in truth["gene"]:
        seqs = cr.read_fasta_seqs(SIM_MSA_DIR / f"{gene}.fasta")
        x = cr.calculate_cnn_input(seqs, alphabet_size=cr.ALPHABET_SIZE).unsqueeze(0)
        s = cr.calculate_scalar_features(seqs, alphabet_size=cr.ALPHABET_SIZE).unsqueeze(0)
        with torch.no_grad():
            nn_rho.append(model(x, s).item())
        codeml_rho.append(cr.parse_codeml_rho(RUNS_DIR / gene / "mlb"))
    truth["nn_rho"] = nn_rho
    truth["codeml_rho"] = codeml_rho
    truth.to_csv("results_sim.csv", index=False)

    print(f"codeml failed/unparsed: {truth['codeml_rho'].isna().sum()}/{len(truth)}")
    ok = truth.dropna(subset=["codeml_rho"])
    pd.set_option("display.width", 200)
    print("\nOverall:\n", summarize(ok).round(3).to_string())
    bins = pd.cut(ok["alpha"], [0, 0.3, 0.6, 1.0, 1.5, 10])
    print("\nBy alpha:\n", ok.groupby(bins, observed=True).apply(summarize).round(3).to_string())


if __name__ == "__main__":
    main()
