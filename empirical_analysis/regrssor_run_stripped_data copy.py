import re, sys, pathlib
import numpy as np, pandas as pd, torch
from torch import nn


from common.features_calculator import calculate_cnn_input, calculate_scalar_features
from common.model import RhoRegressor
SAMPLE = "data/random_sample_200.csv"
STATS = "data/real_strip_stats.csv"
RUNS = {"gapped": "data/no_sync/codeml_runs_real_gapped", "stripped": "data/no_sync/codeml_runs_real_stripped"}
MSAS = {"gapped": "data/no_sync/real_gapped", "stripped": "data/no_sync/real_stripped"}
WEIGHTS = "models/rho_regressor_amino_indel_trees500_gap_w1000.pt"
OUT = "data/results_real_stripped_new.csv"
CAP = 0.989

def codeml_rho(run, g):
    p = pathlib.Path(run) / g / "mlb"
    if not p.exists():
        return np.nan
    m = re.search(r"rho \(correlation\)\s*=\s*([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else np.nan

def zero_branch_frac(g):
    t = (pathlib.Path(RUNS["gapped"]) / g / "tree.nwk").read_text()
    bl = [float(x) for x in re.findall(r":([0-9.eE-]+)", t)]
    return float(np.mean([b <= 1e-5 for b in bl]))

def read_fasta(path):
    seqs, cur = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    seqs.append("".join(cur))
    return seqs

model = RhoRegressor()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

def nn_rho(cond, g):
    seqs = read_fasta(f"{MSAS[cond]}/{g}.fasta")
    x = calculate_cnn_input(seqs, alphabet_size=20).float().unsqueeze(0)
    s = calculate_scalar_features(seqs, alphabet_size=20).float().unsqueeze(0)
    with torch.no_grad():
        return model(x, s).item()

# table
df = pd.read_csv(SAMPLE, dtype={"gene": str})
df = df.merge(pd.read_csv(STATS, dtype={"gene": str}), on="gene")
for c in ("gapped", "stripped"):
    df[f"rho_{c}"] = [codeml_rho(RUNS[c], g) for g in df.gene]
    df[f"nn_rho_{c}"] = [nn_rho(c, g) for g in df.gene]
df["shift"] = df.rho_stripped - df.rho_gapped
df["capped"] = (df.rho_gapped >= CAP) | (df.rho_stripped >= CAP)
df["zero_branch_frac"] = [zero_branch_frac(g) for g in df.gene]
df.to_csv(OUT, index=False)

ok = df.dropna(subset=["shift"])
print(f"{len(ok)} genes with both rho, {int(ok.capped.sum())} capped, {df.shift.isna().sum()} missing")
print("mean shift (uncapped):", round(ok[~ok.capped].shift.mean(), 4))