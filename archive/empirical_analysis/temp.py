import pandas as pd

df = pd.read_csv("results/orthomam_predictions_amino.csv")
df = df[df["status"] == "ok"].dropna(subset=["pred_rho", "pred_rho_sbi"])

T = 0.9
cnn_hi = df["pred_rho"] > T
sbi_hi = df["pred_rho_sbi"] > T
both = df[cnn_hi & sbi_hi]

print(f"genes: {len(df)}")
print(f"CNN > {T}: {cnn_hi.sum()} | SBI > {T}: {sbi_hi.sum()} | both: {len(both)} | only one: {(cnn_hi ^ sbi_hi).sum()}")
print(f"both set: corr={both['pred_rho'].corr(both['pred_rho_sbi']):.3f}, "
      f"mean diff (SBI-CNN)={(both['pred_rho_sbi'] - both['pred_rho']).mean():.3f}")

df = pd.read_csv("results/orthomam_predictions_amino.csv")
df = df[df["status"] == "ok"].dropna(subset=["pred_rho", "pred_rho_sbi"])

for name, d in [("all genes", df), (">500 sites", df[df["alignment_length"] > 500])]:
    print(f"{name}: n={len(d)}, CNN mean={d['pred_rho'].mean():.3f}, SBI mean={d['pred_rho_sbi'].mean():.3f}")


import numpy as np

new = pd.read_csv("results/orthomam_predictions_amino.csv")
new = new[new["status"] == "ok"].dropna(subset=["pred_rho", "pred_rho_sbi"])
new["gene_key"] = new["filename"].str.extract(r"^(\d+)_")[0]

tree = pd.read_csv("manifest_tree_metrics.csv", dtype={"gene_key": str})
df = new.merge(tree, on="gene_key")
df["log10_mean_branch"] = np.log10(df["mean_branch"])

for name, d in [("all manifest genes", df), (">=500 sites (as in the old check)", df[df["alignment_length"] >= 500])]:
    print(f"\n{name}: n={len(d)}, mean CNN={d['pred_rho'].mean():.3f}, SBI={d['pred_rho_sbi'].mean():.3f}")
    for col in ["log10_mean_branch", "frac_zero_branches", "total_length"]:
        print(f"  corr with {col}: CNN {d['pred_rho'].corr(d[col]):+.3f} | SBI {d['pred_rho_sbi'].corr(d[col]):+.3f}")
    q = pd.qcut(d["frac_zero_branches"], 4, duplicates="drop")
    print(d.groupby(q, observed=True)[["pred_rho", "pred_rho_sbi"]].mean().round(3))


import json
import numpy as np
import pandas as pd

manifest = json.load(open("protein_manifest.json"))
alpha = pd.DataFrame({"gene_key": list(manifest), "gamma_alpha": [float(e["gamma_alpha"]) for e in manifest.values()]})

new = pd.read_csv("results/orthomam_predictions_amino.csv")
new = new[new["status"] == "ok"].dropna(subset=["pred_rho", "pred_rho_sbi"])
new["gene_key"] = new["filename"].str.extract(r"^(\d+)_")[0]
df = new.merge(alpha, on="gene_key")

a = df["gamma_alpha"]
print("manifest gamma_alpha percentiles:", {p: round(float(np.percentile(alpha['gamma_alpha'], p)), 3) for p in (1, 5, 25, 50, 75, 95, 99)})
print(f"genes with predictions: {len(df)} | alpha < 0.1: {(a < 0.1).mean():.1%} | 0.1-3.0: {a.between(0.1, 3.0).mean():.1%} | > 3.0: {(a > 3.0).mean():.1%}")

for name, d in [("alpha in prior (0.1-3.0)", df[a.between(0.1, 3.0)]), ("alpha < 0.1", df[a < 0.1])]:
    print(f"{name}: n={len(d)}, mean rho CNN={d['pred_rho'].mean():.3f}, SBI={d['pred_rho_sbi'].mean():.3f}")

d = df[a.between(0.1, 3.0)]
print(f"SBI alpha vs fitted alpha (in-prior genes): log-corr={np.log(d['pred_alpha_sbi']).corr(np.log(d['gamma_alpha'])):.3f}, "
      f"median SBI alpha={d['pred_alpha_sbi'].median():.3f}, median fitted={d['gamma_alpha'].median():.3f}")