"""Real-data rho figures: CNN vs codeml on the OrthoMaM genes.

Reads  results/real_data/results_real_new.csv   (gene, nn_rho)
       results/real_data/codeml_rho_raw.csv     (gene, rho)
Writes rho_distribution_histogram.png   histogram of both estimators (all genes)
       rho_cnn_vs_codeml_hexbin.png     CNN vs codeml per gene (genes with a codeml rho)
       results/real_data/summary.csv    n, r, RMSE, bias (CNN - codeml)
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import config

BLUE, ORANGE = "#0072B2", "#E69F00"  # color-blind friendly (Okabe-Ito)

df_cnn = pd.read_csv(config.REAL_CNN_RESULTS).rename(columns={"rho": "nn_rho"})
df_paml = pd.read_csv(config.REAL_CODEML_RESULTS).rename(columns={"rho": "codeml_rho"})
print(f"read {config.REAL_CNN_RESULTS} ({len(df_cnn)} genes)")
print(f"read {config.REAL_CODEML_RESULTS} ({len(df_paml)} genes)")

plt.style.use("seaborn-v0_8-whitegrid")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)


# ---- 1. histogram (standalone) ----
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 1, 31)
for values, color, ls, label in [
    (df_cnn["nn_rho"], BLUE, "-", "Neural network"),
    (df_paml["codeml_rho"], ORANGE, "--", "PAML (codeml)"),
]:
    ax.hist(values, bins=bins, color=color, alpha=0.25)  # light fill
    ax.hist(values, bins=bins, histtype="step", color=color, linestyle=ls, linewidth=2.2, label=label)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("Number of genes")
ax.set_xlim(0, 1)
ax.legend(frameon=False, fontsize=12)
clean_axes(ax)
fig.tight_layout()
fig.savefig(config.RHO_DISTRIBUTION_PLOT, dpi=300)
plt.close(fig)
print(f"wrote {config.RHO_DISTRIBUTION_PLOT}")

# ---- 2. agreement, per gene ----
df = df_cnn.merge(df_paml, on="gene").dropna(subset=["nn_rho", "codeml_rho"])
x, y = df["codeml_rho"].to_numpy(), df["nn_rho"].to_numpy()
n = len(df)
r = float(np.corrcoef(x, y)[0, 1])
rmse = float(np.sqrt(np.mean((y - x) ** 2)))
bias = float(np.mean(y - x))
print(f"n={n} r={r:.3f} rmse={rmse:.3f} bias(CNN-codeml)={bias:+.3f}")

pd.DataFrame([{"n": n, "r": r, "rmse": rmse, "bias_cnn_minus_codeml": bias}]).to_csv(
    config.REAL_SUMMARY_CSV, index=False, float_format="%.4f")
print(f"wrote {config.REAL_SUMMARY_CSV}")

fig, ax = plt.subplots(figsize=(7, 6))
hb = ax.hexbin(x, y, gridsize=50, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap="viridis")
ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel(r"codeml $\rho$")
ax.set_ylabel(r"CNN $\rho$")
ax.text(0.04, 0.96, f"n = {n:,}\nr = {r:.3f}\nRMSE = {rmse:.3f}\nbias = {bias:+.3f}",
        transform=ax.transAxes, va="top", fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))
cb = fig.colorbar(hb, ax=ax, shrink=0.8)
cb.set_label("Genes per bin (log scale)", fontsize=12)
clean_axes(ax)
fig.tight_layout()
fig.savefig(config.RHO_HEXBIN_PLOT, dpi=300)
plt.close(fig)
print(f"wrote {config.RHO_HEXBIN_PLOT}")