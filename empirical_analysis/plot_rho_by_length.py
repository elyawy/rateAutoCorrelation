"""Predicted rho by alignment length: median and 25-75% band per length bin."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/orthomam_predictions_amino.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "results/plots/rho_by_length.png"

BIN_EDGES = [100, 250, 500, 1000, 2000, np.inf]
BIN_LABELS = ["100-250", "250-500", "500-1k", "1k-2k", "2k+"]
N_LESS_RELIABLE = 2  # first two bins (< 500 sites) are shaded grey

SERIES = "#2a6fb0"      # one series, one hue
INK = "#1f2933"         # text and axes
MUTED = "#6b7785"       # secondary text, grid
SHADE = "#e4e7eb"       # "less reliable" region
SURFACE = "#ffffff"

df = pd.read_csv(CSV_PATH).dropna(subset=["pred_rho"])
df["bin"] = pd.cut(df["alignment_length"], bins=BIN_EDGES, labels=BIN_LABELS, right=False)
g = df.groupby("bin", observed=True)["pred_rho"]
stats = pd.DataFrame({"n": g.size(), "q25": g.quantile(0.25), "median": g.median(), "q75": g.quantile(0.75)})
x = np.arange(len(stats))

fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.axvspan(-0.5, N_LESS_RELIABLE - 0.5, color=SHADE, zorder=0)
ax.text(N_LESS_RELIABLE / 2 - 0.5, 0.97, "less reliable (< 500 sites)", ha="center", va="top", color=MUTED, fontsize=10)

ax.fill_between(x, stats["q25"], stats["q75"], color=SERIES, alpha=0.2, linewidth=0, zorder=2)
ax.plot(x, stats["median"], color=SERIES, linewidth=2, zorder=3)
ax.plot(x, stats["median"], "o", color=SERIES, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

for xi, med in zip(x, stats["median"]):
    ax.annotate(f"{med:.2f}", (xi, med), textcoords="offset points", xytext=(0, 12), ha="center", color=INK, fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f"{label}\nn = {n:,}" for label, n in zip(stats.index, stats["n"])], color=INK, fontsize=10)
ax.set_xlim(-0.5, len(stats) - 0.5)
ax.set_ylim(0, 1)
ax.set_xlabel("Alignment length (sites)", color=INK, fontsize=11)
ax.set_ylabel("Predicted rho", color=INK, fontsize=11)
ax.set_title("Predicted rho by alignment length (OrthoMaM): median and 25-75% range", color=INK, fontsize=12, loc="left")
ax.tick_params(axis="y", colors=INK, labelsize=10)
ax.yaxis.grid(True, color=MUTED, alpha=0.25, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, facecolor=SURFACE)
print(stats.round(3).to_string())
print(f"saved {OUT_PATH}")
