import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = pd.read_csv(sys.argv[1])
tim = pd.read_csv(sys.argv[2])
d = res.merge(tim, on="simulation", how="inner")
d = d[d.success]
print("rows joined:", len(d), "of", len(res), "(timing rows: %d)" % len(tim))

reg, cod = d.regressor_time, d.time_seconds
print("\n== seconds per alignment ==")
print(pd.DataFrame({"regressor": reg.describe(), "codeml": cod.describe()}).loc[
    ["mean", "50%", "min", "max"]].round(3))
print("\nspeedup (codeml/regressor): mean-based %.0fx | median-based %.0fx"
      % (cod.mean() / reg.mean(), cod.median() / reg.median()))
print("total: regressor %.1f s | codeml %.0f s (%.1f min)" % (reg.sum(), cod.sum(), cod.sum() / 60))
print("\ncodeml min/regressor max ratio: %.1fx" % (cod.min() / reg.max()))

fig, ax = plt.subplots(figsize=(5, 4.5))
b = ax.boxplot([reg, cod], widths=0.5, patch_artist=True, tick_labels=["regressor", "codeml"])
for p, c in zip(b["boxes"], ("#4C72B0", "#DD8452")):
    p.set_facecolor(c)
for m in b["medians"]:
    m.set_color("black")
ax.set_yscale("log")
ax.set_ylabel("seconds per alignment (log)")
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=150)
