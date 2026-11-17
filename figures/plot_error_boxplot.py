import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
print("rows:", len(df), "| codeml missing:", df.codeml_rho.isna().sum())

# fair comparison: only rows where both methods have a value
d = df.dropna(subset=["codeml_rho"]).copy()
d["err_reg"] = d.regressor_rho - d.rho
d["err_cod"] = d.codeml_rho - d.rho

def stats(e):
    return pd.Series({"bias": e.mean(), "MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean())})

print("\n== overall (n=%d) ==" % len(d))
print(pd.DataFrame({"regressor": stats(d.err_reg), "codeml": stats(d.err_cod)}).round(4))
print("corr regressor:", round(d.regressor_rho.corr(d.rho), 4),
      "| corr codeml:", round(d.codeml_rho.corr(d.rho), 4))

d["reg_wins"] = d.err_reg.abs() < d.err_cod.abs()
print("\nregressor closer in %d / %d rows (%.0f%%)" % (d.reg_wins.sum(), len(d), 100 * d.reg_wins.mean()))

print("\n== per true rho ==")
rows = []
for r, g in d.groupby("rho"):
    rows.append({"rho": r, "n": len(g),
                 "bias_reg": g.err_reg.mean(), "bias_cod": g.err_cod.mean(),
                 "MAE_reg": g.err_reg.abs().mean(), "MAE_cod": g.err_cod.abs().mean(),
                 "reg_wins%": 100 * g.reg_wins.mean()})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# boxplot
rhos = sorted(d.rho.unique())
fig, ax = plt.subplots(figsize=(7, 4.5))
w = 0.35
for i, r in enumerate(rhos):
    g = d[d.rho == r]
    b1 = ax.boxplot(g.err_reg, positions=[i - w / 2], widths=w * 0.9, patch_artist=True)
    b2 = ax.boxplot(g.err_cod, positions=[i + w / 2], widths=w * 0.9, patch_artist=True)
    for b, c in ((b1, "#4C72B0"), (b2, "#DD8452")):
        for p in b["boxes"]:
            p.set_facecolor(c)
        for m in b["medians"]:
            m.set_color("black")
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_xticks(range(len(rhos)))
ax.set_xticklabels([str(r) for r in rhos])
ax.set_xlabel("true rho")
ax.set_ylabel("estimate - true rho")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#4C72B0"), plt.Rectangle((0, 0), 1, 1, fc="#DD8452")],
          ["regressor", "codeml"])
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
