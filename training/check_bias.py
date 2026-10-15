"""Bias and pred-vs-true plots per test length, from the saved test prediction files."""
import numpy as np
import matplotlib.pyplot as plt

FILE_TAG = "amino_alphaprior"
TRAIN_SIZE = 1000
TEST_SIZES = [100, 250, 500, 1000, 2000]
BINS = [0.0, 0.33, 0.66, 0.99]  # low / mid / high rho

fig, axes = plt.subplots(1, len(TEST_SIZES), figsize=(4 * len(TEST_SIZES), 4), sharex=True, sharey=True)

for ax, size in zip(axes, TEST_SIZES):
    d = np.load(f"test_predictions_{FILE_TAG}_trained{TRAIN_SIZE}_tested{size}.npz")
    true, pred = d["true_rho"], d["pred_rho"]
    err = pred - true
    print(f"w={size}: bias={err.mean():+.4f}  RMSE={np.sqrt((err ** 2).mean()):.4f}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (true >= lo) & (true <= hi)
        print(f"    true rho in [{lo:.2f}, {hi:.2f}]: bias={err[m].mean():+.4f}  n={m.sum()}")

    ax.scatter(true, pred, s=5, alpha=0.4)
    ax.plot([0, 0.99], [0, 0.99], "k--", lw=1)
    ax.set_title(f"w={size}")
    ax.set_xlabel("true rho")
axes[0].set_ylabel("predicted rho")
plt.tight_layout()
plt.savefig(f"pred_vs_true_{FILE_TAG}.png", dpi=150)
print(f"saved pred_vs_true_{FILE_TAG}.png")