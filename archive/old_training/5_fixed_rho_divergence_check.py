"""
Does predicted rho depend on tree divergence when the TRUE rho is constant?
Simulate N_SIMS MSAs at a fixed rho on random manifest genes (their real trees), predict with the CNN and the SBI
posterior, and relate the prediction (and bias) to each tree's zero-branch share / mean branch length.
Run from the training folder (needs the manifest, msasim and the common modules).
"""

import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import archive.old_training.train_rho_regressor_amino_alphaprior as base
from cnn_embedding import CNNScalarEmbedding  # noqa: F401  (needed to unpickle the posterior)

RHO = 0.5
N_SIMS = 2000
WINDOW = 250
N_POSTERIOR_SAMPLES = 1000
MAX_SAMPLING_TIME = 10
TREE_METRICS_CSV = "manifest_tree_metrics.csv"
SIMS_PATH = Path(f"sims_fixedrho{RHO}_w{WINDOW}_n{N_SIMS}.npz")
CNN_WEIGHTS = f"rho_regressor_{base.FILE_TAG}_w{WINDOW}.pt"
POSTERIOR_PATH = "posterior_amino_alphaprior.pkl"


def simulate():
    genes = list(json.loads(base.MANIFEST_PATH.read_text()).items())
    random.seed(1)
    picked = random.sample(genes, min(N_SIMS, len(genes)))
    alphas = [base.sample_alpha() for _ in picked]
    with ProcessPoolExecutor(max_workers=base.N_WORKERS) as executor:
        futures = [executor.submit(base.simulate_one, entry, RHO, a, WINDOW) for (_, entry), a in zip(picked, alphas)]
        results = [f.result() for f in futures]
    np.savez_compressed(
        SIMS_PATH,
        features=torch.stack([r[0] for r in results]).numpy(),
        scalars=torch.stack([r[1] for r in results]).numpy(),
        alpha=np.array(alphas, dtype=np.float32),
        gene_key=np.array([k for k, _ in picked]),
    )


def main():
    if not SIMS_PATH.exists():
        simulate()
    data = np.load(SIMS_PATH)
    features = torch.from_numpy(data["features"]).float()
    scalars = torch.from_numpy(data["scalars"]).float()
    n = len(features)

    model = base.RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        rho_cnn = model(features, scalars).numpy()

    with open(POSTERIOR_PATH, "rb") as f:
        posterior = pickle.load(f)
    x = torch.cat([features.flatten(1), scalars], dim=1)
    rho_sbi = np.full(n, np.nan)
    for i in range(n):
        try:
            s = posterior.sample((N_POSTERIOR_SAMPLES,), x=x[i].unsqueeze(0), show_progress_bars=False,
                                 max_sampling_time=MAX_SAMPLING_TIME)
            rho_sbi[i] = s[:, 1].median().item()
        except RuntimeError:
            pass
        if (i + 1) % 250 == 0:
            print(f"  posterior {i + 1}/{n}")

    df = pd.DataFrame({"gene_key": data["gene_key"].astype(str), "rho_cnn": rho_cnn, "rho_sbi": rho_sbi, "alpha": data["alpha"]})
    tree = pd.read_csv(TREE_METRICS_CSV, dtype={"gene_key": str})
    df = df.merge(tree, on="gene_key").dropna()
    df["log10_mean_branch"] = np.log10(df["mean_branch"])
    df.to_csv(f"fixed_rho{RHO}_divergence_check.csv", index=False)

    print(f"\ntrue rho = {RHO}, n={len(df)}")
    print(f"CNN: mean={df['rho_cnn'].mean():.3f} bias={df['rho_cnn'].mean() - RHO:+.3f} | "
          f"SBI: mean={df['rho_sbi'].mean():.3f} bias={df['rho_sbi'].mean() - RHO:+.3f}")
    for col in ["log10_mean_branch", "frac_zero_branches", "total_length", "alpha"]:
        print(f"  corr with {col}: CNN {df['rho_cnn'].corr(df[col]):+.3f} | SBI {df['rho_sbi'].corr(df[col]):+.3f}")
    q = pd.qcut(df["frac_zero_branches"], 4, duplicates="drop")
    print("\nmean predicted rho by quartile of zero-branch share (true rho constant):")
    print(df.groupby(q, observed=True)[["rho_cnn", "rho_sbi"]].mean().round(3))


if __name__ == "__main__":
    main()
