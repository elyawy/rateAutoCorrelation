"""
Check CNN and SBI on high rho: simulate N_SIMS MSAs at fixed rho (held-out test genes, alpha log-uniform),
then compare bias / RMSE of both models and the SBI 90% interval coverage.
Run from the training folder (needs the manifest, msasim and the common modules).
"""

import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import train_rho_regressor_amino_alphaprior as base
from cnn_embedding import CNNScalarEmbedding  # noqa: F401  (needed to unpickle the posterior)

RHO = 0.9
N_SIMS = 500
WINDOW = 250
N_POSTERIOR_SAMPLES = 1000
MAX_SAMPLING_TIME = 10
SIMS_PATH = Path(f"sims_rho{RHO}_w{WINDOW}_n{N_SIMS}.npz")
CNN_WEIGHTS = f"rho_regressor_{base.FILE_TAG}_w{WINDOW}.pt"
POSTERIOR_PATH = "posterior_amino_alphaprior.pkl"


def simulate():
    genes = list(json.loads(base.MANIFEST_PATH.read_text()).items())
    random.Random(base.SPLIT_SEED).shuffle(genes)
    test_genes = genes[:int(len(genes) * base.TEST_GENE_FRACTION)]  # same held-out genes as training

    random.seed(1)
    tasks = [(random.choice(test_genes), base.sample_alpha()) for _ in range(N_SIMS)]
    with ProcessPoolExecutor(max_workers=base.N_WORKERS) as executor:
        futures = [executor.submit(base.simulate_one, gene_entry, RHO, alpha, WINDOW) for (_, gene_entry), alpha in tasks]
        results = [f.result() for f in futures]

    np.savez_compressed(
        SIMS_PATH,
        features=torch.stack([r[0] for r in results]).numpy(),
        scalars=torch.stack([r[1] for r in results]).numpy(),
        alpha=np.array([a for _, a in tasks], dtype=np.float32),
        gene=np.array([g[0] for g, _ in tasks]),
    )


def main():
    if not SIMS_PATH.exists():
        simulate()
    data = np.load(SIMS_PATH)
    features = torch.from_numpy(data["features"]).float()
    scalars = torch.from_numpy(data["scalars"]).float()

    # CNN
    model = base.RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        rho_cnn = model(features, scalars).numpy()

    # SBI: theta = [log(alpha), rho]
    with open(POSTERIOR_PATH, "rb") as f:
        posterior = pickle.load(f)
    x = torch.cat([features.flatten(1), scalars], dim=1)
    rho_sbi = np.full(N_SIMS, np.nan)
    lo = np.full(N_SIMS, np.nan)
    hi = np.full(N_SIMS, np.nan)
    for i in range(N_SIMS):
        try:
            rho_samples = posterior.sample(
                (N_POSTERIOR_SAMPLES,), x=x[i].unsqueeze(0), show_progress_bars=False,
                max_sampling_time=MAX_SAMPLING_TIME,
            )[:, 1].numpy()
        except RuntimeError:
            continue
        rho_sbi[i] = np.median(rho_samples)
        lo[i], hi[i] = np.quantile(rho_samples, [0.05, 0.95])

    ok = ~np.isnan(rho_sbi)
    print(f"true rho = {RHO}, {N_SIMS} sims ({N_SIMS - ok.sum()} SBI timeouts)")
    for name, pred in [("CNN", rho_cnn), ("SBI", rho_sbi)]:
        p = pred[ok]
        print(f"{name}: mean={p.mean():.3f}  bias={p.mean() - RHO:+.3f}  std={p.std():.3f}  RMSE={np.sqrt(np.mean((p - RHO) ** 2)):.3f}")
    covered = (lo[ok] <= RHO) & (RHO <= hi[ok])
    print(f"SBI 90% interval: mean width={np.mean(hi[ok] - lo[ok]):.3f}, covers {RHO} in {covered.mean():.1%} of sims")

    np.savez(f"high_rho_check_rho{RHO}.npz", rho_cnn=rho_cnn, rho_sbi=rho_sbi, lo=lo, hi=hi, alpha=data["alpha"])


if __name__ == "__main__":
    main()
