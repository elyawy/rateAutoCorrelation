"""
Step 2 (protein version): predict rho on the extracted OrthoMaM chunks with the CNN regressor and the SBI posterior.

Loads the chunk features from step 1, predicts per chunk, and averages per gene.
"""

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import common.config as config

sys.path.insert(0, str(REPO_DIR / "training"))
sys.path.insert(0, str(REPO_DIR / "common"))
from cnn_embedding import CNNScalarEmbedding  # noqa: F401  (needed to unpickle the posterior)

class RhoRegressor(nn.Module):
    """CNN embedding of MSA summary stats -> MLP head -> scalar rho estimate."""
 
    def __init__(self, n_scalars=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + n_scalars, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
 
    def forward(self, x, scalars):
        embedding = self.cnn(x).mean(dim=-1)
        return self.head(torch.cat([embedding, scalars], dim=1)).squeeze(-1)



WINDOW = 250
FEATURES_NAME = f"orthomam_features_amino_w{WINDOW}.npz"
GENE_META_NAME = f"orthomam_gene_meta_amino_w{WINDOW}.csv"
OUTPUT_NAME = "orthomam_predictions_amino.csv"
CNN_WEIGHTS = REPO_DIR / f"rho_regressor_amino_alphaprior_w{WINDOW}.pt"
POSTERIOR_PATH = REPO_DIR / "posterior_amino_alphaprior.pkl"
N_POSTERIOR_SAMPLES = 1000
MAX_SAMPLING_TIME = 2  # seconds per chunk; chunks that exceed it (posterior mostly outside the prior) get NaN

def main():
    data = np.load(config.RESULTS_DIR / FEATURES_NAME)
    features = torch.from_numpy(data["features"]).float()
    scalars = torch.from_numpy(data["scalars"]).float()
    genes = data["gene"]
    print(f"{len(features)} chunks from {len(set(genes))} genes")

    # CNN regressor
    model = RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        rho_cnn = model(features, scalars).numpy()

    # SBI posterior: median per chunk. theta = [log(alpha), rho]
    with open(POSTERIOR_PATH, "rb") as f:
        posterior = pickle.load(f)
    x = torch.cat([features.flatten(1), scalars], dim=1)
    medians = np.zeros((len(x), 2))
    for i in range(len(x)):
        try:
            samples = posterior.sample(
                (N_POSTERIOR_SAMPLES,), x=x[i].unsqueeze(0), show_progress_bars=False,
                max_sampling_time=MAX_SAMPLING_TIME,
            )
            medians[i] = samples.median(dim=0).values.numpy()
        except RuntimeError:
            pass  # timed out: leave NaN
        if (i + 1) % 100 == 0:
            print(f"  posterior {i + 1}/{len(x)}")

    chunks = pd.DataFrame({
        "filename": genes,
        "rho_cnn": rho_cnn,
        "rho_sbi": medians[:, 1],
        "alpha_sbi": np.exp(medians[:, 0]),
    })
    grouped = chunks.groupby("filename")
    per_gene = pd.DataFrame({
        "pred_rho": grouped["rho_cnn"].mean(),
        "rho_std": grouped["rho_cnn"].std(ddof=0),
        "pred_rho_sbi": grouped["rho_sbi"].mean(),
        "rho_sbi_std": grouped["rho_sbi"].std(ddof=0),
        "pred_alpha_sbi": grouped["alpha_sbi"].mean(),
        "n_chunks_sbi_failed": grouped["rho_sbi"].apply(lambda v: int(v.isna().sum())),
    }).reset_index()

    meta = pd.read_csv(config.RESULTS_DIR / GENE_META_NAME)
    result = meta.merge(per_gene, on="filename", how="left")
    output_file = config.RESULTS_DIR / OUTPUT_NAME
    result.to_csv(output_file, index=False)

    print(f"  SBI chunks timed out: {int(np.isnan(medians[:, 0]).sum())}/{len(medians)}")
    ok = result[result["status"] == "ok"]
    print(f"\nResults saved to: {output_file}")
    print(f"  Rho CNN - Mean: {ok['pred_rho'].mean():.3f}, Std: {ok['pred_rho'].std():.3f}")
    print(f"  Rho SBI - Mean: {ok['pred_rho_sbi'].mean():.3f}, Std: {ok['pred_rho_sbi'].std():.3f}")
    print(f"  Corr CNN vs SBI: {ok['pred_rho'].corr(ok['pred_rho_sbi']):.3f}")


if __name__ == "__main__":
    main()
