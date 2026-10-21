"""
Step 2 (protein version): predict rho on the full OrthoMaM alignments with the CNN regressor.

Loads the per-alignment features from step 1 and predicts one rho per alignment, at its own length.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import common.config as config


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


FEATURES_NAME = "orthomam_features_amino_full.npz"
GENE_META_NAME = "orthomam_gene_meta_amino_full.csv"
OUTPUT_NAME = "orthomam_predictions_amino.csv"
CNN_WEIGHTS = REPO_DIR / "rho_regressor_amino_alphaprior_w1000.pt"  # trained on 1000-site alignments


def main():
    data = np.load(config.RESULTS_DIR / FEATURES_NAME)
    lengths = data["lengths"]
    # split the concatenated (2, total sites) array back into one (2, length) array per alignment
    features = np.split(data["features"], np.cumsum(lengths)[:-1], axis=1)
    scalars = torch.from_numpy(data["scalars"]).float()
    genes = data["gene"]
    print(f"{len(features)} alignments, lengths {lengths.min()} to {lengths.max()}")

    model = RhoRegressor()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()

    # one alignment at a time, since lengths differ
    rho = np.zeros(len(features))
    with torch.no_grad():
        for i, feat in enumerate(features):
            x = torch.from_numpy(feat).float().unsqueeze(0)
            rho[i] = model(x, scalars[i].unsqueeze(0)).item()
            if (i + 1) % 1000 == 0:
                print(f"  predicted {i + 1}/{len(features)}")

    meta = pd.read_csv(config.RESULTS_DIR / GENE_META_NAME)
    result = meta.merge(pd.DataFrame({"filename": genes, "pred_rho": rho}), on="filename", how="left")
    output_file = config.RESULTS_DIR / OUTPUT_NAME
    result.to_csv(output_file, index=False)

    ok = result[result["status"] == "ok"]
    print(f"\nResults saved to: {output_file}")
    print(f"  Rho CNN - Mean: {ok['pred_rho'].mean():.3f}, Std: {ok['pred_rho'].std():.3f}")


if __name__ == "__main__":
    main()