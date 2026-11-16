"""
Train the CNN+MLP rho regressor on simulations made by generate_sims_amino_indel_v2.py.

For each window size in WINDOW_SIZES, trains a separate regressor to predict rho from MSA summary features,
then evaluates it on held-out genes at every length in TEST_SIZES.
Needs only the sims_*.npz files (copy them next to where this runs), torch, numpy and common/model.py.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


from common.model import RhoRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keep these in sync with generate_sims_amino_indel_v2.py (they decide which files are read)
WINDOW_SIZES = [1000]
TEST_SIZES = [100, 250, 500, 1000]  # lengths to evaluate the trained model on
N_TRAIN = 20000
N_TEST = 1000
FILE_TAG = "amino_indel_trees500_gap"


def load_sims(window_size: int, n_samples: int, split: str) -> dict:
    """Load a dataset saved by the generation script."""
    path = Path(f"sims_{FILE_TAG}_w{window_size}_n{n_samples}_{split}.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run generate_sims_amino_indel_v2.py and copy its output here")
    print(f"  loading simulations from {path}")
    saved = np.load(path)
    data = {k: torch.from_numpy(saved[k]) for k in ("features", "scalars", "rho", "alpha")}
    data["gene"] = saved["gene"]
    return data


def train_model(train_features, train_scalars, train_rhos, test_features, test_scalars, test_rhos, n_epochs=100, lr=1e-3, batch_size=64):
    model = RhoRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, test_losses = [], []

    # train set lives on the GPU for the whole run (test set stays on CPU, predict() moves chunks)
    train_features, train_scalars, train_rhos = (t.to(DEVICE) for t in (train_features, train_scalars, train_rhos))

    n_samples = train_features.shape[0]
    for epoch in range(n_epochs):
        permutation = torch.randperm(n_samples, device=DEVICE)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = permutation[start:start + batch_size]
            batch_features = train_features[idx]
            batch_scalars = train_scalars[idx]
            batch_rhos = train_rhos[idx]

            optimizer.zero_grad()
            preds = model(batch_features, batch_scalars)
            loss = loss_fn(preds, batch_rhos)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        test_loss = loss_fn(predict(model, test_features, test_scalars), test_rhos).item()
        train_losses.append(epoch_loss / n_samples)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} - train loss: {train_losses[-1]:.5f} - test loss: {test_loss:.5f}")

    return model, train_losses, test_losses


def predict(model, features, scalars, chunk=250):
    """Forward pass in chunks (the 2GB GPU can't take a big test set at once); predictions come back on CPU."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(features), chunk):
            preds.append(model(features[i:i + chunk].to(DEVICE), scalars[i:i + chunk].to(DEVICE)).cpu())
    return torch.cat(preds)


def evaluate_model(model, test_features, test_scalars, test_rhos):
    model.to(DEVICE).eval()
    preds = predict(model, test_features, test_scalars)
    rmse = torch.sqrt(torch.mean((preds - test_rhos) ** 2)).item()
    corr = np.corrcoef(preds.numpy(), test_rhos.numpy())[0, 1]
    return rmse, corr, preds


def main():
    for window_size in WINDOW_SIZES:
        print(f"\n=== window_size={window_size} ===")
        train = load_sims(window_size, N_TRAIN, "train")
        test = load_sims(window_size, N_TEST, "test")

        model, train_losses, test_losses = train_model(
            train["features"], train["scalars"], train["rho"],
            test["features"], test["scalars"], test["rho"],
        )
        rmse, corr, preds = evaluate_model(model, test["features"], test["scalars"], test["rho"])
        print(f"window={window_size}: RMSE={rmse:.4f}, corr={corr:.4f}")

        torch.save(model.cpu().state_dict(), f"rho_regressor_{FILE_TAG}_w{window_size}.pt")  # CPU weights load anywhere
        np.savez(
            f"test_predictions_{FILE_TAG}_w{window_size}.npz",
            true_rho=test["rho"].numpy(),
            true_alpha=test["alpha"].numpy(),
            pred_rho=preds.numpy(),
            train_loss=np.array(train_losses),
            test_loss=np.array(test_losses),
        )

        # length-transfer check: same trained model, held-out genes, other window sizes
        for test_size in TEST_SIZES:
            size_test = load_sims(test_size, N_TEST, "test")
            rmse, corr, size_preds = evaluate_model(model, size_test["features"], size_test["scalars"], size_test["rho"])
            print(f"  trained w={window_size}, tested w={test_size}: RMSE={rmse:.4f}, corr={corr:.4f}")
            np.savez(
                f"test_predictions_{FILE_TAG}_trained{window_size}_tested{test_size}.npz",
                true_rho=size_test["rho"].numpy(),
                true_alpha=size_test["alpha"].numpy(),
                pred_rho=size_preds.numpy(),
            )


if __name__ == "__main__":
    main()