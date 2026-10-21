"""Evaluate the saved 1k-trained RhoRegressor on a 2000-site test set (no retraining)."""
import json
import random

import numpy as np
import torch

from archive.old_training.train_rho_regressor_amino_alphaprior import (
    FILE_TAG, MANIFEST_PATH, N_TEST, SPLIT_SEED, TEST_GENE_FRACTION,
    RhoRegressor, evaluate_model, load_or_generate,
)

TRAIN_SIZE = 1000
TEST_SIZE = 2000


def main():
    # same gene split as the training script, so only held-out genes are used
    genes = list(json.loads(MANIFEST_PATH.read_text()).items())
    random.Random(SPLIT_SEED).shuffle(genes)
    test_genes = genes[:int(len(genes) * TEST_GENE_FRACTION)]

    model = RhoRegressor()
    model.load_state_dict(torch.load(f"rho_regressor_{FILE_TAG}_w{TRAIN_SIZE}.pt"))

    test = load_or_generate(test_genes, N_TEST, TEST_SIZE, "test")
    rmse, corr, preds = evaluate_model(model, test["features"], test["scalars"], test["rho"])
    print(f"trained w={TRAIN_SIZE}, tested w={TEST_SIZE}: RMSE={rmse:.4f}, corr={corr:.4f}")
    np.savez(
        f"test_predictions_{FILE_TAG}_trained{TRAIN_SIZE}_tested{TEST_SIZE}.npz",
        true_rho=test["rho"].numpy(),
        true_alpha=test["alpha"].numpy(),
        pred_rho=preds.numpy(),
    )


if __name__ == "__main__":
    main()
