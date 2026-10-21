"""
Point-estimate regression for rho (site rate correlation) using OrthoMaM gene models.

For each window size in WINDOW_SIZES, trains a separate CNN+MLP regressor to predict
rho from MSA summary features. Simulations use LG on real OrthoMaM topologies with protein
branch lengths per gene (protein_estimates.csv) and each gene's own inferred indel model
(orthomam_indel_params.csv). Rho and gamma alpha are both sampled per simulation: rho from
prior.RHO_RANGE, alpha log-uniform from ALPHA_RANGE.
Simulated datasets are saved to disk and reloaded on later runs, so other architectures can be
trained without re-simulating.
"""

import csv
import json
import math
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from msasim import (
    ALPHABET_CODES,
    MODEL_CODES,
    ReplacementModelSpec,
    SimProtocol,
    Simulator,
    SiteRateModelSpec,
)
from msasim.constants import SITE_RATE_MODELS
from msasim.distributions import ZipfDistribution
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))

import prior
from features_calculator import calculate_cnn_input, calculate_scalar_features

ESTIMATES_CSV = Path("protein_estimates.csv")  # gene, newick
INDEL_CSV = Path("orthomam_indel_params.csv")  # per-gene indel model
WINDOW_SIZES = [1000]
TEST_SIZES = [100, 250, 500, 1000]  # lengths to evaluate the trained model on
N_TRAIN = 20000
N_TEST = 1000
TEST_GENE_FRACTION = 0.1  # genes held out for the test set only
SPLIT_SEED = 0
N_WORKERS = 7
ALPHA_RANGE = (0.1, 3.0)  # log-uniform
FILE_TAG = "amino_indel"
GAMMA_CATEGORIES = 8
 
ALPHABET_SIZE = 20
 
MAX_INDEL_LENGTH = 150  # truncation of the indel length distributions (indel-model prior)


def load_genes() -> list:
    """[(gene, {newick, ins_rate, del_rate, len_ins, len_del, root})] for genes with both a tree and an indel model."""
    indel = {row["name"].removesuffix(".fasta"): row for row in csv.DictReader(open(INDEL_CSV))}
    genes = []
    for row in csv.DictReader(open(ESTIMATES_CSV)):
        ind = indel.get(row["gene"])
        if ind is None:
            continue
        genes.append((row["gene"], {
            "newick": re.sub(r"\d+\.\d+:", ":", row["newick"]),  # drop support values
            **{k: float(ind[k]) for k in ("ins_rate", "del_rate", "len_ins", "len_del", "root")},
        }))
    return genes
 
 
def build_replacement_model(gamma_alpha: float, rho: float) -> ReplacementModelSpec:
    """LG protein replacement model for one gene, with its fixed alpha and a sampled rho."""
    site_rate_model = SiteRateModelSpec(
        gamma_alpha=gamma_alpha,
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
 
    return ReplacementModelSpec(
        model=MODEL_CODES.LG,
        alphabet=ALPHABET_CODES.PROTEIN,
        site_rate_model=site_rate_model,
    )
 
 
def simulate_one(gene_entry: dict, rho: float, alpha: float, window_size: int):
    """Simulate one MSA for a gene at a given rho and window size, return (CNN input, scalar features)."""
    replacement_model = build_replacement_model(alpha, rho)
 
    simulation_protocol = SimProtocol(
        tree=gene_entry["newick"],
        # the gene's own root length, but never below the window, so the cropped window has no padding
        root_seq_size=max(int(round(gene_entry["root"])), window_size),
        insertion_rate=gene_entry["ins_rate"],
        deletion_rate=gene_entry["del_rate"],
    )
    simulation_protocol.set_insertion_length_distributions(
        ZipfDistribution(p=gene_entry["len_ins"], truncation=MAX_INDEL_LENGTH))
    simulation_protocol.set_deletion_length_distributions(
        ZipfDistribution(p=gene_entry["len_del"], truncation=MAX_INDEL_LENGTH))
 
    simulator = Simulator(simulation_protocol, replacement_model=replacement_model)
    msa = simulator()
 
    sequences = [""] * msa.get_num_sequences()
    for i in range(msa.get_num_sequences()):
        row = msa.get_msa_row(i).split("\n")[1][:window_size]
        if len(row) < window_size:
            row += "-" * (window_size - len(row))
        sequences[i] = row
 
    return (
        calculate_cnn_input(sequences, alphabet_size=ALPHABET_SIZE),
        calculate_scalar_features(sequences, alphabet_size=ALPHABET_SIZE),
    )
 
 
def sample_alpha() -> float:
    return math.exp(random.uniform(math.log(ALPHA_RANGE[0]), math.log(ALPHA_RANGE[1])))
 
 
def generate_dataset(genes: list, n_samples: int, window_size: int) -> dict:
    """Sample (gene, rho, alpha), simulate in parallel, return features, scalars and parameters.
 
    `genes` is a list of (gene_key, gene_entry) pairs.
    """
    tasks = [
        (random.choice(genes), random.uniform(*prior.RHO_RANGE), sample_alpha())
        for _ in range(n_samples)
    ]
 
    features_list = [None] * n_samples
    scalars_list = [None] * n_samples
 
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_idx = {
            executor.submit(simulate_one, gene_entry, rho, alpha, window_size): idx
            for idx, ((_, gene_entry), rho, alpha) in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            features_list[idx], scalars_list[idx] = future.result()
            completed += 1
            if completed % 100 == 0:
                print(f"  simulated {completed}/{n_samples}")
 
    return {
        "features": torch.stack(features_list),
        "scalars": torch.stack(scalars_list),
        "rho": torch.tensor([t[1] for t in tasks], dtype=torch.float32),
        "alpha": torch.tensor([t[2] for t in tasks], dtype=torch.float32),
        "gene": np.array([t[0][0] for t in tasks]),
    }
 
 
def load_or_generate(genes: list, n_samples: int, window_size: int, split: str) -> dict:
    """Simulate a dataset once and save it; later runs reload it instead of re-simulating."""
    path = Path(f"sims_{FILE_TAG}_w{window_size}_n{n_samples}_{split}.npz")
    if path.exists():
        print(f"  loading saved simulations from {path}")
        saved = np.load(path)
        data = {k: torch.from_numpy(saved[k]) for k in ("features", "scalars", "rho", "alpha")}
        data["gene"] = saved["gene"]
        return data
 
    data = generate_dataset(genes, n_samples, window_size)
    meta = {"window_size": window_size, "alpha_range": ALPHA_RANGE, "rho_range": list(prior.RHO_RANGE),
            "estimates": str(ESTIMATES_CSV), "indel_params": str(INDEL_CSV), "split_seed": SPLIT_SEED, "split": split}
    np.savez_compressed(
        path,
        features=data["features"].numpy(),
        scalars=data["scalars"].numpy(),
        rho=data["rho"].numpy(),
        alpha=data["alpha"].numpy(),
        gene=data["gene"],
        meta=json.dumps(meta),
    )
    print(f"  saved simulations to {path}")
    return data
 
 
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
 
 
def train_model(train_features, train_scalars, train_rhos, test_features, test_scalars, test_rhos, n_epochs=100, lr=1e-3, batch_size=64):
    model = RhoRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, test_losses = [], []
 
    n_samples = train_features.shape[0]
    for epoch in range(n_epochs):
        permutation = torch.randperm(n_samples)
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
 
        with torch.no_grad():
            test_loss = loss_fn(model(test_features, test_scalars), test_rhos).item()
        train_losses.append(epoch_loss / n_samples)
        test_losses.append(test_loss)
 
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} - train loss: {train_losses[-1]:.5f} - test loss: {test_loss:.5f}")
 
    return model, train_losses, test_losses
 
 
def evaluate_model(model, test_features, test_scalars, test_rhos):
    model.eval()
    with torch.no_grad():
        preds = model(test_features, test_scalars)
    rmse = torch.sqrt(torch.mean((preds - test_rhos) ** 2)).item()
    corr = np.corrcoef(preds.numpy(), test_rhos.numpy())[0, 1]
    return rmse, corr, preds
 
 
def main():
    genes = load_genes()
 
    # split genes once (same split for every window) so test genes are never seen in training
    random.Random(SPLIT_SEED).shuffle(genes)
    n_test_genes = int(len(genes) * TEST_GENE_FRACTION)
    test_genes = genes[:n_test_genes]
    train_genes = genes[n_test_genes:]
    print(f"genes: {len(train_genes)} train, {len(test_genes)} test")
 
    for window_size in WINDOW_SIZES:
        print(f"\n=== window_size={window_size} ===")
        train = load_or_generate(train_genes, N_TRAIN, window_size, "train")
        test = load_or_generate(test_genes, N_TEST, window_size, "test")
 
        model, train_losses, test_losses = train_model(
            train["features"], train["scalars"], train["rho"],
            test["features"], test["scalars"], test["rho"],
        )
        rmse, corr, preds = evaluate_model(model, test["features"], test["scalars"], test["rho"])
        print(f"window={window_size}: RMSE={rmse:.4f}, corr={corr:.4f}")
 
        torch.save(model.state_dict(), f"rho_regressor_{FILE_TAG}_w{window_size}.pt")
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
            size_test = load_or_generate(test_genes, N_TEST, test_size, "test")
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