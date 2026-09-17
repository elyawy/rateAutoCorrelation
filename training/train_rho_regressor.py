"""
Point-estimate regression for rho (site rate correlation) using OrthoMaM gene models.

For each window size in WINDOW_SIZES, trains a separate CNN+MLP regressor to predict
rho from MSA summary features. Simulations use real OrthoMaM trees, GTR models and
gamma_alpha per gene (all fixed, taken from the manifest); only rho is sampled per
simulation, drawn independently from prior.RHO_RANGE.
"""

import json
import random
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
from features_calculator import calculate_cnn_input

MANIFEST_PATH = Path("/home/elyalab/Dev/failed_syncs/SailfishBenchmark/simulation_manifest.json")
WINDOW_SIZES = [100, 250, 500]
N_TRAIN = 5000
N_TEST = 500
N_WORKERS = 4
GAMMA_CATEGORIES = 8
 
BASE_ORDER = ["A", "C", "G", "T"]
GTR_PAIRS = ["AC", "AG", "AT", "CG", "CT", "GT"]
 
# indel rates based on mammalian rates from https://doi.org/10.1093/bioinformatics/btaf686
INSERTION_RATE = 0.007
DELETION_RATE = 0.035
 
 
def exchangeability_to_q(freqs: dict, gtr_matrix: dict) -> list:
    """Convert OrthoMaM/IQ-TREE exchangeability rates (r_ij) to msasim's
    forward transition rates (q_ij = r_ij * freq_target)."""
    q = []
    for pair in GTR_PAIRS:
        target_base = pair[1]
        r_ij = gtr_matrix[pair]
        q.append(r_ij * freqs[target_base])
    return q
 
 
def build_replacement_model(gtr_model: dict, rho: float) -> ReplacementModelSpec:
    """GTR + DNA replacement model for one gene, with its fixed alpha and a sampled rho."""
    freqs = gtr_model["base_freq"]
    freq_list = [freqs[b] for b in BASE_ORDER]
    q_rates = exchangeability_to_q(freqs, gtr_model["gtr_matrix"])
 
    site_rate_model = SiteRateModelSpec(
        gamma_alpha=gtr_model["gamma_alpha"],
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
 
    return ReplacementModelSpec(
        model=MODEL_CODES.GTR,
        alphabet=ALPHABET_CODES.DNA,
        model_parameters=freq_list + q_rates,
        site_rate_model=site_rate_model,
    )
 
 
def simulate_one(gene_entry: dict, rho: float, window_size: int) -> torch.Tensor:
    """Simulate one MSA for a gene at a given rho and window size, return CNN input."""
    replacement_model = build_replacement_model(gene_entry["gtr_model"], rho)
 
    simulation_protocol = SimProtocol(
        tree=gene_entry["newick"],
        root_seq_size=window_size,
        insertion_rate=INSERTION_RATE,
        deletion_rate=DELETION_RATE,
    )
    simulation_protocol.set_insertion_length_distributions(ZipfDistribution(p=1.53, truncation=50))
    simulation_protocol.set_deletion_length_distributions(ZipfDistribution(p=1.11, truncation=50))
 
    simulator = Simulator(simulation_protocol, replacement_model=replacement_model)
    msa = simulator()
 
    sequences = [""] * msa.get_num_sequences()
    for i in range(msa.get_num_sequences()):
        row = msa.get_msa_row(i).split("\n")[1][:window_size]
        if len(row) < window_size:
            row += "-" * (window_size - len(row))
        sequences[i] = row
 
    return calculate_cnn_input(sequences, alphabet_size=4)
 
 
def generate_dataset(gene_entries: list, n_samples: int, window_size: int):
    """Sample (gene, rho) pairs, simulate in parallel, return stacked features + rho tensor."""
    tasks = [
        (random.choice(gene_entries), random.uniform(*prior.RHO_RANGE))
        for _ in range(n_samples)
    ]
 
    features_list = [None] * n_samples
    rhos = [None] * n_samples
 
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_idx = {
            executor.submit(simulate_one, gene_entry, rho, window_size): idx
            for idx, (gene_entry, rho) in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            features_list[idx] = future.result()
            rhos[idx] = tasks[idx][1]
            completed += 1
            if completed % 100 == 0:
                print(f"  simulated {completed}/{n_samples}")
 
    return torch.stack(features_list), torch.tensor(rhos, dtype=torch.float32)
 
 
class RhoRegressor(nn.Module):
    """CNN embedding of MSA summary stats -> MLP head -> scalar rho estimate."""
 
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
 
    def forward(self, x):
        embedding = self.cnn(x).mean(dim=-1)
        return self.head(embedding).squeeze(-1)
 
 
def train_model(train_features, train_rhos, n_epochs=100, lr=1e-3, batch_size=64):
    model = RhoRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
 
    n_samples = train_features.shape[0]
    for epoch in range(n_epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = permutation[start:start + batch_size]
            batch_features = train_features[idx]
            batch_rhos = train_rhos[idx]
 
            optimizer.zero_grad()
            preds = model(batch_features)
            loss = loss_fn(preds, batch_rhos)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
 
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} - loss: {epoch_loss / n_samples:.5f}")
 
    return model
 
 
def evaluate_model(model, test_features, test_rhos):
    model.eval()
    with torch.no_grad():
        preds = model(test_features)
    rmse = torch.sqrt(torch.mean((preds - test_rhos) ** 2)).item()
    corr = np.corrcoef(preds.numpy(), test_rhos.numpy())[0, 1]
    return rmse, corr, preds
 
 
def main():
    gene_entries = list(json.loads(MANIFEST_PATH.read_text()).values())
 
    for window_size in WINDOW_SIZES:
        print(f"\n=== window_size={window_size} ===")
        train_features, train_rhos = generate_dataset(gene_entries, N_TRAIN, window_size)
        test_features, test_rhos = generate_dataset(gene_entries, N_TEST, window_size)
 
        model = train_model(train_features, train_rhos)
        rmse, corr, preds = evaluate_model(model, test_features, test_rhos)
        print(f"window={window_size}: RMSE={rmse:.4f}, corr={corr:.4f}")
 
        torch.save(model.state_dict(), f"rho_regressor_w{window_size}.pt")
        np.savez(
            f"test_predictions_w{window_size}.npz",
            true_rho=test_rhos.numpy(),
            pred_rho=preds.numpy(),
        )
 
 
if __name__ == "__main__":
    main()
 
