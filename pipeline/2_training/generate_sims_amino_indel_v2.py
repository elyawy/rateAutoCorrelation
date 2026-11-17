"""
Simulate the training / test data for the rho regressor (no training here, see
train_rho_regressor_amino_indel_v2.py).

Simulations use LG on real OrthoMaM topologies with protein branch lengths per gene
(protein_estimates.csv) and each gene's own inferred indel model (orthomam_indel_params.csv).
Rho and gamma alpha are both sampled per simulation: rho from RHO_RANGE, alpha log-uniform from ALPHA_RANGE.
Each dataset is saved to sims_{FILE_TAG}_w{window}_n{n}_{split}.npz; files that already exist are skipped.
Copy those files to the machine that trains.
"""

import json
import math
import random

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

import config
from common.features_calculator import calculate_cnn_input, calculate_scalar_features
from common.genes import held_out_genes, load_genes

ESTIMATES_CSV = config.TRAIN_ESTIMATES_CSV
INDEL_CSV = config.INDEL_PARAMS_CSV
WINDOW_SIZES = config.WINDOW_SIZES
TEST_SIZES = config.TEST_SIZES
N_TRAIN = config.N_TRAIN
N_TEST = config.N_TEST
TEST_GENE_FRACTION = config.TEST_GENE_FRACTION
SPLIT_SEED = config.SPLIT_SEED
N_WORKERS = 8
ALPHA_RANGE = config.ALPHA_RANGE
RHO_RANGE = config.RHO_RANGE
FILE_TAG = config.FILE_TAG
GAMMA_CATEGORIES = config.GAMMA_CATEGORIES

ALPHABET_SIZE = 20

MAX_INDEL_LENGTH = 150  # truncation of the indel length distributions (indel-model prior)


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
        (random.choice(genes), random.uniform(*RHO_RANGE), sample_alpha())
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


def generate_and_save(genes: list, n_samples: int, window_size: int, split: str):
    """Simulate one dataset and save it, unless its file already exists."""
    path = Path(config.TRAINING_SIMULATION_DIR) / f"sims_{FILE_TAG}_w{window_size}_n{n_samples}_{split}.npz"
    if path.exists():
        print(f"  {path} already exists, skipping")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(genes, n_samples, window_size)
    meta = {"window_size": window_size, "alpha_range": ALPHA_RANGE, "rho_range": list(RHO_RANGE),
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


def main():
    random.seed(config.SIMULATION_SEED)
    np.random.seed(config.SIMULATION_SEED)
    torch.manual_seed(config.TORCH_SEED)
    print(f"simulation_seed={config.SIMULATION_SEED} split_seed={SPLIT_SEED} torch_seed={config.TORCH_SEED}")
    genes = load_genes(ESTIMATES_CSV, INDEL_CSV)

    # split genes once (same split for every window) so test genes are never seen in training
    train_genes, test_genes = held_out_genes(
        genes, split_seed=SPLIT_SEED, test_gene_fraction=TEST_GENE_FRACTION
    )
    print(f"genes: {len(train_genes)} train, {len(test_genes)} test")

    for window_size in WINDOW_SIZES:
        print(f"\n=== window_size={window_size} ===")
        generate_and_save(train_genes, N_TRAIN, window_size, "train")
        generate_and_save(test_genes, N_TEST, window_size, "test")

    # length-transfer test sets: held-out genes at the other window sizes
    for test_size in TEST_SIZES:
        print(f"\n=== test window_size={test_size} ===")
        generate_and_save(test_genes, N_TEST, test_size, "test")


if __name__ == "__main__":
    main()
