"""
Simulate the training / test data for the rho regressor (no training here, see
train_rho_regressor_amino_indel_v2.py).

Simulations use LG on real OrthoMaM topologies with protein branch lengths per gene
(protein_estimates.csv) and each gene's own inferred indel model (orthomam_indel_params.csv).
Rho and gamma alpha are both sampled per simulation: rho from RHO_RANGE, alpha log-uniform from ALPHA_RANGE.
Each dataset is saved to sims_{FILE_TAG}_w{window}_n{n}_{split}.npz; files that already exist are skipped.
Copy those files to the machine that trains.
"""

import csv
import json
import math
import random
import re

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


from common.features_calculator import calculate_cnn_input, calculate_scalar_features

ESTIMATES_CSV = Path("protein_estimates_train500.csv")  # gene, newick (from build_tree_splits.py)
INDEL_CSV = Path("data/no_sync/indel_estimates_2.csv")  # per-gene indel model
WINDOW_SIZES = [1000]
TEST_SIZES = [100, 250, 500, 1000]  # lengths to evaluate the trained model on
N_TRAIN = 20000
N_TEST = 1000
TEST_GENE_FRACTION = 0.1  # genes held out for the test set only
SPLIT_SEED = 0
N_WORKERS = 8
ALPHA_RANGE = (0.1, 3.0)  # log-uniform
RHO_RANGE = (0.01, 0.99)       # Min and Max Rho
FILE_TAG = "amino_indel_trees500_gap"  # channels: entropy + gap fraction; must match the training script
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
    path = Path(f"sims_{FILE_TAG}_w{window_size}_n{n_samples}_{split}.npz")
    if path.exists():
        print(f"  {path} already exists, skipping")
        return

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
    genes = load_genes()

    # split genes once (same split for every window) so test genes are never seen in training
    random.Random(SPLIT_SEED).shuffle(genes)
    n_test_genes = int(len(genes) * TEST_GENE_FRACTION)
    test_genes = genes[:n_test_genes]
    train_genes = genes[n_test_genes:]
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
