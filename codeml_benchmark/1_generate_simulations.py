"""
Step 1: Generate simulated MSAs, one per simulation.

For each simulation:
  - Generate a random tree with random n_taxa and tree_scale
  - Simulate an MSA with random alpha and rho
  - Save alignment as .phy and tree as .newick in its own directory
  - Log ground truth to simulated_data/ground_truth.csv
"""

import math
import pathlib
import random
import numpy as np
import argparse
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from Bio import SeqIO

from ete3 import Tree
from msasim import Msa, Simulator, SimProtocol
from msasim.substitutions import ReplacementModelSpec, SiteRateModelSpec
from msasim.constants import MODEL_CODES, ALPHABET_CODES, SITE_RATE_MODELS
from msasim.distributions import ZipfDistribution

import config


def generate_random_tree(n_taxa: int, scale: float) -> Tree:
    """Generate a random tree with n_taxa leaves and exponential branch lengths."""
    tree = Tree()
    tree.populate(n_taxa, random_branches=True)
    for node in tree.traverse():
        if node.dist == 0:
            continue
        node.dist = np.random.exponential(scale=scale)
    
    for leaf in tree.get_leaves():
        leaf.name = leaf.name[4:]

    return tree


def simulate_single(sim_index, output_dir, seed):
    """Generate one simulation with its own random tree."""
    random.seed(seed)
    np.random.seed(seed)

    # Sample parameters
    true_alpha = round(random.uniform(*config.ALPHA_RANGE), 3)
    true_rho = round(random.uniform(*config.RHO_RANGE), 3)
    true_tree_scale = 10 ** random.uniform(
        math.log10(config.TREE_SCALE_RANGE[0]),
        math.log10(config.TREE_SCALE_RANGE[1])
    )
    true_tree_scale = round(true_tree_scale, 6)
    n_taxa = random.randint(config.MIN_TAXA, config.MAX_TAXA)

    # Generate tree
    tree = generate_random_tree(n_taxa=n_taxa, scale=true_tree_scale)
    newick_string = tree.write(format=1)

    # Setup simulator
    sim_name = f"sim_{sim_index:04d}_a{true_alpha}_r{true_rho}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    simulation_protocol = SimProtocol(newick_string)
    simulation_protocol.set_sequence_size(config.SEQ_LENGTH)
    simulation_protocol.set_insertion_rates(0.007)
    simulation_protocol.set_deletion_rates(0.035)
    simulation_protocol.set_insertion_length_distributions(ZipfDistribution(p=1.53, truncation=50))
    simulation_protocol.set_deletion_length_distributions(ZipfDistribution(p=1.11, truncation=50))
    simulation_protocol.set_seed(seed)

    site_rate_model_spec = SiteRateModelSpec(
        gamma_alpha=true_alpha,
        gamma_categories=4,
        site_rate_correlation=true_rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE
    )
    replacement_model_spec = ReplacementModelSpec(
        model=MODEL_CODES.WAG,
        alphabet=ALPHABET_CODES.PROTEIN,
        site_rate_model=site_rate_model_spec
    )

    simulator = Simulator(simulation_protocol, replacement_model=replacement_model_spec)
    msa: Msa = simulator()

    # Save alignment
    msa_str = "\n".join(msa.get_msa_row(i) for i in range(msa.get_num_sequences()))
    phy_path = sim_dir / "alignment.phy"
    SeqIO.convert(StringIO(msa_str), "fasta", phy_path, "phylip-sequential")

    # Save tree
    tree_path = sim_dir / "tree.newick"
    tree_path.write_text(newick_string)

    return sim_name, true_alpha, true_rho, true_tree_scale, n_taxa


def main():
    parser = argparse.ArgumentParser(description='Generate simulated MSAs')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()

    output_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sims = config.NUM_SIMULATIONS
    print(f"Generating {n_sims} simulations")
    print(f"Using {args.cores} CPU cores")
    print(f"Master seed: {config.MASTER_SEED}")
    print("=" * 50)

    # Deterministic per-sim seeds
    rng = random.Random(config.MASTER_SEED)
    seeds = [rng.randint(0, 10**7) for _ in range(n_sims)]

    log_path = output_dir / "ground_truth.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_name", "true_alpha", "true_rho", "true_tree_scale", "n_taxa"])

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        future_to_idx = {
            executor.submit(simulate_single, i + 1, output_dir, seeds[i]): i
            for i in range(n_sims)
        }

        completed = 0
        rows = []
        for future in as_completed(future_to_idx):
            try:
                row = future.result()
                rows.append(row)
                completed += 1
                if completed % 10 == 0 or completed == n_sims:
                    print(f"  Completed {completed}/{n_sims}")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Write all rows sorted by sim_name for reproducibility
    rows.sort(key=lambda r: r[0])
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\n" + "=" * 50)
    print(f"Done. Ground truth: {log_path}")
    print(f"Output directory: {output_dir}/")


if __name__ == "__main__":
    main()