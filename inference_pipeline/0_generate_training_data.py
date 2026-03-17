"""
Step 0: Generate training data with random trees and simulated MSAs (PARALLELIZED).

Generates random phylogenetic trees and simulates MSAs for training the inference models.
This makes the inference pipeline independent from the main simulation pipeline.
"""

import pathlib
import random
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from utils.simulation import SimulationParams, generate_random_tree, setup_sim, simulate_msa, sequences_to_fasta


# ==========================================
# TRAINING DATA CONFIGURATION
# ==========================================
TRAINING_SEED = 1337
N_TRAINING_TREES = 200
N_TRAINING_SIMS_PER_TREE = 250

SIM_PARAMS = SimulationParams(
    min_taxa=5,
    max_taxa=200,
    min_seq_length=50,
    max_seq_length=5000,
)


def process_single_tree(tree_idx, training_data_dir):
    """
    Worker function to process a single tree.
    Must be at module level for multiprocessing pickling.

    Args:
        tree_idx: Index of the tree (0 to N_TRAINING_TREES-1)
        training_data_dir: Path to training data directory

    Returns:
        str: Name of the processed tree
    """
    random.seed(TRAINING_SEED + tree_idx)
    n_taxa = random.randint(SIM_PARAMS.min_taxa, SIM_PARAMS.max_taxa)

    tree_seed = TRAINING_SEED + tree_idx * 10000
    tree = generate_random_tree(n_taxa, tree_seed)

    tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"
    tree_dir = training_data_dir / tree_name
    tree_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_file = tree_dir / "ground_truth.csv"
    with open(ground_truth_file, "w") as log:
        log.write("filename,true_alpha,true_rho\n")

    random.seed(tree_seed)
    np.random.seed(tree_seed)

    simulator = setup_sim(tree, sim_seed=tree_seed)

    for msa_idx in range(N_TRAINING_SIMS_PER_TREE):
        sequences, true_alpha, true_rho = simulate_msa(simulator, SIM_PARAMS)

        filename = f"sim_{msa_idx + 1:03d}_a{true_alpha}_r{true_rho}.fasta"
        filepath = tree_dir / filename
        filepath.write_text(sequences_to_fasta(sequences))

        with open(ground_truth_file, "a") as log:
            log.write(f"{filename},{true_alpha},{true_rho}\n")

    return tree_name


def main():
    parser = argparse.ArgumentParser(description='Generate training data with random trees')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING TRAINING DATA (PARALLELIZED)")
    print("=" * 60)
    print(f"Training seed: {TRAINING_SEED}")
    print(f"Generating {N_TRAINING_TREES} random trees ({SIM_PARAMS.min_taxa}-{SIM_PARAMS.max_taxa} taxa)")
    print(f"Simulating {N_TRAINING_SIMS_PER_TREE} MSAs per tree")
    print(f"Sequence length: {SIM_PARAMS.min_seq_length}-{SIM_PARAMS.max_seq_length} aa")
    print(f"Total MSAs: {N_TRAINING_TREES * N_TRAINING_SIMS_PER_TREE}")
    print(f"Using {args.cores} CPU cores for parallel processing")
    print()

    training_data_dir = pathlib.Path("training_data")
    training_data_dir.mkdir(exist_ok=True)

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)

    print("Processing trees in parallel...")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        future_to_tree = {
            executor.submit(process_single_tree, tree_idx, training_data_dir): tree_idx
            for tree_idx in range(N_TRAINING_TREES)
        }

        completed = 0
        for future in as_completed(future_to_tree):
            tree_idx = future_to_tree[future]
            try:
                tree_name = future.result()
                completed += 1
                print(f"Completed {completed}/{N_TRAINING_TREES}: {tree_name}")
            except Exception as e:
                print(f"\nError processing tree {tree_idx}: {e}")

    print("\n" + "=" * 60)
    print("TRAINING DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal trees generated: {N_TRAINING_TREES}")
    print(f"Total MSAs generated: {N_TRAINING_TREES * N_TRAINING_SIMS_PER_TREE}")
    print(f"\nData saved to: {training_data_dir.resolve()}/")
    print("\nNext steps:")
    print("  1. Run: python 1_extract_features.py")
    print("  2. Run: python 2_train_models.py")
    print("  3. Run: python 3_evaluate.py")


if __name__ == "__main__":
    main()