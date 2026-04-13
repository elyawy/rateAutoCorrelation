"""
Step 0: Generate training data with random trees and simulated MSAs (PARALLELIZED).

Features are extracted in-memory and written directly to features/features.csv.
No intermediate FASTA files or ground_truth.csv are saved.
"""

import pathlib
import random
import math
import numpy as np
import argparse
import os
import time 
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from utils.simulation import SimulationParams, generate_random_tree, setup_sim, simulate_msa
from features_calculator import calculate_msa_entropy_stats
from features_calculator import calculate_indel_features


# ==========================================
# TRAINING DATA CONFIGURATION
# ==========================================
TRAINING_SEED = 1337
N_TRAINING_TREES = 1000
N_TRAINING_SIMS_PER_TREE = 100

SIM_PARAMS = SimulationParams(
    min_taxa=20,
    max_taxa=200,
    min_seq_length=50,
    max_seq_length=500,
)

SCALE_MIN = 0.01
SCALE_MAX = 0.5


def process_single_tree(tree_idx):
    """
    Worker function: generate a tree, simulate MSAs, extract features in memory.

    Returns:
        list of dicts, one per MSA, containing features + true_alpha + true_rho
    """
    random.seed(TRAINING_SEED + tree_idx)
    n_taxa = random.randint(SIM_PARAMS.min_taxa, SIM_PARAMS.max_taxa)

    # Sample branch length scale log-uniformly
    random_scale = 10 ** random.uniform(math.log10(SCALE_MIN), math.log10(SCALE_MAX))

    tree_seed = TRAINING_SEED + tree_idx * 10000
    tree = generate_random_tree(n_taxa, scale=random_scale, seed=tree_seed)

    tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"

    random.seed(tree_seed)
    np.random.seed(tree_seed)
    simulator = setup_sim(tree, sim_seed=tree_seed)

    rows = []
    for msa_idx in range(N_TRAINING_SIMS_PER_TREE):
        sequences, true_alpha, true_rho = simulate_msa(simulator, SIM_PARAMS)
        features = calculate_msa_entropy_stats(sequences)
        indel_features = calculate_indel_features(sequences)
        features.update(indel_features)
        
        row = {
            'tree': tree_name,
            'simulation': f"sim_{msa_idx + 1:03d}_a{true_alpha}_r{true_rho}",
            'true_alpha': true_alpha,
            'true_rho': true_rho,
            **{k: features[k] for k in config.FEATURE_COLUMNS if k in features}
        }
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Generate training data and extract features in one step')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING TRAINING DATA + EXTRACTING FEATURES")
    print("=" * 60)
    print(f"Training seed:    {TRAINING_SEED}")
    print(f"Trees:            {N_TRAINING_TREES} ({SIM_PARAMS.min_taxa}-{SIM_PARAMS.max_taxa} taxa)")
    print(f"MSAs per tree:    {N_TRAINING_SIMS_PER_TREE}")
    print(f"Sequence length:  {SIM_PARAMS.min_seq_length}-{SIM_PARAMS.max_seq_length} aa")
    print(f"Branch scale:     log-uniform [{SCALE_MIN}, {SCALE_MAX}]")
    print(f"Total MSAs:       {N_TRAINING_TREES * N_TRAINING_SIMS_PER_TREE}")
    print(f"Cores:            {args.cores}")
    print()

    features_dir = pathlib.Path("features")
    features_dir.mkdir(exist_ok=True)
    output_file = features_dir / "features.csv"

    all_rows = []

    print("Processing trees in parallel...")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        future_to_idx = {
            executor.submit(process_single_tree, tree_idx): tree_idx
            for tree_idx in range(N_TRAINING_TREES)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            tree_idx = future_to_idx[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                completed += 1
                print(f"Completed {completed}/{N_TRAINING_TREES}: tree_{tree_idx:03d} ({len(rows)} MSAs)")
            except Exception as e:
                print(f"Error processing tree {tree_idx}: {e}")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Total MSAs:  {len(df)}")
    print(f"Output:      {output_file}")
    print("\nNext step: python 2_train_models.py")


if __name__ == "__main__":
    main()
