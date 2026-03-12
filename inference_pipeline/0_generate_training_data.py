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
from ete3 import Tree
from io import StringIO
from Bio import SeqIO

try:
    from msasim import protocol, simulator as sim
    from msasim import Msa
    from msasim.constants import MODEL_CODES, SITE_RATE_MODELS
    from msasim.distributions import ZipfDistribution
except ImportError:
    print("Error: 'msasim' library not found.")
    exit()

import config


# Training data generation configuration
TRAINING_SEED = 1337
N_TRAINING_TREES = 200
N_TRAINING_SIMS_PER_TREE = 250
MIN_TAXA = 5
MAX_TAXA = 200
MIN_SEQ_LENGTH = 50
MAX_SEQ_LENGTH = 2000


def generate_random_tree(n_taxa, seed):
    """
    Generate a random tree with n_taxa leaves.
    
    Args:
        n_taxa: Number of taxa (leaves)
        seed: Random seed for reproducibility
        
    Returns:
        ete3.Tree object
    """
    np.random.seed(seed)
    
    # Create a random tree topology
    tree = Tree()
    tree.populate(n_taxa, random_branches=True)
    
    # Assign branch lengths from exponential distribution (mean=0.1)
    for node in tree.traverse():
        if node.dist == 0:  # Root has dist=0 by default
            continue
        node.dist = np.random.exponential(scale=0.1)
    
    return tree

def setup_sim(tree, sim_seed):
    """
    Setup the simulator for a given tree and seed.
    Args:
        tree: ete3.Tree object
        sim_seed: Seed for this simulation

    Returns:
        msasim.Simulator object
    """
    # Get Newick string directly from tree
    newick_string = tree.write(format=1)

    # Setup simulation with Newick string
    simulation_protocol = protocol.SimProtocol(newick_string)
    simulation_protocol.set_insertion_rates(0.03)
    simulation_protocol.set_deletion_rates(0.09)
    simulation_protocol.set_insertion_length_distributions(ZipfDistribution(1.7, 50))
    simulation_protocol.set_deletion_length_distributions(ZipfDistribution(1.7, 50))
    simulation_protocol.set_site_rate_model(SITE_RATE_MODELS.INDEL_AWARE)
    simulation_protocol.set_seed(sim_seed)

    # Create simulator
    simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)

    return simulator

def simulate_msa_for_tree(simulator: sim.Simulator):
    """
    Simulate a single MSA from the given simulator.
    
    Args:
        simulator: msasim.Simulator object
        
    Returns:
        tuple: (sequences_list, true_alpha, true_rho, msa_string)
    """
    # Sample random parameters
    
    true_alpha = round(random.uniform(*config.ALPHA_RANGE), 3)
    true_rho = round(random.uniform(*config.RHO_RANGE), 3)
    seequence_length = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    print(f"Simulating MSA with alpha={true_alpha}, rho={true_rho}, seq_length={seequence_length}")
    simulator.protocol.set_sequence_size(seequence_length)
    # Configure substitution model
    simulator.set_replacement_model(
        model=MODEL_CODES.WAG,
        gamma_parameters_alpha=true_alpha,
        gamma_parameters_categories=8,
        site_rate_correlation=true_rho
    )

    
    # Run simulation
    msa: Msa = simulator()
    msa_str = "\n".join(msa.get_msa_row(i) for i in range(msa.get_num_sequences()))
    
    return msa_str, true_alpha, true_rho


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
    # Determine number of taxa for this tree
    random.seed(TRAINING_SEED + tree_idx)  # Set seed for this tree's n_taxa
    n_taxa = random.randint(MIN_TAXA, MAX_TAXA)
    
    # Generate tree with deterministic seed
    tree_seed = TRAINING_SEED + tree_idx * 10000
    tree = generate_random_tree(n_taxa, tree_seed)
    
    tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"
    tree_dir = training_data_dir / tree_name
    tree_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ground truth log
    ground_truth_file = tree_dir / "ground_truth.csv"
    with open(ground_truth_file, "w") as log:
        log.write("filename,true_alpha,true_rho\n")
    
    random.seed(tree_seed)
    np.random.seed(tree_seed)

    simulator = setup_sim(tree, sim_seed=tree_seed)
    
    # Simulate MSAs for this tree
    for msa_idx in range(N_TRAINING_SIMS_PER_TREE):
        
        # Generate simulation
        msa_str, true_alpha, true_rho = simulate_msa_for_tree(simulator)
        
        # Save as FASTA
        filename = f"sim_{msa_idx + 1:03d}_a{true_alpha}_r{true_rho}.fasta"
        filepath = tree_dir / filename
        
        filepath.write_text(msa_str)
        
        # Log ground truth
        with open(ground_truth_file, "a") as log:
            log.write(f"{filename},{true_alpha},{true_rho}\n")
    
    return tree_name


def main():
    """Main function to generate all training data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate training data with random trees')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING TRAINING DATA (PARALLELIZED)")
    print("=" * 60)
    print(f"Training seed: {TRAINING_SEED}")
    print(f"Generating {N_TRAINING_TREES} random trees ({MIN_TAXA}-{MAX_TAXA} taxa)")
    print(f"Simulating {N_TRAINING_SIMS_PER_TREE} MSAs per tree")
    print(f"Total MSAs: {N_TRAINING_TREES * N_TRAINING_SIMS_PER_TREE}")
    print(f"Using {args.cores} CPU cores for parallel processing")
    print()
    
    # Setup directory
    training_data_dir = pathlib.Path("training_data")
    training_data_dir.mkdir(exist_ok=True)
    
    # Set master seed for reproducible tree parameters
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    
    # Process trees in parallel
    print("Processing trees in parallel...")
    print("-" * 60)
    process_single_tree(1, training_data_dir)  # Run one tree sequentially for debugging
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        # Submit all tree processing tasks
        future_to_tree = {
            executor.submit(
                process_single_tree,
                tree_idx,
                training_data_dir
            ): tree_idx for tree_idx in range(N_TRAINING_TREES)
        }
        
        # Collect results as they complete
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