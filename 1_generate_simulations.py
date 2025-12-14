"""
Step 1: Generate simulated MSAs for each tree.

For each tree in trees/:
  - Create a subdirectory in simulated_data/
  - Generate NUM_SIMULATIONS_PER_TREE MSAs with random alpha/rho
  - Save ground truth to CSV
"""

import pathlib
import random
import numpy as np
import hashlib
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from Bio import SeqIO

try:
    from msasim import protocol 
    from msasim import simulator as sim
except ImportError:
    print("Error: 'msasim' library not found. Please install it to run this script.")
    exit()

import config

def get_tree_seed(tree_filename, master_seed):
    """Generate deterministic seed for a tree based on its filename."""
    hash_obj = hashlib.md5(tree_filename.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return master_seed + (hash_int % 1000000)

def simulate_for_tree(tree_path, output_dir, tree_seed):
    """Generate all simulations for a single tree."""
    tree_name = tree_path.stem
    print(f"\nProcessing tree: {tree_name}")
    
    # Create output directory
    tree_output_dir = output_dir / tree_name
    tree_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize log file
    log_path = tree_output_dir / "ground_truth.csv"
    with open(log_path, "w") as log:
        log.write("filename,true_alpha,true_rho\n")
    
    # Setup simulation protocol
    simulation_protocol = protocol.SimProtocol(str(tree_path))
    simulation_protocol.set_sequence_size(config.SEQ_LENGTH)
    simulation_protocol.set_insertion_rates(0.0)
    simulation_protocol.set_deletion_rates(0.0)
    
    # Create simulator
    simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)
    sim_seed = tree_seed
    random.seed(sim_seed)
    np.random.seed(sim_seed)
    simulation_protocol.set_seed(sim_seed)

    # Generate simulations
    for i in range(1, config.NUM_SIMULATIONS_PER_TREE + 1):
        # Set seed for this specific simulation
        
        # Sample random parameters
        true_rho = round(random.uniform(*config.RHO_RANGE), 3)
        true_alpha = round(random.uniform(*config.ALPHA_RANGE), 3)
        
        # Configure model
        simulator.set_replacement_model(
            model=sim.MODEL_CODES.WAG,
            gamma_parameters_alpha=true_alpha,
            gamma_parameters_categories=8,
            site_rate_correlation=true_rho
        )
        
        # simulator.save_root_sequence()
        
        # Run simulation
        msa = simulator()
        msa_str = msa.get_msa()
        
        # Save as PHYLIP
        filename = f"sim_{i:03d}_a{true_alpha}_r{true_rho}.phy"
        filepath = tree_output_dir / filename
        
        handle = StringIO(msa_str)
        SeqIO.convert(handle, "fasta", filepath, "phylip-sequential")
        
        # Log ground truth
        with open(log_path, "a") as log:
            log.write(f"{filename},{true_alpha},{true_rho}\n")
        
        if i % 10 == 0:
            print(f"  Generated {i}/{config.NUM_SIMULATIONS_PER_TREE} simulations")
    
    print(f"  Completed {tree_name}. Ground truth: {log_path}")

def main():
    """Main function to process all trees."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate simulated MSAs')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()
    
    trees_dir = pathlib.Path(config.TREES_DIR)
    output_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    
    # Find all tree files
    tree_files = sorted(trees_dir.glob("*.newick"))
    
    if not tree_files:
        print(f"Error: No .newick files found in {trees_dir}/")
        return
    
    print(f"Found {len(tree_files)} tree(s)")
    print(f"Generating {config.NUM_SIMULATIONS_PER_TREE} simulations per tree")
    print(f"Using {args.cores} CPU cores for parallel processing")
    print(f"Master seed: {config.MASTER_SEED}")
    print("=" * 50)
    
    # Process trees in parallel
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        # Submit all tree processing tasks
        future_to_tree = {
            executor.submit(
                simulate_for_tree,
                tree_path,
                output_dir,
                get_tree_seed(tree_path.name, config.MASTER_SEED)
            ): tree_path for tree_path in tree_files
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_tree):
            tree_path = future_to_tree[future]
            try:
                future.result()  # This will raise any exceptions that occurred
                completed += 1
                print(f"\nCompleted {completed}/{len(tree_files)} trees")
            except Exception as e:
                print(f"\nError processing {tree_path.name}: {e}")
    
    print("\n" + "=" * 50)
    print("All simulations complete!")
    print(f"Output directory: {output_dir}/")
    
if __name__ == "__main__":
    main()
