"""
Step 1: Extract entropy and parsimony features from simulated MSAs.

For each simulated MSA:
  - Calculate Shannon entropy statistics
  - Calculate parsimony-based statistics
  - Calculate gamma shape features (NEW!)
  - Save features to results/features.csv

These features will be used for training a model to infer alpha and rho.
"""

import pandas as pd
import pathlib
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import config
from features_calculator import (
    calculate_msa_entropy_stats,
    read_fasta_sequences
)


def process_single_simulation(sim_file):
    """
    Worker function to process a single simulation file.
    Must be at module level for multiprocessing pickling.
    
    Args:
        sim_file: Path to .fasta simulation file        
    Returns:
        dict: Feature dictionary or None if error
    """
    sim_name = sim_file.stem
    
    try:
        # Read sequences
        sequences = read_fasta_sequences(sim_file)
        
        # Calculate entropy statistics
        entropy_stats = calculate_msa_entropy_stats(sequences)
        # Combine all features
        return {
            'tree': sim_file.parent.name,
            'simulation': sim_name,
            # Entropy features
            'avg_entropy': entropy_stats['avg_entropy'],
            'entropy_variance': entropy_stats['entropy_variance'],
            'max_entropy': entropy_stats['max_entropy'],
            'lag1_autocorr': entropy_stats['lag1_autocorr'],
            'entropy_skewness': entropy_stats['entropy_skewness'],
            'entropy_kurtosis': entropy_stats['entropy_kurtosis'],
            'bimodality_coefficient': entropy_stats['bimodality_coefficient'],
            'gamma_shape_entropy': entropy_stats['gamma_shape_entropy'],
            # Histogram bins
            **{f'entropy_bin_{i}': entropy_stats[f'entropy_bin_{i}'] for i in range(10)},
            # Multi-lag autocorrelations
            **{f'lag{lag}_autocorr': entropy_stats[f'lag{lag}_autocorr'] for lag in range(2, 11)},
        }
    
    except Exception as e:
        print(f"  ERROR processing {sim_file.name}: {e}")
        return None


def process_simulations_folder(simulations_folder, num_workers=1):
    """
    Process all simulations for a single tree using multiprocessing.
    
    Args:
        tree_dir: Path to tree directory containing .phy files
        trees_dir: Path to directory containing .newick tree files
        num_workers: Number of parallel workers
        
    Returns:
        list: List of dicts with simulations and all features
    """
    results = []
    
    
    
    # Find all simulation files
    sim_files = sorted(simulations_folder.glob("sim_*.fasta"))
    
    if not sim_files:
        print(f"  WARNING: No simulation files found in {simulations_folder}")
        return results
    
    total = len(sim_files)
    print(f"    Processing {total} simulations using {num_workers} cores...")
    
    # Use multiprocessing to process simulations in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_sim = {
            executor.submit(process_single_simulation, sim_file): sim_file
            for sim_file in sim_files
        }
        
        completed = 0
        for future in as_completed(future_to_sim):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"    Completed {completed}/{total} simulations")
                    
            except Exception as e:
                sim_file = future_to_sim[future]
                print(f"    ERROR processing {sim_file.name}: {e}")
    
    return results


def main():
    """Main function to extract all features from all MSAs."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract entropy and parsimony features from MSAs')
    parser.add_argument('--cores', type=int, default=os.cpu_count(),
                        help=f'Number of cores to use (default: {os.cpu_count()})')
    args = parser.parse_args()
    
    simulated_data_dir = config.SIMULATED_DATA_DIR
    results_dir = config.FEATURES_DIR
    
    if not simulated_data_dir.exists():
        print(f"Error: {simulated_data_dir}/ does not exist.")
        print("Run the main pipeline's step 1 first to generate simulations.")
        return
    
    
    print("Extracting features from simulated MSAs...")
    print(f"Feature set: {len(config.FEATURE_COLUMNS)} features")
    print(f"Using {args.cores} CPU cores for parallel processing")
    print("=" * 50)
    
    # Find all tree directories
    simulation_dirs = sorted([d for d in simulated_data_dir.iterdir() if d.is_dir()])
    
    if not simulation_dirs:
        print(f"Error: No simulation directories found in {simulated_data_dir}/")
        return
    
    print(f"Found {len(simulation_dirs)} tree(s)")
    print()
    
    # Process each tree
    all_results = []
    
    for simulation_dir in simulation_dirs:
        dir_name = simulation_dir.name
        print(f"Processing simulations in: {dir_name}")
        
        simulation_results = process_simulations_folder(simulation_dir, num_workers=args.cores)
        all_results.extend(simulation_results)
        
        print(f"  Extracted features from {len(simulation_results)} simulations")
        print()
    
    if not all_results:
        print("Error: No features extracted. Check simulation files.")
        return
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    output_file = results_dir / 'features.csv'
    df.to_csv(output_file, index=False)
    
    print("=" * 50)
    print(f"Extraction complete!")
    print(f"Total features extracted: {len(all_results)}")
    print(f"Output file: {output_file}")
    
    # Display summary statistics
    print("\nFeature Summary:")
    print(df[config.FEATURE_COLUMNS].describe())


if __name__ == "__main__":
    main()