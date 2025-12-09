"""
Step 1: Extract entropy features from simulated MSAs.

For each simulated MSA:
  - Calculate Shannon entropy statistics
  - Save features to results/entropy_features.csv

These features will be used for training a model to infer alpha and rho.
"""

import pandas as pd
import config
from entropy_calculator import calculate_msa_entropy_stats, read_phylip_sequences


def process_tree(tree_dir):
    """
    Process all simulations for a single tree.
    
    Args:
        tree_dir: Path to tree directory containing .phy files
        
    Returns:
        list: List of dicts with tree, simulation, and entropy features
    """
    tree_name = tree_dir.name
    results = []
    
    # Find all simulation files
    sim_files = sorted(tree_dir.glob("sim_*.phy"))
    
    if not sim_files:
        print(f"  WARNING: No simulation files found in {tree_dir}")
        return results
    
    for i, sim_file in enumerate(sim_files, 1):
        sim_name = sim_file.stem  # e.g., sim_001_a0.5_r0.3
        
        try:
            # Read sequences
            sequences = read_phylip_sequences(sim_file)
            
            # Calculate entropy statistics
            stats = calculate_msa_entropy_stats(sequences)
            
            # Store results
            results.append({
                'tree': tree_name,
                'simulation': sim_name,
                'avg_entropy': stats['avg_entropy'],
                'entropy_variance': stats['entropy_variance'],
                'min_entropy': stats['min_entropy'],
                'max_entropy': stats['max_entropy']
            })
            
            # Progress update
            if i % 10 == 0 or i == len(sim_files):
                print(f"    Processed {i}/{len(sim_files)} simulations")
        
        except Exception as e:
            print(f"  ERROR processing {sim_file.name}: {e}")
            continue
    
    return results


def main():
    """Main function to extract entropy features from all MSAs."""
    simulated_data_dir = config.SIMULATED_DATA_DIR
    results_dir = config.RESULTS_DIR
    
    if not simulated_data_dir.exists():
        print(f"Error: {simulated_data_dir}/ does not exist.")
        print("Run the main pipeline's step 1 first to generate simulations.")
        return
    
    print("Extracting entropy features from simulated MSAs...")
    print("=" * 50)
    
    # Find all tree directories
    tree_dirs = sorted([d for d in simulated_data_dir.iterdir() if d.is_dir()])
    
    if not tree_dirs:
        print(f"Error: No tree directories found in {simulated_data_dir}/")
        return
    
    print(f"Found {len(tree_dirs)} tree(s)")
    print()
    
    # Process each tree
    all_results = []
    
    for tree_dir in tree_dirs:
        tree_name = tree_dir.name
        print(f"Processing tree: {tree_name}")
        
        tree_results = process_tree(tree_dir)
        all_results.extend(tree_results)
        
        print(f"  Extracted features from {len(tree_results)} simulations")
        print()
    
    if not all_results:
        print("Error: No features extracted. Check simulation files.")
        return
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    output_file = results_dir / 'entropy_features.csv'
    df.to_csv(output_file, index=False)
    
    print("=" * 50)
    print(f"Extraction complete!")
    print(f"Total features extracted: {len(all_results)}")
    print(f"Output file: {output_file}")
    
    # Display summary statistics
    print("\nFeature Summary:")
    print(df[['avg_entropy', 'entropy_variance', 'min_entropy', 'max_entropy']].describe())


if __name__ == "__main__":
    main()
