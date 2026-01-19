"""
Validation script for trained models using freshly generated trees.

Generates random trees, simulates MSAs, extracts features, and evaluates
both Random Forest and Neural Network models.
"""

import pathlib
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from ete3 import Tree
from io import StringIO
from Bio import SeqIO

try:
    from msasim import protocol, simulator as sim
except ImportError:
    print("Error: 'msasim' library not found.")
    exit()

import config
from features_calculator import calculate_msa_entropy_stats, read_phylip_sequences


# Validation configuration
VALIDATION_SEED = 42
N_TREES = 50
N_MSAS_PER_TREE = 10
MIN_TAXA = 5
MAX_TAXA = 100
MIN_SEQ_LENGTH = 500
MAX_SEQ_LENGTH = 5000


def generate_random_tree(n_taxa, seed):
    """
    Generate a random tree with n_taxa leaves.
    
    Args:
        n_taxa: Number of taxa (leaves)
        seed: Random seed for reproducibility
        
    Returns:
        ete3.Tree object
    """
    # Set numpy seed for branch length sampling
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


def simulate_msa_for_tree(tree, sim_seed):
    """
    Simulate a single MSA for a given tree.
    
    Args:
        tree: ete3.Tree object
        sim_seed: Seed for this simulation
        
    Returns:
        tuple: (sequences_list, true_alpha, true_rho)
    """
    # Sample random parameters
    random.seed(sim_seed)
    np.random.seed(sim_seed)
    
    true_alpha = round(random.uniform(*config.ALPHA_RANGE), 3)
    true_rho = round(random.uniform(*config.RHO_RANGE), 3)
    seequence_length = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)

    # Get Newick string directly from tree
    newick_string = tree.write(format=1)
    
    # Setup simulation with Newick string
    simulation_protocol = protocol.SimProtocol(newick_string)
    simulation_protocol.set_sequence_size(seequence_length)
    simulation_protocol.set_insertion_rates(0.0)
    simulation_protocol.set_deletion_rates(0.0)
    simulation_protocol.set_seed(sim_seed)
    
    # Create simulator
    simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)
    
    # Configure model
    simulator.set_replacement_model(
        model=sim.MODEL_CODES.WAG,
        gamma_parameters_alpha=true_alpha,
        gamma_parameters_categories=8,
        site_rate_correlation=true_rho
    )
    
    # Run simulation
    msa = simulator()
    msa_str = msa.get_msa()
    
    # Parse sequences from FASTA string
    handle = StringIO(msa_str)
    sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    
    return sequences, true_alpha, true_rho

def main():
    """Main validation workflow."""
    print("=" * 60)
    print("MODEL VALIDATION WITH RANDOM TREES")
    print("=" * 60)
    print(f"Validation seed: {VALIDATION_SEED}")
    print(f"Generating {N_TREES} trees ({MIN_TAXA}-{MAX_TAXA} taxa)")
    print(f"Simulating {N_MSAS_PER_TREE} MSAs per tree")
    print(f"Total MSAs: {N_TREES * N_MSAS_PER_TREE}")
    print()
    
    # Setup directories
    validation_dir = pathlib.Path("results/validation")
    trees_dir = validation_dir / "validation_trees"
    data_dir = validation_dir / "validation_data"
    plots_dir = validation_dir / "plots"
    
    for dir_path in [trees_dir, data_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set master seed
    random.seed(VALIDATION_SEED)
    np.random.seed(VALIDATION_SEED)
    
    # Step 1: Generate trees and simulate MSAs
    print("Step 1: Generating trees and simulating MSAs...")
    print("-" * 60)
    
    all_data = []

    for tree_idx in range(N_TREES):
        # Determine number of taxa for this tree
        n_taxa = random.randint(MIN_TAXA, MAX_TAXA)
        
        # Generate tree with deterministic seed
        tree_seed = VALIDATION_SEED + tree_idx * 1000
        tree = generate_random_tree(n_taxa, tree_seed)
        
        tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"
        
        if (tree_idx + 1) % 10 == 0:
            print(f"  Generated {tree_idx + 1}/{N_TREES} trees")
        
        # Simulate MSAs for this tree (no file writing needed)
        for msa_idx in range(N_MSAS_PER_TREE):
            sim_seed = tree_seed + msa_idx + 1
            sequences, true_alpha, true_rho = simulate_msa_for_tree(tree, sim_seed)
            
            # Extract features
            entropy_stats = calculate_msa_entropy_stats(sequences)
            
            # Store all data
            all_data.append({
                'tree': tree_name,
                'msa_idx': msa_idx,
                'n_taxa': n_taxa,
                'true_alpha': true_alpha,
                'true_rho': true_rho,
                **{col: entropy_stats[col] for col in config.FEATURE_COLUMNS if col in entropy_stats}
            })
    
    print(f"  Completed: {len(all_data)} MSAs generated and features extracted")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save ground truth and features
    features_file = data_dir / "validation_features.csv"
    df.to_csv(features_file, index=False)
    print(f"\n  Features saved to: {features_file}")
    
    # Step 2: Load trained models and predict
    print("\n" + "=" * 60)
    print("Step 2: Loading models and making predictions...")
    print("-" * 60)
    
    models_dir = pathlib.Path("models")
    
    results = {}
    
    for model_type in ['random_forest', 'neural_net']:
        model_file = models_dir / f"{model_type}_model.pkl"
        
        if not model_file.exists():
            print(f"  WARNING: {model_file} not found. Skipping {model_type}.")
            continue
        
        print(f"\n  Loading {model_type} model...")
        model = joblib.load(model_file)
        
        # Prepare features
        X = df[config.FEATURE_COLUMNS].values
        
        # Predict
        predictions = model.predict(X)
        
        # Store predictions
        df[f'pred_alpha_{model_type}'] = predictions[:, 0]
        df[f'pred_rho_{model_type}'] = predictions[:, 1]
        
        # Calculate metrics
        mse_alpha = np.mean((df['true_alpha'] - predictions[:, 0]) ** 2)
        mse_rho = np.mean((df['true_rho'] - predictions[:, 1]) ** 2)
        
        results[model_type] = {
            'mse_alpha': mse_alpha,
            'mse_rho': mse_rho
        }
        
        print(f"    Alpha MSE: {mse_alpha:.6f}")
        print(f"    Rho MSE:   {mse_rho:.6f}")
        
        # Save predictions
        pred_file = validation_dir / f"predictions_{model_type}.csv"
        pred_cols = ['tree', 'msa_idx', 'n_taxa', 'true_alpha', 'true_rho', 
                     f'pred_alpha_{model_type}', f'pred_rho_{model_type}']
        df[pred_cols].to_csv(pred_file, index=False)
        print(f"    Predictions saved to: {pred_file}")
    
    if len(results) == 2:  # Only if both models were loaded
        print("\n" + "-" * 60)
        print("Computing ensemble predictions...")
        
        df['pred_alpha_ensemble'] = (df['pred_alpha_random_forest'] + df['pred_alpha_neural_net']) / 2
        df['pred_rho_ensemble'] = (df['pred_rho_random_forest'] + df['pred_rho_neural_net']) / 2
        
        mse_alpha_ensemble = np.mean((df['true_alpha'] - df['pred_alpha_ensemble']) ** 2)
        mse_rho_ensemble = np.mean((df['true_rho'] - df['pred_rho_ensemble']) ** 2)
        
        results['ensemble'] = {
            'mse_alpha': mse_alpha_ensemble,
            'mse_rho': mse_rho_ensemble
        }
        
        print(f"  Alpha MSE: {mse_alpha_ensemble:.6f}")
        print(f"  Rho MSE:   {mse_rho_ensemble:.6f}")
        
        # Save ensemble predictions
        pred_file = validation_dir / "predictions_ensemble.csv"
        pred_cols = ['tree', 'msa_idx', 'n_taxa', 'true_alpha', 'true_rho', 
                    'pred_alpha_ensemble', 'pred_rho_ensemble']
        df[pred_cols].to_csv(pred_file, index=False)
        print(f"  Predictions saved to: {pred_file}")
    
    # Step 3: Create scatter plots
    print("\n" + "=" * 60)
    print("Step 3: Creating scatter plots...")
    print("-" * 60)
    
    for model_type in results.keys():
        for param in ['alpha', 'rho']:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            true_col = f'true_{param}'
            pred_col = f'pred_{param}_{model_type}'
            
            # Scatter plot
            ax.scatter(df[true_col], df[pred_col], alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(df[true_col].min(), df[pred_col].min())
            max_val = max(df[true_col].max(), df[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Labels and title
            ax.set_xlabel(f'True {param}', fontsize=12)
            ax.set_ylabel(f'Predicted {param}', fontsize=12)
            ax.set_title(f'{model_type.replace("_", " ").title()} - {param.upper()}\n'
                        f'MSE: {results[model_type][f"mse_{param}"]:.6f}', 
                        fontsize=14)
            # R^2 calculation
            correlation_matrix = np.corrcoef(df[true_col], df[pred_col])
            r_squared = correlation_matrix[0, 1] ** 2
            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.4f}', 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = plots_dir / f"{model_type}_{param}_scatter.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            
            print(f"  Saved: {plot_file}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {validation_dir}/")
    print("\nSummary:")
    for model_type, metrics in results.items():
        print(f"\n{model_type.replace('_', ' ').title()}:")
        print(f"  Alpha MSE: {metrics['mse_alpha']:.6f}")
        print(f"  Rho MSE:   {metrics['mse_rho']:.6f}")


if __name__ == "__main__":
    main()