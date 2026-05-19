"""
Step 4: Calculate Mean Squared Error (MSE) for alpha and rho.

Compare ground truth values with inferred values and compute:
  - MSE per tree
  - Overall MSE
  - Additional statistics (MAE, correlation, etc.)
"""

import pathlib
import pandas as pd
import numpy as np
import config

def load_ground_truth(simulated_data_dir):
    """
    Load all ground truth files and combine them.
    
    Returns:
        DataFrame with columns: tree, simulation, true_alpha, true_rho
    """
    all_truth = []
    
    for tree_dir in sorted(simulated_data_dir.iterdir()):
        if not tree_dir.is_dir():
            continue
        
        tree_name = tree_dir.name
        truth_file = tree_dir / 'ground_truth.csv'
        
        if not truth_file.exists():
            continue
        
        df = pd.read_csv(truth_file)
        
        # Extract simulation name from filename (remove .phy extension)
        df['simulation'] = df['filename'].str.replace('.phy', '', regex=False)
        df['tree'] = tree_name
        
        # Select relevant columns
        df = df[['tree', 'simulation', 'true_alpha', 'true_rho']]
        
        all_truth.append(df)
    
    return pd.concat(all_truth, ignore_index=True)

def calculate_metrics(true_values, inferred_values):
    """Calculate various error metrics."""
    errors = inferred_values - true_values
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)
    
    return {
        'mse': np.mean(squared_errors),
        'rmse': np.sqrt(np.mean(squared_errors)),
        'mae': np.mean(absolute_errors),
        'correlation': np.corrcoef(true_values, inferred_values)[0, 1]
    }

def main():
    """Main function to calculate MSE."""
    simulated_data_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    results_dir = pathlib.Path(config.RESULTS_DIR)
    inferred_file = results_dir / 'inferred_parameters.csv'
    
    if not inferred_file.exists():
        print(f"Error: {inferred_file} does not exist. Run step 3 first.")
        return
    
    print("Loading data...")
    
    # Load ground truth and inferred parameters
    ground_truth = load_ground_truth(simulated_data_dir)
    inferred = pd.read_csv(inferred_file)
    
    # Merge on tree and simulation
    merged = pd.merge(
        ground_truth,
        inferred,
        on=['tree', 'simulation'],
        how='inner'
    )
    
    print(f"Matched {len(merged)} simulation runs")
    
    if len(merged) == 0:
        print("Error: No matching simulations found.")
        return
    
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    
    # Calculate overall metrics
    alpha_metrics = calculate_metrics(merged['true_alpha'], merged['inferred_alpha'])
    rho_metrics = calculate_metrics(merged['true_rho'], merged['inferred_rho'])
    
    print("\nAlpha:")
    print(f"  MSE:         {alpha_metrics['mse']:.6f}")
    print(f"  RMSE:        {alpha_metrics['rmse']:.6f}")
    print(f"  MAE:         {alpha_metrics['mae']:.6f}")
    print(f"  Correlation: {alpha_metrics['correlation']:.6f}")
    
    print("\nRho:")
    print(f"  MSE:         {rho_metrics['mse']:.6f}")
    print(f"  RMSE:        {rho_metrics['rmse']:.6f}")
    print(f"  MAE:         {rho_metrics['mae']:.6f}")
    print(f"  Correlation: {rho_metrics['correlation']:.6f}")
    
    # Calculate per-tree metrics
    print("\n" + "=" * 50)
    print("PER-TREE RESULTS")
    print("=" * 50)
    
    per_tree_results = []
    
    for tree_name in sorted(merged['tree'].unique()):
        tree_data = merged[merged['tree'] == tree_name]
        
        alpha_tree_metrics = calculate_metrics(
            tree_data['true_alpha'],
            tree_data['inferred_alpha']
        )
        rho_tree_metrics = calculate_metrics(
            tree_data['true_rho'],
            tree_data['inferred_rho']
        )
        
        per_tree_results.append({
            'tree': tree_name,
            'n_simulations': len(tree_data),
            'alpha_mse': alpha_tree_metrics['mse'],
            'alpha_rmse': alpha_tree_metrics['rmse'],
            'alpha_mae': alpha_tree_metrics['mae'],
            'alpha_correlation': alpha_tree_metrics['correlation'],
            'rho_mse': rho_tree_metrics['mse'],
            'rho_rmse': rho_tree_metrics['rmse'],
            'rho_mae': rho_tree_metrics['mae'],
            'rho_correlation': rho_tree_metrics['correlation']
        })
        
        print(f"\n{tree_name} (n={len(tree_data)}):")
        print(f"  Alpha MSE: {alpha_tree_metrics['mse']:.6f}, Rho MSE: {rho_tree_metrics['mse']:.6f}")
    
    # Save results
    output_file = results_dir / 'mse_analysis.csv'
    
    # Add overall results as first row
    overall_row = {
        'tree': 'OVERALL',
        'n_simulations': len(merged),
        'alpha_mse': alpha_metrics['mse'],
        'alpha_rmse': alpha_metrics['rmse'],
        'alpha_mae': alpha_metrics['mae'],
        'alpha_correlation': alpha_metrics['correlation'],
        'rho_mse': rho_metrics['mse'],
        'rho_rmse': rho_metrics['rmse'],
        'rho_mae': rho_metrics['mae'],
        'rho_correlation': rho_metrics['correlation']
    }
    
    results_df = pd.DataFrame([overall_row] + per_tree_results)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 50)
    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    
    # Also save the merged data for further analysis
    merged_output = results_dir / 'merged_data.csv'
    merged.to_csv(merged_output, index=False)
    print(f"Merged data saved to: {merged_output}")

if __name__ == "__main__":
    main()
