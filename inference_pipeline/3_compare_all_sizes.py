"""
Step 3b: Compare all three ML model sizes with codeml.

Compares ML models trained with different data sizes (50, 100, 200 sims per tree)
against codeml predictions on the EXACT same test set.

Input:
  - results/alpha_model_{50,100,200}.pkl (trained models)
  - results/rho_model_{50,100,200}.pkl (trained models)
  - results/features.csv (features for all samples)
  - ../simulated_data/{tree}/ground_truth.csv (true values)
  - ../results/inferred_parameters.csv (codeml results)

Output:
  - results/method_comparison_all_sizes.csv (metrics for all methods)
  - results/comparison_plot.png (visualization)
  - results/detailed_predictions_all_sizes.csv (per-sample predictions)
"""

import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Import local config
import config as local_config

# Import parent config
sys.path.append('..')
import config as parent_config


def load_ground_truth(simulated_data_dir):
    """
    Load all ground truth files and combine them.
    
    Returns:
        DataFrame with columns: tree, simulation, true_alpha, true_rho
    """
    all_truth = []
    
    simulated_data_path = pathlib.Path(simulated_data_dir)
    
    for tree_dir in sorted(simulated_data_path.iterdir()):
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


def calculate_metrics(true_values, predicted_values, method_name, param_name):
    """Calculate evaluation metrics for a method."""
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    
    return {
        'method': method_name,
        'parameter': param_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': len(true_values)
    }


def get_complete_codeml_trees(codeml_results_file, required_sims=50):
    """
    Identify trees where codeml completed all required simulations.
    
    Returns:
        set: Tree names with complete codeml results
    """
    if not codeml_results_file.exists():
        return None
    
    codeml_df = pd.read_csv(codeml_results_file)
    tree_counts = codeml_df.groupby('tree').size()
    complete_trees = set(tree_counts[tree_counts >= required_sims].index)
    
    return complete_trees


def create_comparison_plots(comparison_df, all_metrics, output_file):
    """Create visualization comparing all methods."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data for plotting
    metrics_df = pd.DataFrame(all_metrics)
    
    # Colors for different methods
    colors = {
        'ML_50': '#1f77b4',
        'ML_100': '#ff7f0e', 
        'ML_200': '#2ca02c',
        'codeml': '#d62728'
    }
    
    # Plot 1: Alpha RMSE comparison
    alpha_metrics = metrics_df[metrics_df['parameter'] == 'alpha']
    ax = axes[0, 0]
    x_pos = np.arange(len(alpha_metrics))
    bars = ax.bar(x_pos, alpha_metrics['rmse'], color=[colors[m] for m in alpha_metrics['method']])
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Alpha: RMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha_metrics['method'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, alpha_metrics['rmse'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Alpha R² comparison
    ax = axes[0, 1]
    bars = ax.bar(x_pos, alpha_metrics['r2'], color=[colors[m] for m in alpha_metrics['method']])
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Alpha: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(alpha_metrics['method'], rotation=45, ha='right')
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, alpha_metrics['r2'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Rho RMSE comparison
    rho_metrics = metrics_df[metrics_df['parameter'] == 'rho']
    ax = axes[1, 0]
    x_pos = np.arange(len(rho_metrics))
    bars = ax.bar(x_pos, rho_metrics['rmse'], color=[colors[m] for m in rho_metrics['method']])
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Rho: RMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rho_metrics['method'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, rho_metrics['rmse'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Rho R² comparison
    ax = axes[1, 1]
    bars = ax.bar(x_pos, rho_metrics['r2'], color=[colors[m] for m in rho_metrics['method']])
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Rho: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rho_metrics['method'], rotation=45, ha='right')
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, rho_metrics['r2'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Comparison plots saved to: {output_file}")
    plt.close()


def main():
    """Main comparison function."""
    
    # Paths
    results_dir = pathlib.Path('results')
    parent_results_dir = pathlib.Path('..') / 'results'
    features_file = results_dir / 'features.csv'
    codeml_file = parent_results_dir / 'inferred_parameters.csv'
    tree_split_file = results_dir / 'tree_split.csv'
    simulated_data_dir = parent_config.SIMULATED_DATA_DIR
    
    # Check required files
    required_files = {
        'Features': features_file,
        'Codeml results': codeml_file,
        'Tree split': tree_split_file,
    }
    
    # Check model files
    model_sizes = [50, 100, 200]
    for size in model_sizes:
        required_files[f'Alpha model {size}'] = results_dir / f'alpha_model_{size}.pkl'
        required_files[f'Rho model {size}'] = results_dir / f'rho_model_{size}.pkl'
    
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        print("\nMake sure you've run:")
        print("  1. Step 1: Extract features")
        print("  2. Step 2b: Train models with varying sizes")
        print("  3. Main pipeline step 2 & 3: Run codeml and extract parameters")
        return
    
    print("=" * 70)
    print("COMPARING ALL ML MODEL SIZES WITH CODEML")
    print("=" * 70)
    print()
    
    print("Loading data...")
    print("-" * 70)
    
    # Load ground truth
    ground_truth = load_ground_truth(simulated_data_dir)
    print(f"Ground truth: {len(ground_truth)} samples")
    
    # Load features
    features_df = pd.read_csv(features_file)
    print(f"Features: {len(features_df)} samples")
    
    # Load codeml predictions
    codeml_df = pd.read_csv(codeml_file)
    print(f"Codeml predictions: {len(codeml_df)} samples")
    
    # Load tree split info
    tree_split = pd.read_csv(tree_split_file)
    train_trees = set(tree_split[tree_split['split'] == 'train']['tree'])
    test_trees = set(tree_split[tree_split['split'] == 'test']['tree'])
    print(f"Tree split: {len(train_trees)} train, {len(test_trees)} test")
    
    # Load all ML models
    ml_models = {}
    for size in model_sizes:
        with open(results_dir / f'alpha_model_{size}.pkl', 'rb') as f:
            ml_models[f'alpha_{size}'] = pickle.load(f)
        with open(results_dir / f'rho_model_{size}.pkl', 'rb') as f:
            ml_models[f'rho_{size}'] = pickle.load(f)
    print(f"Loaded ML models for sizes: {model_sizes}")
    
    # Merge features with ground truth
    merged = pd.merge(
        features_df,
        ground_truth,
        on=['tree', 'simulation'],
        how='inner'
    )
    print(f"\nMerged features + ground truth: {len(merged)} samples")
    
    # Filter to test trees only
    test_data = merged[merged['tree'].isin(test_trees)].copy()
    print(f"Test tree samples: {len(test_data)}")
    
    # Further filter to first 50 simulations per tree (to match codeml)
    def get_first_n_sims(group, n=50):
        return group.sort_values('simulation').head(n)
    
    test_data = test_data.groupby('tree', group_keys=False).apply(
        lambda x: get_first_n_sims(x, 50)
    ).reset_index(drop=True)
    print(f"Test samples (first 50 sims/tree): {len(test_data)}")
    
    # Generate ML predictions for all model sizes
    feature_columns = local_config.FEATURE_COLUMNS
    X_test = test_data[feature_columns].values
    
    for size in model_sizes:
        test_data[f'ml_alpha_{size}'] = ml_models[f'alpha_{size}'].predict(X_test)
        test_data[f'ml_rho_{size}'] = ml_models[f'rho_{size}'].predict(X_test)
    
    print(f"Generated ML predictions for all model sizes")
    
    # Merge with codeml predictions (inner join - only samples where codeml succeeded)
    comparison_df = pd.merge(
        test_data,
        codeml_df,
        on=['tree', 'simulation'],
        how='inner',
        suffixes=('', '_codeml')
    )
    
    print(f"\nSamples with both ML and codeml predictions: {len(comparison_df)}")
    
    if len(comparison_df) == 0:
        print("\nError: No overlapping samples between ML and codeml predictions.")
        return
    
    # Calculate metrics for all methods
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)
    
    all_metrics = []
    
    # ML models at different sizes
    for size in model_sizes:
        # Alpha
        ml_alpha = calculate_metrics(
            comparison_df['true_alpha'],
            comparison_df[f'ml_alpha_{size}'],
            f'ML_{size}',
            'alpha'
        )
        all_metrics.append(ml_alpha)
        
        # Rho
        ml_rho = calculate_metrics(
            comparison_df['true_rho'],
            comparison_df[f'ml_rho_{size}'],
            f'ML_{size}',
            'rho'
        )
        all_metrics.append(ml_rho)
    
    # Codeml
    codeml_alpha = calculate_metrics(
        comparison_df['true_alpha'],
        comparison_df['inferred_alpha'],
        'codeml',
        'alpha'
    )
    all_metrics.append(codeml_alpha)
    
    codeml_rho = calculate_metrics(
        comparison_df['true_rho'],
        comparison_df['inferred_rho'],
        'codeml',
        'rho'
    )
    all_metrics.append(codeml_rho)
    
    # Display results
    print("\n" + "=" * 70)
    print(f"RESULTS (n={len(comparison_df)} test samples)")
    print("=" * 70)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Alpha results
    print("\nALPHA:")
    print("-" * 70)
    alpha_results = metrics_df[metrics_df['parameter'] == 'alpha'][
        ['method', 'rmse', 'mae', 'r2']
    ].sort_values('rmse')
    print(alpha_results.to_string(index=False))
    
    # Calculate improvements
    print("\nPerformance vs codeml:")
    codeml_alpha_rmse = alpha_results[alpha_results['method'] == 'codeml']['rmse'].values[0]
    for size in model_sizes:
        ml_rmse = alpha_results[alpha_results['method'] == f'ML_{size}']['rmse'].values[0]
        ratio = ml_rmse / codeml_alpha_rmse
        print(f"  ML_{size}: {ratio:.2f}× worse than codeml")
    
    # Rho results
    print("\n" + "-" * 70)
    print("RHO:")
    print("-" * 70)
    rho_results = metrics_df[metrics_df['parameter'] == 'rho'][
        ['method', 'rmse', 'mae', 'r2']
    ].sort_values('rmse')
    print(rho_results.to_string(index=False))
    
    # Calculate improvements
    print("\nPerformance vs codeml:")
    codeml_rho_rmse = rho_results[rho_results['method'] == 'codeml']['rmse'].values[0]
    for size in model_sizes:
        ml_rmse = rho_results[rho_results['method'] == f'ML_{size}']['rmse'].values[0]
        ratio = ml_rmse / codeml_rho_rmse
        print(f"  ML_{size}: {ratio:.2f}× worse than codeml")
    
    # Analyze ML improvement across sizes
    print("\n" + "-" * 70)
    print("ML IMPROVEMENT ACROSS TRAINING SIZES:")
    print("-" * 70)
    
    print("\nAlpha:")
    ml_alpha_rmse = [metrics_df[(metrics_df['parameter'] == 'alpha') & 
                                 (metrics_df['method'] == f'ML_{size}')]['rmse'].values[0] 
                     for size in model_sizes]
    for i in range(len(model_sizes) - 1):
        improvement = (ml_alpha_rmse[i] - ml_alpha_rmse[i+1]) / ml_alpha_rmse[i] * 100
        print(f"  {model_sizes[i]}→{model_sizes[i+1]} sims: {improvement:+.2f}% RMSE improvement")
    
    print("\nRho:")
    ml_rho_rmse = [metrics_df[(metrics_df['parameter'] == 'rho') & 
                               (metrics_df['method'] == f'ML_{size}')]['rmse'].values[0] 
                   for size in model_sizes]
    for i in range(len(model_sizes) - 1):
        improvement = (ml_rho_rmse[i] - ml_rho_rmse[i+1]) / ml_rho_rmse[i] * 100
        print(f"  {model_sizes[i]}→{model_sizes[i+1]} sims: {improvement:+.2f}% RMSE improvement")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save metrics
    metrics_file = results_dir / 'method_comparison_all_sizes.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Metrics saved to: {metrics_file}")
    
    # Save detailed predictions
    output_columns = ['tree', 'simulation', 'true_alpha', 'true_rho']
    for size in model_sizes:
        output_columns.extend([f'ml_alpha_{size}', f'ml_rho_{size}'])
    output_columns.extend(['inferred_alpha', 'inferred_rho'])
    
    predictions_file = results_dir / 'detailed_predictions_all_sizes.csv'
    comparison_df[output_columns].to_csv(predictions_file, index=False)
    print(f"  Detailed predictions saved to: {predictions_file}")
    
    # Create plots
    plot_file = results_dir / 'comparison_plot.png'
    create_comparison_plots(comparison_df, all_metrics, plot_file)
    
    # Calculate coverage
    total_test_samples = len(test_data)
    codeml_coverage = len(comparison_df) / total_test_samples * 100
    print(f"\n  Codeml coverage: {len(comparison_df)}/{total_test_samples} samples ({codeml_coverage:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
