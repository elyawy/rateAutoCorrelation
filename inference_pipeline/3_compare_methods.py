"""
Step 3: Compare ML predictions with codeml inferred parameters.

Compares:
  - ML model predictions (from trained RandomForest models)
  - Codeml inferred parameters (from results/inferred_parameters.csv)

Against ground truth values, computing metrics for both methods.
Handles partial codeml results (only compares samples where both methods have predictions).
"""

import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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


def main():
    """Main comparison function."""
    
    # Paths
    results_dir = pathlib.Path('results')
    parent_results_dir = pathlib.Path('..') / 'results'
    entropy_file = results_dir / 'entropy_features.csv'
    codeml_file = parent_results_dir / 'inferred_parameters.csv'
    tree_split_file = results_dir / 'tree_split.csv'
    alpha_model_file = results_dir / 'alpha_model.pkl'
    rho_model_file = results_dir / 'rho_model.pkl'
    simulated_data_dir = parent_config.SIMULATED_DATA_DIR
    
    # Check required files
    required_files = {
        'Entropy features': entropy_file,
        'Codeml results': codeml_file,
        'Tree split': tree_split_file,
        'Alpha model': alpha_model_file,
        'Rho model': rho_model_file
    }
    
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        print("\nMake sure you've run:")
        print("  1. Step 1: Extract entropy features")
        print("  2. Step 2: Train ML models")
        print("  3. Main pipeline step 2 & 3: Run codeml and extract parameters")
        return
    
    print("Loading data...")
    print("=" * 70)
    
    # Load ground truth
    ground_truth = load_ground_truth(simulated_data_dir)
    print(f"Ground truth: {len(ground_truth)} samples")
    
    # Load entropy features
    entropy_df = pd.read_csv(entropy_file)
    print(f"Entropy features: {len(entropy_df)} samples")
    
    # Load codeml predictions
    codeml_df = pd.read_csv(codeml_file)
    print(f"Codeml predictions: {len(codeml_df)} samples")
    
    # Load tree split info
    tree_split = pd.read_csv(tree_split_file)
    train_trees = set(tree_split[tree_split['split'] == 'train']['tree'])
    test_trees = set(tree_split[tree_split['split'] == 'test']['tree'])
    print(f"Tree split: {len(train_trees)} train, {len(test_trees)} test")
    
    # Load ML models
    with open(alpha_model_file, 'rb') as f:
        alpha_model = pickle.load(f)
    with open(rho_model_file, 'rb') as f:
        rho_model = pickle.load(f)
    print("ML models loaded")
    
    # Merge entropy features with ground truth
    merged = pd.merge(
        entropy_df,
        ground_truth,
        on=['tree', 'simulation'],
        how='inner'
    )
    print(f"\nMerged entropy + ground truth: {len(merged)} samples")
    
    # Generate ML predictions for all samples
    feature_columns = ['avg_entropy', 'entropy_variance', 'max_entropy', 'lag1_autocorr']
    X = merged[feature_columns].values
    
    merged['ml_alpha'] = alpha_model.predict(X)
    merged['ml_rho'] = rho_model.predict(X)
    
    # Merge with codeml predictions (inner join - only keep samples where both methods have predictions)
    comparison_df = pd.merge(
        merged,
        codeml_df,
        on=['tree', 'simulation'],
        how='inner',
        suffixes=('', '_codeml')
    )
    
    print(f"Samples with BOTH ML and codeml predictions: {len(comparison_df)}")
    
    if len(comparison_df) == 0:
        print("\nError: No overlapping samples between ML and codeml predictions.")
        return
    
    # Add train/test split indicator
    comparison_df['split'] = comparison_df['tree'].apply(
        lambda x: 'train' if x in train_trees else 'test'
    )
    
    # Split data by train/test
    train_df = comparison_df[comparison_df['split'] == 'train']
    test_df = comparison_df[comparison_df['split'] == 'test']
    
    all_metrics = []
    
    # ========================================================================
    # PRIMARY COMPARISON: TEST TREES (HELD-OUT) - Fair comparison for both methods
    # ========================================================================
    if len(test_df) > 0:
        print("\n" + "=" * 70)
        print(f"PRIMARY COMPARISON: TEST TREES (held-out, n={len(test_df)} samples)")
        print("=" * 70)
        print("Both methods evaluated on trees never seen during ML training")
        
        test_metrics = []
        
        # Alpha
        ml_test_alpha = calculate_metrics(test_df['true_alpha'], test_df['ml_alpha'], 'ML', 'alpha')
        codeml_test_alpha = calculate_metrics(test_df['true_alpha'], test_df['inferred_alpha'], 'codeml', 'alpha')
        test_metrics.extend([ml_test_alpha, codeml_test_alpha])
        all_metrics.extend([ml_test_alpha, codeml_test_alpha])
        
        print(f"\nAlpha:")
        print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print(f"  {'ML':<10} {ml_test_alpha['rmse']:<10.4f} {ml_test_alpha['mae']:<10.4f} {ml_test_alpha['r2']:<10.4f}")
        print(f"  {'codeml':<10} {codeml_test_alpha['rmse']:<10.4f} {codeml_test_alpha['mae']:<10.4f} {codeml_test_alpha['r2']:<10.4f}")
        
        alpha_test_winner = 'ML' if ml_test_alpha['rmse'] < codeml_test_alpha['rmse'] else 'codeml'
        alpha_improvement = abs(ml_test_alpha['rmse'] - codeml_test_alpha['rmse']) / max(ml_test_alpha['rmse'], codeml_test_alpha['rmse']) * 100
        print(f"  → {alpha_test_winner} wins by {alpha_improvement:.1f}% (RMSE)")
        
        # Rho
        ml_test_rho = calculate_metrics(test_df['true_rho'], test_df['ml_rho'], 'ML', 'rho')
        codeml_test_rho = calculate_metrics(test_df['true_rho'], test_df['inferred_rho'], 'codeml', 'rho')
        test_metrics.extend([ml_test_rho, codeml_test_rho])
        all_metrics.extend([ml_test_rho, codeml_test_rho])
        
        print(f"\nRho:")
        print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print(f"  {'ML':<10} {ml_test_rho['rmse']:<10.4f} {ml_test_rho['mae']:<10.4f} {ml_test_rho['r2']:<10.4f}")
        print(f"  {'codeml':<10} {codeml_test_rho['rmse']:<10.4f} {codeml_test_rho['mae']:<10.4f} {codeml_test_rho['r2']:<10.4f}")
        
        rho_test_winner = 'ML' if ml_test_rho['rmse'] < codeml_test_rho['rmse'] else 'codeml'
        rho_improvement = abs(ml_test_rho['rmse'] - codeml_test_rho['rmse']) / max(ml_test_rho['rmse'], codeml_test_rho['rmse']) * 100
        print(f"  → {rho_test_winner} wins by {rho_improvement:.1f}% (RMSE)")
    else:
        print("\n" + "=" * 70)
        print("WARNING: No test tree samples with both predictions")
        print("=" * 70)
    
    # ========================================================================
    # SECONDARY: OVERALL (includes training trees where ML has advantage)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"OVERALL COMPARISON (All samples, n={len(comparison_df)})")
    print("=" * 70)
    print("Note: ML has advantage on training trees")
    
    # ML Alpha
    ml_alpha_metrics = calculate_metrics(
        comparison_df['true_alpha'],
        comparison_df['ml_alpha'],
        'ML',
        'alpha'
    )
    
    # Codeml Alpha
    codeml_alpha_metrics = calculate_metrics(
        comparison_df['true_alpha'],
        comparison_df['inferred_alpha'],
        'codeml',
        'alpha'
    )
    
    # ML Rho
    ml_rho_metrics = calculate_metrics(
        comparison_df['true_rho'],
        comparison_df['ml_rho'],
        'ML',
        'rho'
    )
    
    # Codeml Rho
    codeml_rho_metrics = calculate_metrics(
        comparison_df['true_rho'],
        comparison_df['inferred_rho'],
        'codeml',
        'rho'
    )
    
    # Print overall results
    print(f"\nAlpha:")
    print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"  {'ML':<10} {ml_alpha_metrics['rmse']:<10.4f} {ml_alpha_metrics['mae']:<10.4f} {ml_alpha_metrics['r2']:<10.4f}")
    print(f"  {'codeml':<10} {codeml_alpha_metrics['rmse']:<10.4f} {codeml_alpha_metrics['mae']:<10.4f} {codeml_alpha_metrics['r2']:<10.4f}")
    
    print(f"\nRho:")
    print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"  {'ML':<10} {ml_rho_metrics['rmse']:<10.4f} {ml_rho_metrics['mae']:<10.4f} {ml_rho_metrics['r2']:<10.4f}")
    print(f"  {'codeml':<10} {codeml_rho_metrics['rmse']:<10.4f} {codeml_rho_metrics['mae']:<10.4f} {codeml_rho_metrics['r2']:<10.4f}")
    
    # ========================================================================
    # TRAIN TREES (for reference only)
    # ========================================================================
    if len(train_df) > 0:
        print("\n" + "=" * 70)
        print(f"TRAIN TREES (reference only, n={len(train_df)} samples)")
        print("=" * 70)
        print("ML trained on these trees - unfair advantage")
        
        train_metrics = []
        
        # Alpha
        ml_train_alpha = calculate_metrics(train_df['true_alpha'], train_df['ml_alpha'], 'ML', 'alpha')
        codeml_train_alpha = calculate_metrics(train_df['true_alpha'], train_df['inferred_alpha'], 'codeml', 'alpha')
        train_metrics.extend([ml_train_alpha, codeml_train_alpha])
        
        print(f"\nAlpha:")
        print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print(f"  {'ML':<10} {ml_train_alpha['rmse']:<10.4f} {ml_train_alpha['mae']:<10.4f} {ml_train_alpha['r2']:<10.4f}")
        print(f"  {'codeml':<10} {codeml_train_alpha['rmse']:<10.4f} {codeml_train_alpha['mae']:<10.4f} {codeml_train_alpha['r2']:<10.4f}")
        
        # Rho
        ml_train_rho = calculate_metrics(train_df['true_rho'], train_df['ml_rho'], 'ML', 'rho')
        codeml_train_rho = calculate_metrics(train_df['true_rho'], train_df['inferred_rho'], 'codeml', 'rho')
        train_metrics.extend([ml_train_rho, codeml_train_rho])
        
        print(f"\nRho:")
        print(f"  {'Method':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print(f"  {'ML':<10} {ml_train_rho['rmse']:<10.4f} {ml_train_rho['mae']:<10.4f} {ml_train_rho['r2']:<10.4f}")
        print(f"  {'codeml':<10} {codeml_train_rho['rmse']:<10.4f} {codeml_train_rho['mae']:<10.4f} {codeml_train_rho['r2']:<10.4f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = results_dir / 'method_comparison.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Metrics saved to: {metrics_file}")
    
    # Save detailed predictions for inspection
    output_columns = [
        'tree', 'simulation', 'split',
        'true_alpha', 'ml_alpha', 'inferred_alpha',
        'true_rho', 'ml_rho', 'inferred_rho'
    ]
    predictions_file = results_dir / 'detailed_predictions.csv'
    comparison_df[output_columns].to_csv(predictions_file, index=False)
    print(f"  Detailed predictions saved to: {predictions_file}")
    
    # Calculate coverage (what % of samples have codeml predictions)
    total_samples = len(merged)
    codeml_coverage = len(comparison_df) / total_samples * 100
    print(f"\n  Codeml coverage: {len(comparison_df)}/{total_samples} samples ({codeml_coverage:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()