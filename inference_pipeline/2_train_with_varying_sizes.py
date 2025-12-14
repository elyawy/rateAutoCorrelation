"""
Step 2b: Train ML models with varying training set sizes.

Trains models with different numbers of simulations per tree:
  - 50 simulations per tree (2,500 samples with 50 training trees)
  - 100 simulations per tree (5,000 samples with 50 training trees)
  - 200 simulations per tree (10,000 samples with 50 training trees)

Compares performance to identify if there's a learning plateau.

Input:
  - results/features.csv (all features)
  - ../simulated_data/{tree}/ground_truth.csv (true alpha/rho values)
  - ../results/inferred_parameters.csv (codeml results, for test set filtering)

Output:
  - results/alpha_model_{size}.pkl (trained models for each size)
  - results/rho_model_{size}.pkl (trained models for each size)
  - results/training_size_comparison.csv (performance metrics)
  - results/learning_curve.png (visualization)
"""

import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import sys

# Import local config (inference_pipeline/config.py)
import config as local_config


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


def get_complete_codeml_trees(codeml_results_file, required_sims=50):
    """
    Identify trees where codeml completed all required simulations.
    
    Args:
        codeml_results_file: Path to inferred_parameters.csv
        required_sims: Number of simulations expected per tree (default 50)
        
    Returns:
        set: Tree names with complete codeml results
    """
    if not codeml_results_file.exists():
        print(f"  Warning: codeml results file not found: {codeml_results_file}")
        print(f"  Proceeding without filtering test trees.")
        return None
    
    codeml_df = pd.read_csv(codeml_results_file)
    
    # Count simulations per tree
    tree_counts = codeml_df.groupby('tree').size()
    
    # Filter to trees with complete results
    complete_trees = set(tree_counts[tree_counts >= required_sims].index)
    
    print(f"  Found {len(complete_trees)} trees with complete codeml results ({required_sims}+ simulations)")
    
    return complete_trees


def filter_to_n_sims(df, n_sims):
    """
    Filter dataframe to use only first N simulations per tree.
    
    Assumes simulation names are sortable (e.g., sim_001, sim_002, ...).
    """
    def get_first_n_sims(group):
        # Sort by simulation name and take first n_sims
        return group.sort_values('simulation').head(n_sims)
    
    return df.groupby('tree', group_keys=False).apply(get_first_n_sims).reset_index(drop=True)


def train_model_for_size(X_train, X_test, y_train, y_test, train_groups, 
                         param_name, n_sims, random_state=42):
    """
    Train a RandomForestRegressor with hyperparameter tuning.
    
    Returns:
        - Best model
        - Metrics dict
        - Best parameters
    """
    
    # Define parameter grid (simplified for faster runs)
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt']
    }
    
    # Base model
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # GroupKFold to keep trees together (5-fold CV)
    gkf = GroupKFold(n_splits=5)
    
    print(f"    Running GridSearchCV for {param_name} with {len(X_train)} training samples...")
    
    # Grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=gkf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit on training data with group information
    grid_search.fit(X_train, y_train, groups=train_groups)
    
    # Best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = -grid_search.best_score_  # Convert back to MSE
    
    # Predict on train and test
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'n_sims_per_tree': n_sims,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'parameter': param_name,
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'cv_mse': best_cv_score,
        'cv_rmse': np.sqrt(best_cv_score)
    }
    
    return best_model, metrics, best_params


def create_learning_curves(all_metrics, output_file):
    """Create visualization showing performance vs training size."""
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(all_metrics)
    
    # Separate by parameter
    alpha_metrics = metrics_df[metrics_df['parameter'] == 'alpha'].sort_values('n_train_samples')
    rho_metrics = metrics_df[metrics_df['parameter'] == 'rho'].sort_values('n_train_samples')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Alpha - RMSE
    axes[0, 0].plot(alpha_metrics['n_train_samples'], alpha_metrics['train_rmse'], 
                    'o-', label='Train', linewidth=2, markersize=8)
    axes[0, 0].plot(alpha_metrics['n_train_samples'], alpha_metrics['test_rmse'], 
                    's-', label='Test (held-out trees)', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Training Samples')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Alpha: RMSE vs Training Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha - R²
    axes[0, 1].plot(alpha_metrics['n_train_samples'], alpha_metrics['train_r2'], 
                    'o-', label='Train', linewidth=2, markersize=8)
    axes[0, 1].plot(alpha_metrics['n_train_samples'], alpha_metrics['test_r2'], 
                    's-', label='Test (held-out trees)', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Training Samples')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('Alpha: R² vs Training Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rho - RMSE
    axes[1, 0].plot(rho_metrics['n_train_samples'], rho_metrics['train_rmse'], 
                    'o-', label='Train', linewidth=2, markersize=8)
    axes[1, 0].plot(rho_metrics['n_train_samples'], rho_metrics['test_rmse'], 
                    's-', label='Test (held-out trees)', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Training Samples')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Rho: RMSE vs Training Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rho - R²
    axes[1, 1].plot(rho_metrics['n_train_samples'], rho_metrics['train_r2'], 
                    'o-', label='Train', linewidth=2, markersize=8)
    axes[1, 1].plot(rho_metrics['n_train_samples'], rho_metrics['test_r2'], 
                    's-', label='Test (held-out trees)', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Training Samples')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Rho: R² vs Training Size')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Learning curves saved to: {output_file}")
    plt.close()


def main():
    """Main function to train models with varying training sizes."""
    
    # Paths
    results_dir = pathlib.Path('results')
    features_file = results_dir / 'features.csv'
    simulated_data_dir = local_config.SIMULATED_DATA_DIR
    codeml_results_file = pathlib.Path('..') / 'results' / 'inferred_parameters.csv'
    
    # Check if features file exists
    if not features_file.exists():
        print(f"Error: {features_file} does not exist. Run step 1 first.")
        return
    
    print("=" * 70)
    print("TRAINING WITH VARYING DATASET SIZES")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Training trees: {local_config.N_TRAIN_TREES}")
    print(f"  Filter test trees to complete codeml: {local_config.USE_COMPLETE_TEST_TREES_ONLY}")
    print(f"  Training sizes to test: 50, 100, 200 simulations per tree")
    print()
    
    print("Loading data...")
    print("-" * 70)
    
    # Load features
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df)} feature rows")
    
    # Load ground truth
    ground_truth = load_ground_truth(simulated_data_dir)
    print(f"Loaded {len(ground_truth)} ground truth rows")
    
    # Merge on tree and simulation
    merged = pd.merge(
        features_df,
        ground_truth,
        on=['tree', 'simulation'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged)} rows")
    
    if len(merged) == 0:
        print("Error: No matching data found after merge.")
        return
    
    # Split trees into train/test sets
    all_trees = sorted(merged['tree'].unique())
    n_train_trees = local_config.N_TRAIN_TREES
    
    if n_train_trees >= len(all_trees):
        print(f"\nError: N_TRAIN_TREES ({n_train_trees}) >= total trees ({len(all_trees)})")
        return
    
    train_trees = all_trees[:n_train_trees]
    test_trees = all_trees[n_train_trees:]
    
    # Filter test trees to those with complete codeml results (if enabled)
    if local_config.USE_COMPLETE_TEST_TREES_ONLY:
        print(f"\nFiltering test trees to those with complete codeml results...")
        complete_trees = get_complete_codeml_trees(codeml_results_file, required_sims=50)
        
        if complete_trees is not None:
            # Keep only test trees that have complete codeml results
            test_trees_filtered = [t for t in test_trees if t in complete_trees]
            print(f"  Test trees before filtering: {len(test_trees)}")
            print(f"  Test trees after filtering: {len(test_trees_filtered)}")
            test_trees = test_trees_filtered
            
            if len(test_trees) == 0:
                print("\nWarning: No test trees have complete codeml results.")
                print("Proceeding with all test trees.")
                test_trees = all_trees[n_train_trees:]
    
    print(f"\nFinal tree split:")
    print(f"  Training trees: {len(train_trees)}")
    print(f"  Test trees: {len(test_trees)}")
    
    # Feature columns
    feature_columns = local_config.FEATURE_COLUMNS
    print(f"\nUsing {len(feature_columns)} features")
    
    # Train models for different sizes
    training_sizes = [400]
    all_metrics = []
    all_best_params = {}
    
    for n_sims in training_sizes:
        print("\n" + "=" * 70)
        print(f"TRAINING WITH {n_sims} SIMULATIONS PER TREE")
        print("=" * 70)
        
        # Filter to first n_sims per tree
        train_data = merged[merged['tree'].isin(train_trees)].copy()
        train_data = filter_to_n_sims(train_data, n_sims)
        
        # For test data, always use first 50 sims (for consistency with codeml)
        test_data = merged[merged['tree'].isin(test_trees)].copy()
        test_data = filter_to_n_sims(test_data, 50)
        
        print(f"\nData split for this size:")
        print(f"  Training samples: {len(train_data)} ({len(train_trees)} trees × {n_sims} sims)")
        print(f"  Test samples: {len(test_data)} ({len(test_trees)} trees × 50 sims)")
        
        # Prepare features and targets
        X_train = train_data[feature_columns].values
        y_alpha_train = train_data['true_alpha'].values
        y_rho_train = train_data['true_rho'].values
        train_groups = train_data['tree'].values
        
        X_test = test_data[feature_columns].values
        y_alpha_test = test_data['true_alpha'].values
        y_rho_test = test_data['true_rho'].values
        
        print(f"\n  Training Alpha model...")
        alpha_model, alpha_metrics, alpha_best_params = train_model_for_size(
            X_train, X_test, y_alpha_train, y_alpha_test,
            train_groups, 'alpha', n_sims, random_state=42
        )
        
        print(f"    CV RMSE: {alpha_metrics['cv_rmse']:.4f}")
        print(f"    Test RMSE: {alpha_metrics['test_rmse']:.4f}, R²: {alpha_metrics['test_r2']:.4f}")
        
        print(f"\n  Training Rho model...")
        rho_model, rho_metrics, rho_best_params = train_model_for_size(
            X_train, X_test, y_rho_train, y_rho_test,
            train_groups, 'rho', n_sims, random_state=42
        )
        
        print(f"    CV RMSE: {rho_metrics['cv_rmse']:.4f}")
        print(f"    Test RMSE: {rho_metrics['test_rmse']:.4f}, R²: {rho_metrics['test_r2']:.4f}")
        
        # Save models
        alpha_model_path = results_dir / f'alpha_model_{n_sims}.pkl'
        rho_model_path = results_dir / f'rho_model_{n_sims}.pkl'
        
        with open(alpha_model_path, 'wb') as f:
            pickle.dump(alpha_model, f)
        with open(rho_model_path, 'wb') as f:
            pickle.dump(rho_model, f)
        
        print(f"\n  Models saved:")
        print(f"    {alpha_model_path}")
        print(f"    {rho_model_path}")
        
        # Store metrics
        all_metrics.extend([alpha_metrics, rho_metrics])
        all_best_params[f'alpha_{n_sims}'] = alpha_best_params
        all_best_params[f'rho_{n_sims}'] = rho_best_params
    
    # Save comparison results
    print("\n" + "=" * 70)
    print("SAVING COMPARISON RESULTS")
    print("=" * 70)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    comparison_file = results_dir / 'training_size_comparison.csv'
    metrics_df.to_csv(comparison_file, index=False)
    print(f"  Comparison metrics saved to: {comparison_file}")
    
    # Create learning curves
    plot_file = results_dir / 'learning_curve.png'
    create_learning_curves(all_metrics, plot_file)
    
    # Save best parameters
    params_file = results_dir / 'best_params_by_size.txt'
    with open(params_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BEST HYPERPARAMETERS BY TRAINING SIZE\n")
        f.write("=" * 70 + "\n\n")
        for key, params in all_best_params.items():
            f.write(f"{key}:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            
            # Find corresponding metrics
            param_type = 'alpha' if 'alpha' in key else 'rho'
            n_sims = int(key.split('_')[1])
            metric = [m for m in all_metrics if m['parameter'] == param_type and m['n_sims_per_tree'] == n_sims][0]
            f.write(f"  CV RMSE: {metric['cv_rmse']:.6f}\n")
            f.write(f"  Test RMSE: {metric['test_rmse']:.6f}\n\n")
    
    print(f"  Best parameters saved to: {params_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Display metrics table
    summary = metrics_df[['parameter', 'n_sims_per_tree', 'n_train_samples', 
                          'test_rmse', 'test_r2', 'test_mae']].copy()
    summary = summary.sort_values(['parameter', 'n_sims_per_tree'])
    print("\nTest Set Performance:")
    print(summary.to_string(index=False))
    
    # Check for plateau
    print("\n" + "-" * 70)
    print("Plateau Analysis:")
    print("-" * 70)
    
    for param in ['alpha', 'rho']:
        param_metrics = metrics_df[metrics_df['parameter'] == param].sort_values('n_train_samples')
        rmse_values = param_metrics['test_rmse'].values
        
        # Calculate improvement from smallest to largest
        improvement_50_to_100 = (rmse_values[0] - rmse_values[1]) / rmse_values[0] * 100
        improvement_100_to_200 = (rmse_values[1] - rmse_values[2]) / rmse_values[1] * 100
        
        print(f"\n{param.upper()}:")
        print(f"  50→100 sims: {improvement_50_to_100:+.2f}% RMSE change")
        print(f"  100→200 sims: {improvement_100_to_200:+.2f}% RMSE change")
        
        if abs(improvement_100_to_200) < 2:
            print(f"  → Appears to plateau after 100 simulations")
        elif improvement_100_to_200 < improvement_50_to_100 / 2:
            print(f"  → Diminishing returns, approaching plateau")
        else:
            print(f"  → Still improving, not yet at plateau")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
