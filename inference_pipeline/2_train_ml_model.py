"""
Step 2: Train ML models to predict alpha and rho from entropy features.

Trains two separate RandomForestRegressor models:
  - One for predicting alpha
  - One for predicting rho

Input:
  - results/features.csv (statistics)
  - ../simulated_data/{tree}/ground_truth.csv (true alpha/rho values)

Output:
  - results/alpha_model.pkl (trained model)
  - results/rho_model.pkl (trained model)
  - results/ml_evaluation.csv (performance metrics)
"""

import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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


def train_and_evaluate_model(X_train, X_test, y_train, y_test, param_name, random_state=42):
    """
    Train a RandomForestRegressor and return model + metrics.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'parameter': param_name,
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return model, metrics


def main():
    """Main function to train ML models."""
    
    # Paths
    results_dir = pathlib.Path('results')
    entropy_file = results_dir / 'features.csv'
    simulated_data_dir = local_config.SIMULATED_DATA_DIR
    
    # Check if entropy file exists
    if not entropy_file.exists():
        print(f"Error: {entropy_file} does not exist. Run step 1 first.")
        return
    
    print("Loading data...")
    print("=" * 50)
    
    # Load entropy features
    entropy_df = pd.read_csv(entropy_file)
    print(f"Loaded {len(entropy_df)} entropy feature rows")
    
    # Load ground truth
    ground_truth = load_ground_truth(simulated_data_dir)
    print(f"Loaded {len(ground_truth)} ground truth rows")
    
    # Merge on tree and simulation
    merged = pd.merge(
        entropy_df,
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
        print(f"\nWarning: N_TRAIN_TREES ({n_train_trees}) >= total trees ({len(all_trees)})")
        print("Using all trees for training. No held-out test set.")
        train_trees = all_trees
        test_trees = []
    else:
        train_trees = all_trees[:n_train_trees]
        test_trees = all_trees[n_train_trees:]
    
    print(f"\nTree split:")
    print(f"  Total trees: {len(all_trees)}")
    print(f"  Training trees: {len(train_trees)} (first {n_train_trees})")
    print(f"  Test trees: {len(test_trees)} (held-out)")
    print(f"\nTraining trees: {train_trees[:5]}{'...' if len(train_trees) > 5 else ''}")
    if test_trees:
        print(f"Test trees: {test_trees[:5]}{'...' if len(test_trees) > 5 else ''}")
    
    # Filter data by tree membership
    train_data = merged[merged['tree'].isin(train_trees)].copy()
    test_data = merged[merged['tree'].isin(test_trees)].copy() if test_trees else None
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data) if test_data is not None else 0}")
    
    # Prepare features and targets
    feature_columns = [
        'avg_entropy', 'entropy_variance', 'max_entropy', 'lag1_autocorr',
        'avg_parsimony_score', 'var_parsimony_score', 
        'lag1_parsimony_autocorr', 'parsimony_entropy_correlation'
    ]
    
    X_train = train_data[feature_columns].values
    y_alpha_train = train_data['true_alpha'].values
    y_rho_train = train_data['true_rho'].values
    
    if test_data is not None and len(test_data) > 0:
        X_test = test_data[feature_columns].values
        y_alpha_test = test_data['true_alpha'].values
        y_rho_test = test_data['true_rho'].values
    else:
        # No test set - use dummy values (won't be used)
        X_test = X_train[:1]
        y_alpha_test = y_alpha_train[:1]
        y_rho_test = y_rho_train[:1]
    
    print(f"\nFeatures shape: Train {X_train.shape}, Test {X_test.shape if test_data is not None else 'N/A'}")
    print(f"Feature columns: {feature_columns}")
    
    print("\n" + "=" * 50)
    print("Training Alpha Model...")
    print("=" * 50)
    
    alpha_model, alpha_metrics = train_and_evaluate_model(
        X_train, X_test, y_alpha_train, y_alpha_test, 
        'alpha', random_state=42
    )
    
    print(f"\nAlpha Model Performance:")
    print(f"  Train R²: {alpha_metrics['train_r2']:.4f}")
    if test_data is not None and len(test_data) > 0:
        print(f"  Test R² (held-out trees):  {alpha_metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {alpha_metrics['test_rmse']:.4f}")
        print(f"  Test MAE:  {alpha_metrics['test_mae']:.4f}")
    
    print(f"\nFeature Importance (Alpha):")
    for feat, imp in zip(feature_columns, alpha_model.feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")
    
    print("\n" + "=" * 50)
    print("Training Rho Model...")
    print("=" * 50)
    
    rho_model, rho_metrics = train_and_evaluate_model(
        X_train, X_test, y_rho_train, y_rho_test, 
        'rho', random_state=42
    )
    
    print(f"\nRho Model Performance:")
    print(f"  Train R²: {rho_metrics['train_r2']:.4f}")
    if test_data is not None and len(test_data) > 0:
        print(f"  Test R² (held-out trees):  {rho_metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {rho_metrics['test_rmse']:.4f}")
        print(f"  Test MAE:  {rho_metrics['test_mae']:.4f}")
    
    print(f"\nFeature Importance (Rho):")
    for feat, imp in zip(feature_columns, rho_model.feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")
    
    # Save models
    print("\n" + "=" * 50)
    print("Saving models...")
    
    alpha_model_path = results_dir / 'alpha_model.pkl'
    rho_model_path = results_dir / 'rho_model.pkl'
    
    with open(alpha_model_path, 'wb') as f:
        pickle.dump(alpha_model, f)
    print(f"  Alpha model saved to: {alpha_model_path}")
    
    with open(rho_model_path, 'wb') as f:
        pickle.dump(rho_model, f)
    print(f"  Rho model saved to: {rho_model_path}")
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([alpha_metrics, rho_metrics])
    metrics_file = results_dir / 'ml_evaluation.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Evaluation metrics saved to: {metrics_file}")
    
    # Save tree split information
    tree_split_file = results_dir / 'tree_split.csv'
    tree_split_df = pd.DataFrame({
        'tree': all_trees,
        'split': ['train' if t in train_trees else 'test' for t in all_trees]
    })
    tree_split_df.to_csv(tree_split_file, index=False)
    print(f"  Tree split info saved to: {tree_split_file}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()