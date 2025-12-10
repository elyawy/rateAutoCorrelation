"""
Step 4: Analyze per-tree prediction errors to identify problematic datasets.

Investigates where the ML models struggle by:
  1. Calculating per-sample and per-tree errors
  2. Identifying worst-performing trees
  3. Analyzing error patterns (extremes, feature correlations)
  4. Generating diagnostic plots and summaries

Input:
  - results/detailed_predictions.csv (from step 3)
  - results/features.csv (from step 1)

Output:
  - results/per_tree_errors.csv (error statistics per tree)
  - results/error_analysis.txt (summary report)
  - results/error_diagnostics.png (visualization)
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config as local_config


def calculate_errors(df):
    """Add error columns to dataframe."""
    df['alpha_error'] = np.abs(df['ml_alpha'] - df['true_alpha'])
    df['rho_error'] = np.abs(df['ml_rho'] - df['true_rho'])
    df['alpha_squared_error'] = (df['ml_alpha'] - df['true_alpha']) ** 2
    df['rho_squared_error'] = (df['ml_rho'] - df['true_rho']) ** 2
    return df


def analyze_per_tree_errors(df):
    """Calculate error statistics per tree."""
    tree_stats = df.groupby('tree').agg({
        'alpha_error': ['mean', 'std', 'max'],
        'rho_error': ['mean', 'std', 'max'],
        'alpha_squared_error': 'mean',
        'rho_squared_error': 'mean',
        'true_alpha': 'mean',
        'true_rho': 'mean',
        'simulation': 'count'
    }).reset_index()
    
    # Flatten column names
    tree_stats.columns = [
        'tree', 
        'alpha_mae', 'alpha_std', 'alpha_max_error',
        'rho_mae', 'rho_std', 'rho_max_error',
        'alpha_mse', 'rho_mse',
        'avg_true_alpha', 'avg_true_rho',
        'n_samples'
    ]
    
    # Calculate RMSE
    tree_stats['alpha_rmse'] = np.sqrt(tree_stats['alpha_mse'])
    tree_stats['rho_rmse'] = np.sqrt(tree_stats['rho_mse'])
    
    return tree_stats


def identify_worst_trees(tree_stats, n=10):
    """Identify trees with highest errors."""
    worst_alpha = tree_stats.nlargest(n, 'alpha_mae')[
        ['tree', 'alpha_mae', 'alpha_rmse', 'avg_true_alpha', 'n_samples']
    ]
    
    worst_rho = tree_stats.nlargest(n, 'rho_mae')[
        ['tree', 'rho_mae', 'rho_rmse', 'avg_true_rho', 'n_samples']
    ]
    
    return worst_alpha, worst_rho


def analyze_error_vs_true_values(df):
    """Analyze if errors correlate with extreme true values."""
    # Bin true values
    df['alpha_bin'] = pd.cut(df['true_alpha'], bins=5, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
    df['rho_bin'] = pd.cut(df['true_rho'], bins=5, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
    
    alpha_by_bin = df.groupby('alpha_bin')['alpha_error'].agg(['mean', 'std', 'count'])
    rho_by_bin = df.groupby('rho_bin')['rho_error'].agg(['mean', 'std', 'count'])
    
    return alpha_by_bin, rho_by_bin


def merge_with_features(tree_stats, features_df):
    """Merge tree errors with average features per tree."""
    # Calculate average features per tree
    feature_cols = local_config.FEATURE_COLUMNS
    tree_features = features_df.groupby('tree')[feature_cols].mean().reset_index()
    
    # Merge
    merged = pd.merge(tree_stats, tree_features, on='tree', how='left')
    
    return merged


def calculate_feature_correlations(merged_df):
    """Calculate correlations between errors and features."""
    feature_cols = local_config.FEATURE_COLUMNS
    
    alpha_corr = merged_df[['alpha_mae'] + feature_cols].corr()['alpha_mae'].drop('alpha_mae').sort_values(ascending=False)
    rho_corr = merged_df[['rho_mae'] + feature_cols].corr()['rho_mae'].drop('rho_mae').sort_values(ascending=False)
    
    return alpha_corr, rho_corr


def create_diagnostic_plots(df, tree_stats, output_file):
    """Create visualization of error patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Error vs true value (Alpha)
    axes[0, 0].scatter(df['true_alpha'], df['alpha_error'], alpha=0.3, s=10)
    axes[0, 0].set_xlabel('True Alpha')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].set_title('Alpha: Error vs True Value')
    axes[0, 0].axhline(y=df['alpha_error'].mean(), color='r', linestyle='--', label='Mean Error')
    axes[0, 0].legend()
    
    # 2. Error vs true value (Rho)
    axes[0, 1].scatter(df['true_rho'], df['rho_error'], alpha=0.3, s=10)
    axes[0, 1].set_xlabel('True Rho')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Rho: Error vs True Value')
    axes[0, 1].axhline(y=df['rho_error'].mean(), color='r', linestyle='--', label='Mean Error')
    axes[0, 1].legend()
    
    # 3. Prediction vs true (Alpha)
    axes[0, 2].scatter(df['true_alpha'], df['ml_alpha'], alpha=0.3, s=10)
    axes[0, 2].plot([df['true_alpha'].min(), df['true_alpha'].max()], 
                     [df['true_alpha'].min(), df['true_alpha'].max()], 
                     'r--', label='Perfect prediction')
    axes[0, 2].set_xlabel('True Alpha')
    axes[0, 2].set_ylabel('Predicted Alpha')
    axes[0, 2].set_title('Alpha: Predicted vs True')
    axes[0, 2].legend()
    
    # 4. Prediction vs true (Rho)
    axes[1, 0].scatter(df['true_rho'], df['ml_rho'], alpha=0.3, s=10)
    axes[1, 0].plot([df['true_rho'].min(), df['true_rho'].max()], 
                     [df['true_rho'].min(), df['true_rho'].max()], 
                     'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('True Rho')
    axes[1, 0].set_ylabel('Predicted Rho')
    axes[1, 0].set_title('Rho: Predicted vs True')
    axes[1, 0].legend()
    
    # 5. Distribution of errors (Alpha)
    axes[1, 1].hist(df['alpha_error'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=df['alpha_error'].mean(), color='r', linestyle='--', label=f"Mean: {df['alpha_error'].mean():.3f}")
    axes[1, 1].axvline(x=df['alpha_error'].median(), color='g', linestyle='--', label=f"Median: {df['alpha_error'].median():.3f}")
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Alpha Error Distribution')
    axes[1, 1].legend()
    
    # 6. Distribution of errors (Rho)
    axes[1, 2].hist(df['rho_error'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(x=df['rho_error'].mean(), color='r', linestyle='--', label=f"Mean: {df['rho_error'].mean():.3f}")
    axes[1, 2].axvline(x=df['rho_error'].median(), color='g', linestyle='--', label=f"Median: {df['rho_error'].median():.3f}")
    axes[1, 2].set_xlabel('Absolute Error')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Rho Error Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Diagnostic plots saved to: {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    results_dir = pathlib.Path('results')
    predictions_file = results_dir / 'detailed_predictions.csv'
    features_file = results_dir / 'features.csv'
    
    # Check files exist
    if not predictions_file.exists():
        print(f"Error: {predictions_file} does not exist. Run step 3 first.")
        return
    
    if not features_file.exists():
        print(f"Error: {features_file} does not exist. Run step 1 first.")
        return
    
    print("Loading data...")
    print("=" * 70)
    
    # Load predictions
    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df)} predictions")
    
    # Load features
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df)} feature rows")
    
    # Calculate errors
    df = calculate_errors(df)
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL ERROR STATISTICS")
    print("=" * 70)
    print(f"\nAlpha:")
    print(f"  Mean Absolute Error: {df['alpha_error'].mean():.4f}")
    print(f"  Median Absolute Error: {df['alpha_error'].median():.4f}")
    print(f"  Std Dev of Errors: {df['alpha_error'].std():.4f}")
    print(f"  Max Error: {df['alpha_error'].max():.4f}")
    print(f"  RMSE: {np.sqrt(df['alpha_squared_error'].mean()):.4f}")
    
    print(f"\nRho:")
    print(f"  Mean Absolute Error: {df['rho_error'].mean():.4f}")
    print(f"  Median Absolute Error: {df['rho_error'].median():.4f}")
    print(f"  Std Dev of Errors: {df['rho_error'].std():.4f}")
    print(f"  Max Error: {df['rho_error'].max():.4f}")
    print(f"  RMSE: {np.sqrt(df['rho_squared_error'].mean()):.4f}")
    
    # Per-tree analysis
    print("\n" + "=" * 70)
    print("PER-TREE ERROR ANALYSIS")
    print("=" * 70)
    
    tree_stats = analyze_per_tree_errors(df)
    print(f"\nAnalyzed {len(tree_stats)} trees")
    
    # Identify worst trees
    worst_alpha, worst_rho = identify_worst_trees(tree_stats, n=10)
    
    print("\n" + "-" * 70)
    print("TOP 10 WORST TREES FOR ALPHA PREDICTION:")
    print("-" * 70)
    print(worst_alpha.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("TOP 10 WORST TREES FOR RHO PREDICTION:")
    print("-" * 70)
    print(worst_rho.to_string(index=False))
    
    # Error vs true value bins
    print("\n" + "=" * 70)
    print("ERROR BY TRUE VALUE RANGE")
    print("=" * 70)
    
    alpha_by_bin, rho_by_bin = analyze_error_vs_true_values(df)
    
    print("\nAlpha errors by true value range:")
    print(alpha_by_bin.to_string())
    
    print("\nRho errors by true value range:")
    print(rho_by_bin.to_string())
    
    # Merge with features
    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS WITH ERRORS")
    print("=" * 70)
    
    merged = merge_with_features(tree_stats, features_df)
    alpha_corr, rho_corr = calculate_feature_correlations(merged)
    
    print("\nFeatures most correlated with ALPHA errors:")
    print(alpha_corr.to_string())
    
    print("\nFeatures most correlated with RHO errors:")
    print(rho_corr.to_string())
    
    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save per-tree errors
    output_file = results_dir / 'per_tree_errors.csv'
    merged.to_csv(output_file, index=False)
    print(f"  Per-tree errors saved to: {output_file}")
    
    # Create diagnostic plots
    plot_file = results_dir / 'error_diagnostics.png'
    create_diagnostic_plots(df, tree_stats, plot_file)
    
    # Save text report
    report_file = results_dir / 'error_analysis.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Alpha MAE: {df['alpha_error'].mean():.4f}\n")
        f.write(f"Alpha RMSE: {np.sqrt(df['alpha_squared_error'].mean()):.4f}\n")
        f.write(f"Rho MAE: {df['rho_error'].mean():.4f}\n")
        f.write(f"Rho RMSE: {np.sqrt(df['rho_squared_error'].mean()):.4f}\n\n")
        
        f.write("WORST TREES (Alpha)\n")
        f.write("-" * 70 + "\n")
        f.write(worst_alpha.to_string(index=False) + "\n\n")
        
        f.write("WORST TREES (Rho)\n")
        f.write("-" * 70 + "\n")
        f.write(worst_rho.to_string(index=False) + "\n\n")
        
        f.write("FEATURE CORRELATIONS WITH ALPHA ERRORS\n")
        f.write("-" * 70 + "\n")
        f.write(alpha_corr.to_string() + "\n\n")
        
        f.write("FEATURE CORRELATIONS WITH RHO ERRORS\n")
        f.write("-" * 70 + "\n")
        f.write(rho_corr.to_string() + "\n")
    
    print(f"  Text report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review error_diagnostics.png for visual patterns")
    print("  2. Check per_tree_errors.csv for detailed per-tree statistics")
    print("  3. Investigate worst-performing trees in your data")
    print("  4. Look for patterns in feature correlations")


if __name__ == "__main__":
    main()