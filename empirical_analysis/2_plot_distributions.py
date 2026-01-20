"""
Step 2: Plot distributions of inferred alpha and rho parameters.

Creates separate histograms for alpha and rho from OrthoMaM predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config


def plot_distribution(values, param_name, output_file, bins=50):
    """
    Create a histogram for a parameter distribution.
    
    Args:
        values: Array of parameter values
        param_name: Name of parameter ('Alpha' or 'Rho')
        output_file: Path to save the plot
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins_edges, patches = ax.hist(
        values, 
        bins=bins, 
        density=True,
        alpha=0.7, 
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add labels and title
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Distribution of {param_name} (OrthoMaM)', fontsize=16, fontweight='bold')
    
    # Add statistics text box
    stats_text = f'n = {len(values)}\n'
    stats_text += f'Mean = {np.mean(values):.3f}\n'
    stats_text += f'Median = {np.median(values):.3f}\n'
    stats_text += f'Std = {np.std(values):.3f}\n'
    stats_text += f'Min = {np.min(values):.3f}\n'
    stats_text += f'Max = {np.max(values):.3f}'
    
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def main():
    """Main plotting function."""
    print("=" * 60)
    print("PLOTTING DISTRIBUTIONS")
    print("=" * 60)
    
    # Load predictions
    predictions_file = config.RESULTS_DIR / "orthomam_predictions.csv"
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print("Run 1_process_orthomam.py first.")
        return
    
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    print(f"Loaded {len(df)} predictions")
    print()
    
    # Create plots directory
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot alpha distribution
    print("Creating Alpha distribution plot...")
    alpha_plot = config.PLOTS_DIR / "alpha_distribution.png"
    plot_distribution(
        df['pred_alpha'].values, 
        'Alpha', 
        alpha_plot,
        bins=50
    )
    
    # Plot rho distribution
    print("Creating Rho distribution plot...")
    rho_plot = config.PLOTS_DIR / "rho_distribution.png"
    plot_distribution(
        df['pred_rho'].values, 
        'Rho', 
        rho_plot,
        bins=50
    )
    
    print()
    print("=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)
    print(f"Plots saved to: {config.PLOTS_DIR}/")
    print()
    print("Summary statistics:")
    print("\nAlpha:")
    print(df['pred_alpha'].describe())
    print("\nRho:")
    print(df['pred_rho'].describe())


if __name__ == "__main__":
    main()
