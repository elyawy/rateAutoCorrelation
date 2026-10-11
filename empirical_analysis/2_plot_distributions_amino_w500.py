"""
Step 2 (protein, w500 only): Plot the distribution of inferred rho, and rho against
alignment length, for genes whose estimate came from the w500 model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `common` imports without PYTHONPATH
import common.config as config

PREDICTIONS_NAME = "orthomam_predictions_amino.csv"
WINDOW = 500


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


def plot_rho_vs_length(df, output_file):
    """Scatter of predicted rho against alignment length, to check estimates stabilize for long genes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['alignment_length'], df['pred_rho'], s=6, alpha=0.3, color='steelblue')
    ax.set_xscale('log')
    ax.axvline(WINDOW, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Alignment length (amino acids)', fontsize=14)
    ax.set_ylabel('Predicted rho', fontsize=14)
    ax.set_title('Predicted rho vs alignment length (OrthoMaM, w500 genes)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
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
    predictions_file = config.RESULTS_DIR / PREDICTIONS_NAME
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print("Run 1_process_orthomam.py first.")
        return
    
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    df = df[df['window_used'] == WINDOW]
    print(f"Loaded {len(df)} predictions that used the w{WINDOW} model")
    print()
    
    # Create plots directory
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot rho distribution
    print("Creating Rho distribution plot...")
    rho_plot = config.PLOTS_DIR / "rho_distribution_amino_w500.png"
    plot_distribution(
        df['pred_rho'].values, 
        'Rho (w500 genes)', 
        rho_plot,
        bins=50
    )
    
    # Plot rho vs alignment length
    print("Creating Rho vs length plot...")
    plot_rho_vs_length(df, config.PLOTS_DIR / "rho_vs_length_amino_w500.png")
    
    print()
    print("=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)
    print(f"Plots saved to: {config.PLOTS_DIR}/")
    print()
    print("Summary statistics:")
    print("\nRho:")
    print(df['pred_rho'].describe())
    print("\nrho_std across chunks:")
    print(df['rho_std'].describe())


if __name__ == "__main__":
    main()
