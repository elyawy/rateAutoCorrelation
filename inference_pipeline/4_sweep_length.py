"""
Step 4: Sweep sequence length and evaluate model performance at each setting.

For each fixed sequence length:
  - Simulate MSAs of exactly that length
  - Evaluate trained models
  - Collect MSE for alpha and rho

Saves a summary CSV and a MSE-vs-length plot.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.simulation import SimulationParams
from utils.evaluate import run_evaluation


# ==========================================
# SWEEP CONFIGURATION
# ==========================================
SWEEP_SEED = 123
N_TREES = 10
N_MSAS_PER_TREE = 10

# Fixed lengths to evaluate (min == max forces a single length per run)
LENGTHS_TO_SWEEP = [50, 100, 250, 500, 1000, 2500, 5000, 10000]

# Taxa range stays constant across the sweep
MIN_TAXA = 10
MAX_TAXA = 10


def main():
    models_dir = pathlib.Path("models")
    output_dir = pathlib.Path("results/sweep_length")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SEQUENCE LENGTH SWEEP")
    print("=" * 60)
    print(f"Lengths: {LENGTHS_TO_SWEEP}")
    print(f"Trees per length: {N_TREES}, MSAs per tree: {N_MSAS_PER_TREE}")
    print(f"Total MSAs per length: {N_TREES * N_MSAS_PER_TREE}")
    print()

    summary_rows = []

    for length in LENGTHS_TO_SWEEP:
        print(f"\n--- Sequence length: {length} ---")

        sim_params = SimulationParams(
            min_taxa=MIN_TAXA,
            max_taxa=MAX_TAXA,
            min_seq_length=length,
            max_seq_length=length,
        )

        df, results = run_evaluation(sim_params, N_TREES, N_MSAS_PER_TREE, SWEEP_SEED, models_dir)

        # Save per-length predictions
        df.to_csv(output_dir / f"predictions_len{length}.csv", index=False)

        # Collect summary row
        row = {'seq_length': length}
        for model_type, metrics in results.items():
            row[f'{model_type}_mse_alpha'] = metrics['mse_alpha']
            row[f'{model_type}_mse_rho'] = metrics['mse_rho']
        summary_rows.append(row)

        for model_type, metrics in results.items():
            print(f"  {model_type}: alpha MSE={metrics['mse_alpha']:.6f}, rho MSE={metrics['mse_rho']:.6f}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    # Plot MSE vs sequence length
    _plot_sweep(summary_df, output_dir)

    print("=" * 60)
    print(f"SWEEP COMPLETE — results in {output_dir}/")


def _plot_sweep(summary_df, output_dir):
    """Plot MSE vs sequence length for all models, one figure per parameter."""
    model_types = [c.replace('_mse_alpha', '') for c in summary_df.columns if c.endswith('_mse_alpha')]

    for param in ['alpha', 'rho']:
        fig, ax = plt.subplots(figsize=(9, 5))

        for model_type in model_types:
            col = f'{model_type}_mse_{param}'
            if col not in summary_df.columns:
                continue
            ax.plot(
                summary_df['seq_length'],
                summary_df[col],
                marker='o',
                label=model_type.replace('_', ' ').title()
            )

        ax.set_xlabel('Sequence length (aa)', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(f'MSE vs Sequence Length — {param.upper()}', fontsize=14)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_file = output_dir / f"sweep_{param}_mse.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_file}")


if __name__ == "__main__":
    main()