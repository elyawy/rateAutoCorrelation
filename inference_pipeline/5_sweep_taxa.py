"""
Step 5: Sweep number of taxa and evaluate model performance at each setting.

For each fixed taxon count:
  - Simulate MSAs with exactly that many taxa
  - Evaluate trained models
  - Collect MSE for alpha and rho

Saves a summary CSV and MSE-vs-taxa plots.
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
SEQ_LENGTH = 200

TAXA_TO_SWEEP = [5, 10, 20, 40, 80, 160, 200]


def main():
    models_dir = pathlib.Path("models")
    output_dir = pathlib.Path("results/sweep_taxa")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NUMBER OF TAXA SWEEP")
    print("=" * 60)
    print(f"Taxa counts: {TAXA_TO_SWEEP}")
    print(f"Fixed sequence length: {SEQ_LENGTH} aa")
    print(f"Trees per count: {N_TREES}, MSAs per tree: {N_MSAS_PER_TREE}")
    print(f"Total MSAs per count: {N_TREES * N_MSAS_PER_TREE}")
    print()

    summary_rows = []

    for n_taxa in TAXA_TO_SWEEP:
        print(f"\n--- Taxa count: {n_taxa} ---")

        sim_params = SimulationParams(
            min_taxa=n_taxa,
            max_taxa=n_taxa,
            min_seq_length=SEQ_LENGTH,
            max_seq_length=SEQ_LENGTH,
        )

        df, results = run_evaluation(sim_params, N_TREES, N_MSAS_PER_TREE, SWEEP_SEED, models_dir)

        df.to_csv(output_dir / f"predictions_taxa{n_taxa}.csv", index=False)

        row = {'n_taxa': n_taxa}
        for model_type, metrics in results.items():
            row[f'{model_type}_mse_alpha'] = metrics['mse_alpha']
            row[f'{model_type}_mse_rho'] = metrics['mse_rho']
        summary_rows.append(row)

        for model_type, metrics in results.items():
            print(f"  {model_type}: alpha MSE={metrics['mse_alpha']:.6f}, rho MSE={metrics['mse_rho']:.6f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    _plot_sweep(summary_df, output_dir)

    print("=" * 60)
    print(f"SWEEP COMPLETE — results in {output_dir}/")


def _plot_sweep(summary_df, output_dir):
    """Plot MSE vs number of taxa for all models, one figure per parameter."""
    model_types = [c.replace('_mse_alpha', '') for c in summary_df.columns if c.endswith('_mse_alpha')]

    for param in ['alpha', 'rho']:
        fig, ax = plt.subplots(figsize=(9, 5))

        for model_type in model_types:
            col = f'{model_type}_mse_{param}'
            if col not in summary_df.columns:
                continue
            ax.plot(
                summary_df['n_taxa'],
                summary_df[col],
                marker='o',
                label=model_type.replace('_', ' ').title()
            )

        ax.set_xlabel('Number of taxa', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(f'MSE vs Number of Taxa — {param.upper()} ({SEQ_LENGTH} aa)', fontsize=14)
        ax.set_xscale('log')
        ax.set_xticks(summary_df['n_taxa'])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_file = output_dir / f"sweep_{param}_mse.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_file}")


if __name__ == "__main__":
    main()
