"""
Core evaluation logic: simulate MSAs, extract features, run trained models.
"""

import math
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import config
from features_calculator import calculate_msa_entropy_stats
from utils.simulation import SimulationParams, generate_random_tree, setup_sim, simulate_msa


def run_evaluation(sim_params: SimulationParams, n_trees: int, n_msas_per_tree: int,
                   seed: int, models_dir) -> tuple:
    """
    Generate MSAs, extract features, and evaluate all available trained models.

    Args:
        sim_params: SimulationParams controlling tree/sequence shape
        n_trees: Number of random trees to generate
        n_msas_per_tree: Number of MSAs to simulate per tree
        seed: Master random seed
        models_dir: Path to directory containing trained model .pkl files

    Returns:
        tuple: (df, results)
            df: DataFrame with true values, predicted values, and features
            results: dict {model_type: {'mse_alpha': float, 'mse_rho': float}}
    """
    random.seed(seed)
    np.random.seed(seed)

    all_data = []

    for tree_idx in range(n_trees):
        n_taxa = random.randint(sim_params.min_taxa, sim_params.max_taxa)
        tree_seed = seed + tree_idx * 1000

        random_scale = 10 ** random.uniform(math.log10(config.SCALE_MIN), math.log10(config.SCALE_MAX))

        tree = generate_random_tree(n_taxa, random_scale, tree_seed)
        tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"

        random.seed(tree_seed)
        np.random.seed(tree_seed)
        simulator = setup_sim(tree, sim_seed=tree_seed)

        for msa_idx in range(n_msas_per_tree):
            sequences, true_alpha, true_rho = simulate_msa(simulator, sim_params)
            entropy_stats = calculate_msa_entropy_stats(sequences)

            all_data.append({
                'tree': tree_name,
                'msa_idx': msa_idx,
                'n_taxa': n_taxa,
                'true_alpha': true_alpha,
                'true_rho': true_rho,
                **{col: entropy_stats[col] for col in config.FEATURE_COLUMNS if col in entropy_stats}
            })

    df = pd.DataFrame(all_data)
    X = df[config.FEATURE_COLUMNS].values
    results = {}

    for model_type in ['neural_net', 'lightgbm']:
        model_file = models_dir / f"{model_type}_model.pkl"
        if not model_file.exists():
            print(f"  WARNING: {model_file} not found. Skipping {model_type}.")
            continue

        model = joblib.load(model_file)
        predictions = model.predict(X)

        df[f'pred_alpha_{model_type}'] = predictions[:, 0]
        df[f'pred_rho_{model_type}'] = predictions[:, 1]

        results[model_type] = {
            'mse_alpha': float(np.mean((df['true_alpha'] - predictions[:, 0]) ** 2)),
            'mse_rho': float(np.mean((df['true_rho'] - predictions[:, 1]) ** 2)),
        }


    return df, results


def save_scatter_plots(df: pd.DataFrame, results: dict, plots_dir):
    """Save true-vs-predicted scatter plots for all models and parameters."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_type, metrics in results.items():
        for param in ['alpha', 'rho']:
            true_col = f'true_{param}'
            pred_col = f'pred_{param}_{model_type}'

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(df[true_col], df[pred_col], alpha=0.5, s=20)

            min_val = min(df[true_col].min(), df[pred_col].min())
            max_val = max(df[true_col].max(), df[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

            r2 = np.corrcoef(df[true_col], df[pred_col])[0, 1] ** 2
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')

            ax.set_xlabel(f'True {param}', fontsize=12)
            ax.set_ylabel(f'Predicted {param}', fontsize=12)
            ax.set_title(
                f'{model_type.replace("_", " ").title()} — {param.upper()}\n'
                f'MSE: {metrics[f"mse_{param}"]:.6f}',
                fontsize=14
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_file = plots_dir / f"{model_type}_{param}_scatter.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"  Saved: {plot_file}")
