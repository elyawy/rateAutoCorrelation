"""
Step 3: Evaluate trained models using freshly generated trees.
"""

import pathlib
from utils.simulation import SimulationParams
from utils.evaluate import run_evaluation, save_scatter_plots



# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
VALIDATION_SEED = 42
N_TREES = 50
N_MSAS_PER_TREE = 10

SIM_PARAMS = SimulationParams(
    min_taxa=100,
    max_taxa=100,
    min_seq_length=250,
    max_seq_length=250,
)


def main():
    models_dir = pathlib.Path("models")
    output_dir = pathlib.Path("results/validation")

    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    print(f"Trees: {N_TREES}, MSAs/tree: {N_MSAS_PER_TREE}, seed: {VALIDATION_SEED}")
    print(f"Taxa: {SIM_PARAMS.min_taxa}-{SIM_PARAMS.max_taxa}, "
          f"seq length: {SIM_PARAMS.min_seq_length}-{SIM_PARAMS.max_seq_length} aa")
    print()

    df, results = run_evaluation(SIM_PARAMS, N_TREES, N_MSAS_PER_TREE, VALIDATION_SEED, models_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "validation_data.csv", index=False)

    print("\nRESULTS")
    print("=" * 60)
    for model_type, metrics in results.items():
        print(f"\n{model_type.replace('_', ' ').title()}:")
        print(f"  Alpha MSE: {metrics['mse_alpha']:.6f}")
        print(f"  Rho MSE:   {metrics['mse_rho']:.6f}")

    print("\nSaving scatter plots...")
    save_scatter_plots(df, results, output_dir / "plots")
    print(f"\nDone — results saved to: {output_dir}/")


if __name__ == "__main__":
    main()