"""
Check which features are sensitive to branch length scale.

Simulates MSAs with identical alpha/rho but different branch length scales,
then compares feature values to identify scale-dependent features.

Run from inside inference_pipeline/:
    python check_scale_sensitivity.py
"""

import numpy as np
import pandas as pd
from ete3 import Tree
from utils.simulation import setup_sim, simulate_msa, SimulationParams
import config
from features_calculator import calculate_msa_entropy_stats

# ==========================================
# CONFIGURATION
# ==========================================
SEED       = 42
N_TAXA     = 150
N_MSAS     = 30          # MSAs per scale (more = stabler averages)
TRUE_ALPHA = 0.2
TRUE_RHO   = 0.7
SCALES     = [0.01, 0.05, 0.1, 0.5, 1.0]   # branch length scales to compare

SIM_PARAMS = SimulationParams(
    min_taxa=N_TAXA, max_taxa=N_TAXA,
    min_seq_length=200, max_seq_length=200,
)


def make_tree(n_taxa, seed, scale):
    """Generate a random tree with a specific branch length scale."""
    np.random.seed(seed)
    tree = Tree()
    tree.populate(n_taxa, random_branches=True)
    for node in tree.traverse():
        if node.dist == 0:
            continue
        node.dist = np.random.exponential(scale=scale)
    return tree


def collect_features(scale, seed):
    """Simulate N_MSAS MSAs at the given scale and return a DataFrame of features."""
    tree = make_tree(N_TAXA, seed, scale)
    simulator = setup_sim(tree, sim_seed=seed)

    rows = []
    for i in range(N_MSAS):
        # Override alpha/rho with fixed values for fair comparison
        simulator.protocol.set_sequence_size(SIM_PARAMS.min_seq_length)
        from msasim.constants import MODEL_CODES
        simulator.set_replacement_model(
            model=MODEL_CODES.WAG,
            gamma_parameters_alpha=TRUE_ALPHA,
            gamma_parameters_categories=8,
            site_rate_correlation=TRUE_RHO
        )
        from msasim.msa import Msa
        msa = simulator()
        sequences = [msa.get_msa_row(j).split("\n")[1] for j in range(msa.get_num_sequences())]

        stats = calculate_msa_entropy_stats(sequences)
        rows.append({k: stats[k] for k in config.FEATURE_COLUMNS if k in stats})

    return pd.DataFrame(rows)


def main():
    print(f"Fixed: alpha={TRUE_ALPHA}, rho={TRUE_RHO}, n_taxa={N_TAXA}, n_msas={N_MSAS}")
    print(f"Scales tested: {SCALES}\n")

    results = {}
    for scale in SCALES:
        print(f"  Simulating scale={scale}...")
        df = collect_features(scale, seed=SEED)
        results[scale] = df.mean()

    summary = pd.DataFrame(results).T   # rows=scales, cols=features
    summary.index.name = "branch_scale"

    # Compute coefficient of variation across scales for each feature
    cv = summary.std() / summary.mean().abs()
    cv = cv.sort_values(ascending=False)

    print("\n=== Feature sensitivity to branch-length scale ===")
    print("(CV = std across scales / mean across scales — higher = more scale-dependent)\n")
    print(cv.to_string())

    print("\n=== Raw feature means per scale ===")
    # Show only the top sensitive features to keep output readable
    top_features = cv.head(10).index.tolist()
    print(summary[top_features].to_string())


if __name__ == "__main__":
    main()
