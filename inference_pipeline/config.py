"""
Configuration for the entropy-based inference pipeline.
"""
import pathlib
import sys

MASTER_SEED = 42

# ==========================================
# PARENT DIRECTORIES (from main pipeline)
# ==========================================
# Point directly to the simulated data directory
SIMULATED_DATA_DIR = pathlib.Path("..") / "simulated_data"

# ==========================================
# INFERENCE PIPELINE DIRECTORIES
# ==========================================
RESULTS_DIR = pathlib.Path("results")
TREES_DIR = pathlib.Path("..") / "trees"

# ==========================================
# MACHINE LEARNING PARAMETERS
# ==========================================
N_TRAIN_TREES = 100  # Which trees to use for training (first N trees)
N_SIMS_PER_TREE = 50  # How many simulations per tree to use for training (50, 100, or 200)
USE_COMPLETE_TEST_TREES_ONLY = True  # Filter test set to only trees with complete codeml results

# ==========================================
# FEATURE CONFIGURATION
# ==========================================
# Central definition of all features used in ML models
# Add/remove features here to change the feature set globally
FEATURE_COLUMNS = [
    # Entropy-based features
    'avg_entropy',
    'entropy_variance',
    'lag1_autocorr',
    'entropy_skewness',
    'entropy_kurtosis',
    # Alignment features
    'bimodality_coefficient',
]

# ==========================================
# AMINO ACID ALPHABET
# ==========================================
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')