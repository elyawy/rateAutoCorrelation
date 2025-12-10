"""
Configuration for the entropy-based inference pipeline.
"""
import pathlib

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
N_TRAIN_TREES = 100  # Number of trees to use for training (rest used for testing)

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