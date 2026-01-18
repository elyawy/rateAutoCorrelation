"""
Configuration for the entropy-based inference pipeline.
"""
import pathlib


MASTER_SEED = 42

# ==========================================
# INFERENCE PIPELINE DIRECTORIES
# ==========================================
SIMULATED_DATA_DIR = pathlib.Path("training_data")  # Self-generated training data
RESULTS_DIR = pathlib.Path("results")
FEATURES_DIR = pathlib.Path("features")

# ==========================================
# MACHINE LEARNING PARAMETERS
# ==========================================
TRAINING_METHOD = 'neural_net'  # or 'random_forest'
N_TRAIN_TREES = 100  # Which trees to use for training (first N trees)
N_SIMS_PER_TREE = 50  # How many simulations per tree to use for training (50, 100, or 200)
USE_COMPLETE_TEST_TREES_ONLY = True  # Filter test set to only trees with complete codeml results

# ==========================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ==========================================
N_OPTUNA_TRIALS = 50  # Number of trials for hyperparameter search

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
    # 'inverse_entropy_variance',
]


# ==========================================
# AMINO ACID ALPHABET
# ==========================================
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Parameter ranges for random sampling
ALPHA_RANGE = (0.1, 2.0)       # Min and Max Alpha
RHO_RANGE = (0.01, 0.95)       # Min and Max Rho