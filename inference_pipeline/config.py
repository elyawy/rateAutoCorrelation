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
TRAINING_METHOD = 'lightgbm'    # options: 'lightgbm', 'neural_net', 'random_forest'
# TRAINING_METHOD = 'neural_net'
# TRAINING_METHOD = 'random_forest'

N_TRAIN_TREES = 1000  # Which trees to use for training (first N trees)
N_SIMS_PER_TREE = 100  # How many simulations per tree to use for training (50, 100, or 200)

# ==========================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ==========================================
N_OPTUNA_TRIALS = 30  # Number of trials for hyperparameter search

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
    'bimodality_coefficient',
    'gamma_shape_entropy',
    
    # Histogram bins (10 bins)
    'entropy_bin_0', 'entropy_bin_1', 'entropy_bin_2', 'entropy_bin_3', 'entropy_bin_4',
    'entropy_bin_5', 'entropy_bin_6', 'entropy_bin_7', 'entropy_bin_8', 'entropy_bin_9',
    
    # Multi-lag autocorrelations (lags 2-10, lag1 already exists above)
    'lag2_autocorr', 'lag3_autocorr', 'lag4_autocorr', 'lag5_autocorr',
    'lag6_autocorr', 'lag7_autocorr', 'lag8_autocorr', 'lag9_autocorr', 'lag10_autocorr',
 
    # Run-length features (high- and low-entropy runs separately)
    'high_run_mean', 'high_run_var', 'high_run_max', 'high_run_count',
    'low_run_mean',  'low_run_var',  'low_run_max',  'low_run_count',

    # Indel-related features (from msastats)
    'avg_gap_size', 'msa_len', 'msa_max_len', 'msa_min_len', 'tot_num_gaps',
    'num_gaps_len_one', 'num_gaps_len_two', 'num_gaps_len_three', 'num_gaps_len_at_least_four'

]

# ==========================================
# AMINO ACID ALPHABET
# ==========================================
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Parameter ranges for random sampling
ALPHA_RANGE = (0.01, 5.0)       # Min and Max Alpha
RHO_RANGE = (0.01, 0.99)       # Min and Max Rho
SCALE_MIN = 0.01
SCALE_MAX = 0.5