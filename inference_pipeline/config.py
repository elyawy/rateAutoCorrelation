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

# ==========================================
# MACHINE LEARNING PARAMETERS
# ==========================================
N_TRAIN_TREES = 71  # Number of trees to use for training (rest used for testing)

# ==========================================
# AMINO ACID ALPHABET
# ==========================================
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')