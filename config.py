"""
Central configuration for the PAML simulation pipeline.
All scripts import from this file to ensure reproducibility.
"""

# ==========================================
# REPRODUCIBILITY
# ==========================================
MASTER_SEED = 42  # Change this to get different random datasets

# ==========================================
# DIRECTORIES
# ==========================================
TREES_DIR = "trees"
SIMULATED_DATA_DIR = "simulated_data"
BASEML_RUNS_DIR = "baseml_runs"
RESULTS_DIR = "results"

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
NUM_SIMULATIONS_PER_TREE = 50  # How many MSAs per tree
SEQ_LENGTH = 5000              # Length of each sequence

# Parameter ranges for random sampling
ALPHA_RANGE = (0.1, 2.0)       # Min and Max Alpha
RHO_RANGE = (0.01, 0.95)       # Min and Max Rho

# ==========================================
# BASEML PARAMETERS
# ==========================================
BASEML_TEMPLATE = "baseml_template.ctl"
BASEML_EXECUTABLE = "baseml"   # Assumes baseml is in PATH
