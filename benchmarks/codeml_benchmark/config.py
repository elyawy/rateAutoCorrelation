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
SIMULATED_DATA_DIR = "simulated_data"
CODEML_RUNS_DIR = "codeml_runs"
RESULTS_DIR = "results"

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
NUM_SIMULATIONS = 400         # Total number of simulations
SEQ_LENGTH = 500              # Length of each sequence

# Parameter ranges for random sampling
ALPHA_RANGE = (0.01, 2.0)           # Min and Max Alpha
RHO_RANGE = (0.01, 0.99)           # Min and Max Rho
TREE_SCALE_RANGE = (0.005, 0.1)    # Min and Max tree scale (sampled in log10 space)
MIN_TAXA = 20                       # Min number of taxa per tree
MAX_TAXA = 200                      # Max number of taxa per tree

# ==========================================
# CODEML PARAMETERS
# ==========================================
CODEML_TEMPLATE = "benchmarks/codeml_benchmark/codeml_template.ctl"
CODEML_EXECUTABLE = "bins/paml-4.10.10-linux-x86_64/bin/codeml"   # Assumes codeml is in PATH
MODELDAT_FILE = "bins/paml-4.10.10-linux-x86_64/dat/lg.dat"