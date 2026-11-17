"""Project-wide configuration for the rate-autocorrelation pipeline.

All scripts should reach data and outputs through this module instead of using
cwd-relative paths or hard-coded file names.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RHO_DATA_DIR", REPO_ROOT / "data")).resolve()

SPLIT_SEED = 0
TEST_GENE_FRACTION = 0.1

# PAML / RAxML locations used by the paper pipeline.
PAML_ROOT = REPO_ROOT / "bins" / "paml-4.10.10-linux-x86_64"
RAXML_ROOT = REPO_ROOT / "bins"
CODEML_EXECUTABLE = str((PAML_ROOT / "bin" / "codeml").resolve())
MODELDAT_FILE = str((PAML_ROOT / "dat" / "lg.dat").resolve())
CODEML_TEMPLATE = str((REPO_ROOT / "pipeline" / "3_codeml" / "codeml_template.ctl").resolve())

CODEML_TIMEOUT = 120
CODEML_TIMEOUT_REAL = 60
SIMULATION_SEED = 0
TORCH_SEED = 0

SIMULATED_DATA_DIR = DATA_DIR / "simulated_data"
CODEML_RUNS_DIR = DATA_DIR / "codeml_runs"
RESULTS_DIR = str((REPO_ROOT / "results").resolve())
REAL_RESULTS_DIR = str((Path(RESULTS_DIR) / "real_data").resolve())
PLOTS_DIR = str((Path(RESULTS_DIR) / "plots").resolve())
REAL_CNN_RESULTS = str((Path(REAL_RESULTS_DIR) / "results_real_new.csv").resolve())
REAL_CODEML_RESULTS = str((Path(REAL_RESULTS_DIR) / "codeml_rho_raw.csv").resolve())
RHO_DISTRIBUTION_PLOT = str((Path(PLOTS_DIR) / "rho_distribution_histogram.png").resolve())
RHO_HEXBIN_PLOT = str((Path(PLOTS_DIR) / "rho_cnn_vs_codeml_hexbin.png").resolve())
REAL_SUMMARY_CSV = str((Path(REAL_RESULTS_DIR) / "summary.csv").resolve())
MODEL_DIR = str((REPO_ROOT / "models").resolve())
TRAINING_SIMULATION_DIR = DATA_DIR / "training_sims"
MODEL_PATH = str((REPO_ROOT / "models" / "rho_regressor_amino_indel_trees500_gap_w1000.pt").resolve())

TRAIN_ESTIMATES_CSV = (REPO_ROOT / "pipeline" / "2_training" / "inputs" / "protein_estimates_train500.csv").resolve()
TEST_ESTIMATES_CSV = (REPO_ROOT / "pipeline" / "2_training" / "inputs" / "protein_estimates_test200.csv").resolve()
INDEL_PARAMS_CSV = (DATA_DIR  / "indel_estimates_orthomam.csv").resolve()
REAL_MSA_DIR = (DATA_DIR / "msas").resolve()
REAL_TREE_DIR = (DATA_DIR / "trees").resolve()

REAL_GAPPED_MSA_DIR = (DATA_DIR  / "real_gapped").resolve()
REAL_STRIPPED_MSA_DIR = (DATA_DIR / "real_stripped").resolve()
REAL_ESTIMATES_CSV = (DATA_DIR / "protein_estimates.csv").resolve()
REAL_MSAS_PARQUET = (DATA_DIR / "protein_msas.parquet").resolve()
REAL_CODEML_RUNS_DIR = (DATA_DIR  / "codeml_runs_real").resolve()
REAL_GAPPED_CODEML_RUNS_DIR = (DATA_DIR  / "codeml_runs_real_gapped").resolve()
RAW_MSA_DIR = (DATA_DIR / "raw" / "msas").resolve()
RAW_TREE_DIR = (DATA_DIR / "raw" / "trees").resolve()

# Paper configuration values described in the refactor guide.
ALPHA_RANGE = (0.1, 3.0)
RHO_RANGE = (0.01, 0.99)
GAMMA_CATEGORIES = 8
TEST_SIZES = [100, 250, 500, 1000]
WINDOW_SIZES = [1000]
N_TRAIN = 20000
N_TEST = 1000
FILE_TAG = "amino_indel_trees500_gap"

# Zenodo download metadata.
RECORD_ID = ""  # Placeholder until the publication record is live.
MANIFEST_PATH = DATA_DIR / "MANIFEST.csv"


def as_path(value):
    return Path(value).expanduser() if value else None
