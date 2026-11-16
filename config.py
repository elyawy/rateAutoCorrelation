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
CODEML_TEMPLATE = str((REPO_ROOT / "benchmarks" / "codeml_benchmark" / "codeml_template.ctl").resolve())

CODEML_TIMEOUT = 120
SIMULATION_SEED = 0
TORCH_SEED = 0

SIMULATED_DATA_DIR = str((REPO_ROOT / "simulated_data").resolve())
CODEML_RUNS_DIR = str((REPO_ROOT / "codeml_runs").resolve())
RESULTS_DIR = str((REPO_ROOT / "results").resolve())
MODEL_DIR = str((REPO_ROOT / "models").resolve())

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
