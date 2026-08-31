"""
Configuration for OrthoMaM empirical analysis.
"""
import pathlib

# ==========================================
# DATA PATHS
# ==========================================
ORTHOMAM_DIR = pathlib.Path("/home/elyalab/Data/orthomam/msas")
RESULTS_DIR = pathlib.Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

# ==========================================
# MODEL PATHS
# ==========================================
# Path to trained neural network model
MODEL_PATH = pathlib.Path("../inference_pipeline/models/neural_net_model.pkl")

# ==========================================
# FEATURE EXTRACTION
# ==========================================
# Import feature columns from inference pipeline
import sys
import importlib.util

# Load inference_pipeline config module explicitly
_inference_config_path = pathlib.Path(__file__).parent.parent / "inference_pipeline" / "config.py"
_spec = importlib.util.spec_from_file_location("inference_config", _inference_config_path)
_inference_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference_config)

# Import the feature columns
FEATURE_COLUMNS = _inference_config.FEATURE_COLUMNS
# ==========================================
# AMINO ACID ALPHABET
# ==========================================
AMINO_ACIDS = _inference_config.AMINO_ACIDS

# ==========================================
# PROCESSING PARAMETERS
# ==========================================
# Number of parallel workers for processing
# Set to None to use all available cores
N_WORKERS = None

RAXML_EXECUTABLE = "/home/pupkolab/Apps/raxml-ng"
RAXML_MODEL = "WAG+G4"
