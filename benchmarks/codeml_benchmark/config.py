"""
Central configuration for the PAML simulation pipeline.
All scripts import from this file to ensure reproducibility.
"""
import subprocess
import pathlib

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
CODEML_TEMPLATE = "codeml_template.ctl"
CODEML_EXECUTABLE = "codeml"   # Assumes codeml is in PATH

def find_wag_dat_path():
    """Find wag.dat by parsing codeml's default control file."""
    import tempfile
    import re

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([CODEML_EXECUTABLE],
                      cwd=tmpdir,
                      capture_output=True)

        ctl_file = pathlib.Path(tmpdir) / "codeml.ctl"
        if ctl_file.exists():
            content = ctl_file.read_text()
            match = re.search(r'aaRatefile\s*=\s*(.+\.dat)', content)
            if match:
                return match.group(1).replace('jones.dat', 'wag.dat').strip()

    return "wag.dat"  # fallback


WAGDAT_FILE = find_wag_dat_path()