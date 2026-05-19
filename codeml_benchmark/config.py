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
TREES_DIR = "trees"
SIMULATED_DATA_DIR = "simulated_data"
CODEML_RUNS_DIR = "codeml_runs"
RESULTS_DIR = "results"

# ==========================================
# SIMULATION PARAMETERS
# ==========================================
NUM_SIMULATIONS_PER_TREE = 400  # How many MSAs per tree
SEQ_LENGTH = 5000              # Length of each sequence

# Parameter ranges for random sampling
ALPHA_RANGE = (0.1, 2.0)       # Min and Max Alpha
RHO_RANGE = (0.01, 0.95)       # Min and Max Rho

# ==========================================
# BASEML PARAMETERS
# ==========================================
CODEML_TEMPLATE = "codeml_template.ctl"
CODEML_EXECUTABLE = "codeml"   # Assumes codeml is in PATH

def find_wag_dat_path():
    """Find wag.dat by parsing codeml's default control file."""
    import tempfile
    import re
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run codeml to generate default control file
        subprocess.run([CODEML_EXECUTABLE], 
                      cwd=tmpdir, 
                      capture_output=True)
        
        ctl_file = pathlib.Path(tmpdir) / "codeml.ctl"
        if ctl_file.exists():
            content = ctl_file.read_text()
            match = re.search(r'aaRatefile\s*=\s*(.+\.dat)', content)
            if match:
                # Replace jones.dat with wag.dat
                return match.group(1).replace('jones.dat', 'wag.dat').strip()
    
    return "wag.dat"  # fallback


WAGDAT_FILE = find_wag_dat_path()