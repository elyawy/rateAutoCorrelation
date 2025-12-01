import pathlib
import random
import numpy as np
from io import StringIO
from Bio import SeqIO

# --- Hypothetical imports based on your snippet ---
# Ensure msasim is installed in your environment

try:
    from msasim import protocol 
    from msasim import simulator as sim
    from msasim import correlation
except ImportError:
    print("Error: 'msasim' library not found. Please install it to run this script.")
    exit()

# ==========================================
# CONFIGURATION
# ==========================================
NUM_SIMULATIONS = 50       # How many MSAs to generate
SEQ_LENGTH = 5000          # Length of each sequence
OUTPUT_DIR = "simulated_msas"
TREE_FILE = "tree.newick"

# Define ranges for random sampling
ALPHA_RANGE = (0.1, 2.0)   # Min and Max Alpha
RHO_RANGE = (0.01, 0.95)   # Min and Max Rho (avoid 1.0/0.0 strictly)

# ==========================================
# 1. SETUP OUTPUT DIRECTORY
# ==========================================
if not (OUTPUT_DIR := pathlib.Path(OUTPUT_DIR)).exists():
    OUTPUT_DIR.mkdir(parents=True)



# ==========================================
# 2. SIMULATION LOOP
# ==========================================
print(f"Starting generation of {NUM_SIMULATIONS} datasets...")

# Initialize Log File (CSV)
log_path = os.path.join(OUTPUT_DIR, "ground_truth.csv")
with open(log_path, "w") as log:
    log.write("filename,true_alpha,true_rho\n")


simulation_protocol = protocol.SimProtocol(TREE_FILE)
simulation_protocol.set_sequence_size(SEQ_LENGTH)
simulation_protocol.set_insertion_rates(0.0)
simulation_protocol.set_deletion_rates(0.0)

current_seed = 42 
    # B. Configure Simulator
simulation_protocol.set_seed(current_seed)
random.seed(current_seed)
simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)


for i in range(1, NUM_SIMULATIONS + 1):
    # A. Sample Random Parameters
    # Using uniform distribution for Rho
    true_rho = round(random.uniform(*RHO_RANGE), 3)
    
    # Using log-uniform for Alpha (since alpha varies across orders of magnitude)
    # or just uniform if you prefer. Here is simple uniform:
    true_alpha = round(random.uniform(*ALPHA_RANGE), 3)


    # Apply your specific model settings
    simulator.set_replacement_model(
        model=sim.MODEL_CODES.WAG,
        gamma_parameters_alpha=true_alpha,
        gamma_parameters_categories=4,
        site_rate_correlation=true_rho
    )

    simulator.save_root_sequence()
    # C. Run Simulation

    msa = simulator()
    msa_str = msa.get_msa()
    
    # D. Convert to PHYLIP and Save
    # Filename includes parameters for easy spotting, but ID is unique
    filename = f"sim_{i:03d}_a{true_alpha}_r{true_rho}.phy"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    handle = StringIO(msa_str)
    
    # PAML requires strict sequential or interleaved. 
    # 'phylip-sequential' is generally safest for PAML.
    SeqIO.convert(handle, "fasta", filepath, "phylip-sequential")
    
    # E. Log Ground Truth
    with open(log_path, "a") as log:
        log.write(f"{filename},{true_alpha},{true_rho}\n")
        
    if i % 10 == 0:
        print(f"Generated {i}/{NUM_SIMULATIONS} simulations...")


print("------------------------------------------------")
print(f"Done! Files are in '{OUTPUT_DIR}/'")
print(f"Ground truth log saved to '{log_path}'")