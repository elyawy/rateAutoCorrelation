# PAML Simulation Pipeline

Reproducible workflow for testing baseml parameter inference using simulated data.

## Project Structure

```
project/
├── trees/                      # Input: Newick tree files
├── simulated_data/             # Generated: Simulated MSAs (gitignored)
├── baseml_runs/                # Generated: baseml outputs (gitignored)
├── results/                    # Generated: Analysis results (gitignored)
├── config.py                   # Central configuration
├── baseml_template.ctl         # Template for baseml control files
├── clean.py                    # Cleanup script
├── 1_generate_simulations.py  # Step 1: Generate data
├── 2_run_baseml.py            # Step 2: Run baseml
├── 3_extract_parameters.py    # Step 3: Extract results
└── 4_calculate_mse.py         # Step 4: Calculate MSE
```

## Setup

1. Install required Python packages:
   ```bash
   pip install biopython pandas numpy msasim
   ```

2. Ensure `baseml` is installed and in your PATH:
   ```bash
   which baseml  # Should return a path
   ```

3. Add your tree files to `trees/` directory (`.newick` extension)

## Usage

### Full Pipeline

Run all steps in order:

```bash
python 1_generate_simulations.py  # Generate simulated MSAs
python 2_run_baseml.py            # Run baseml on all simulations
python 3_extract_parameters.py    # Extract inferred parameters
python 4_calculate_mse.py         # Calculate MSE
```

### Individual Steps

Each script can be run independently (after prerequisites are met):

- **Step 1**: Generates simulated MSAs with known alpha/rho values
- **Step 2**: Runs baseml to infer alpha/rho from simulations
- **Step 3**: Parses baseml output files and extracts inferred parameters
- **Step 4**: Compares ground truth with inferred values, calculates MSE

### Clean Up

To remove all generated data and start fresh:

```bash
python clean.py
```

## Configuration

Edit `config.py` to change:

- `MASTER_SEED`: Random seed for reproducibility
- `NUM_SIMULATIONS_PER_TREE`: Number of MSAs per tree
- `SEQ_LENGTH`: Length of simulated sequences
- `ALPHA_RANGE`, `RHO_RANGE`: Parameter ranges for simulation
- Directory names and paths

## Reproducibility

The pipeline is fully reproducible:
- Same `MASTER_SEED` → same simulated data
- Each tree gets a deterministic seed derived from `MASTER_SEED + hash(tree_name)`
- Each simulation gets a seed derived from the tree seed + simulation number

## Output Files

- `simulated_data/{tree}/ground_truth.csv`: True alpha/rho values
- `baseml_runs/{tree}/{sim}/mlb`: baseml output files
- `results/inferred_parameters.csv`: Extracted inferred parameters
- `results/mse_analysis.csv`: MSE and other metrics per tree and overall
- `results/merged_data.csv`: Combined ground truth and inferred values

## Notes

- baseml runs with model=7 (REV), estimating alpha and rho (kappa is fixed)
- Failed baseml runs are skipped during extraction
- Check console output for warnings about missing files or parse errors
