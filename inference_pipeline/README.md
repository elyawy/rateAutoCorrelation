# Entropy-Based Inference Pipeline

Alternative to PAML's codeml for inferring gamma shape (alpha) and correlation (rho) parameters using entropy-based features.

## Overview

This pipeline extracts Shannon entropy statistics from simulated MSAs and will use them as features for machine learning models to predict alpha and rho parameters.

## Structure

```
inference_pipeline/
├── config.py                       # Configuration
├── entropy_calculator.py           # Core entropy calculation logic
├── 1_extract_entropy_features.py  # Extract features from MSAs
└── results/                        # Generated: entropy features (gitignored)
    └── entropy_features.csv
```

## Prerequisites

1. Run the main pipeline's Step 1 to generate simulated data:
   ```bash
   cd ..
   python 1_generate_simulations.py
   ```

2. Required packages (should already be installed):
   ```bash
   pip install biopython pandas numpy
   ```

## Usage

### Extract Entropy Features

From the `inference_pipeline/` directory:

```bash
python 1_extract_entropy_features.py
```

This will:
- Read all simulated MSAs from `../simulated_data/`
- Calculate entropy statistics for each MSA:
  - **Average Entropy**: Mean entropy across all columns
  - **Entropy Variance**: Sample variance of column entropies
  - **Min Entropy**: Minimum entropy across columns
  - **Max Entropy**: Maximum entropy across columns
- Save results to `results/entropy_features.csv`

### Output Format

`results/entropy_features.csv` contains:
- `tree`: Tree name
- `simulation`: Simulation identifier (e.g., sim_001_a0.5_r0.3)
- `avg_entropy`: Average entropy per site (bits)
- `entropy_variance`: Variance of entropy values
- `min_entropy`: Minimum column entropy
- `max_entropy`: Maximum column entropy

## Entropy Calculation Details

Shannon entropy is calculated for each column (site) in the alignment:

1. **Filter gaps**: Remove '-' characters from the column
2. **Count frequencies**: Count each amino acid occurrence
3. **Calculate proportions**: p_i = count_i / total_valid_chars
4. **Apply Shannon formula**: H = -Σ(p_i × log₂(p_i))

Key properties:
- Uses base-2 logarithm (entropy measured in bits)
- Handles 0 × log(0) as 0 (mathematical convention)
- Empty columns (all gaps) have entropy = 0
- Uses standard 20 amino acid alphabet (ACDEFGHIKLMNPQRSTVWY)

## Next Steps

Future additions will include:
- Script to merge entropy features with ground truth values
- Model training scripts
- Parameter inference using trained models
- Comparison with codeml results
