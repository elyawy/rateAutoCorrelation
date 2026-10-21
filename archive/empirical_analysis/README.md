# OrthoMaM Empirical Analysis

Apply trained neural network model to infer alpha and rho parameters from the OrthoMaM dataset.

## Overview

This pipeline processes the OrthoMaM multiple sequence alignments to infer site rate heterogeneity (alpha) and site-to-site rate correlation (rho) parameters using a pre-trained neural network model.

## Directory Structure

```
empirical_analysis/
├── config.py                      # Configuration and paths
├── 1_process_orthomam.py         # Main processing script
├── 2_plot_distributions.py       # Visualization script
└── results/
    ├── orthomam_predictions.csv  # Inferred parameters with metadata
    └── plots/
        ├── alpha_distribution.png
        └── rho_distribution.png
```

## Prerequisites

1. Trained neural network model at: `../inference_pipeline/models/neural_net_model.pkl`
2. OrthoMaM dataset at: `/home/elyalab/Data/orthomam/msas/*.fasta.zip`

## Usage

### Step 1: Process OrthoMaM Dataset

Extract features from all FASTA files and infer alpha/rho parameters:

```bash
python 1_process_orthomam.py
```

This will:
- Read all `.fasta.zip` files from the OrthoMaM directory
- Extract FASTA content from each archive
- Calculate entropy-based features
- Predict alpha and rho using the trained model
- Save results to `results/orthomam_predictions.csv`

**Output CSV columns:**
- `filename`: Name of the FASTA file
- `n_sequences`: Number of sequences in the alignment
- `alignment_length`: Length of the alignment
- `n_variable_sites`: Number of variable (polymorphic) sites
- `pred_alpha`: Predicted alpha parameter (site rate heterogeneity)
- `pred_rho`: Predicted rho parameter (site-to-site correlation)

### Step 2: Plot Distributions

Generate histogram plots:

```bash
python 2_plot_distributions.py
```

This creates:
- `results/plots/alpha_distribution.png` - Distribution of inferred alpha values
- `results/plots/rho_distribution.png` - Distribution of inferred rho values

Each plot includes summary statistics (mean, median, std, min, max).

## Gap Handling

The pipeline keeps gap characters as-is in the alignments. The feature extraction code (from `inference_pipeline/features_calculator.py`) already handles gaps appropriately by:
- Excluding gaps when computing column entropy
- Only counting valid amino acids for statistical calculations

This preserves alignment structure while avoiding signal loss.

## Notes

- Processing is currently sequential (one file at a time) to avoid memory issues with loading the model multiple times
- Progress is printed every 10 files
- Files that fail to process are logged and skipped
