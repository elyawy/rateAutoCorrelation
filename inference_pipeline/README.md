# Entropy-Based Inference Pipeline

Alternative to PAML's codeml for inferring gamma shape (alpha) and correlation (rho) parameters using entropy and parsimony-based features.

## Overview

This pipeline extracts Shannon entropy and parsimony statistics from simulated MSAs and uses them as features for machine learning models to predict alpha and rho parameters.

## Structure

```
inference_pipeline/
├── config.py                       # Configuration
├── features_calculator.py          # Core entropy & parsimony calculation logic
├── 1_extract_entropy_features.py  # Extract features from MSAs
├── 2_train_ml_model.py            # Train RandomForest models
├── 3_compare_methods.py           # Compare ML vs codeml
└── results/                        # Generated: features & models (gitignored)
    ├── features.csv        # Extracted features (8 columns)
    ├── alpha_model.pkl             # Trained alpha model
    ├── rho_model.pkl               # Trained rho model
    ├── ml_evaluation.csv           # ML model metrics
    ├── method_comparison.csv       # ML vs codeml comparison
    └── detailed_predictions.csv    # Per-sample predictions
```

## Prerequisites

1. Run the main pipeline's Step 1 to generate simulated data:
   ```bash
   cd ..
   python 1_generate_simulations.py
   ```

2. Required packages:
   ```bash
   pip install biopython pandas numpy scikit-learn
   ```

## Usage

### Step 1: Extract Features

From the `inference_pipeline/` directory:

```bash
python 1_extract_entropy_features.py
```

This will:
- Read all simulated MSAs from `../simulated_data/`
- Calculate **8 features** for each MSA:
  - **Entropy features** (4):
    - `avg_entropy`: Mean Shannon entropy across sites
    - `entropy_variance`: Variance of entropy values
    - `max_entropy`: Maximum entropy value
    - `lag1_autocorr`: Lag-1 autocorrelation of entropy
  - **Parsimony features** (4):
    - `avg_parsimony_score`: Mean Fitch parsimony score across sites
    - `var_parsimony_score`: Variance of parsimony scores
    - `lag1_parsimony_autocorr`: Lag-1 autocorrelation of parsimony scores
    - `parsimony_entropy_correlation`: Correlation between parsimony and entropy
- Save results to `results/features.csv`

### Step 2: Train ML Models

```bash
python 2_train_ml_model.py
```

This will:
- Load features and ground truth values
- Split trees into train/test sets (configurable in `config.py`)
- Train two RandomForest models (one for alpha, one for rho)
- Save trained models and evaluation metrics
- Display feature importance

### Step 3: Compare Methods

```bash
python 3_compare_methods.py
```

This will:
- Load ML predictions and codeml results
- Compare both methods against ground truth
- Show separate metrics for:
  - **Test trees** (held-out, fair comparison)
  - **Train trees** (ML has advantage)
  - **Overall** (all samples)
- Save comparison results

## Feature Details

### Entropy Features

Shannon entropy is calculated for each column (site) in the alignment:

1. **Filter gaps**: Remove '-' characters from the column
2. **Count frequencies**: Count each amino acid occurrence
3. **Calculate proportions**: p_i = count_i / total_valid_chars
4. **Apply Shannon formula**: H = -Σ(p_i × log₂(p_i))

Properties:
- Uses base-2 logarithm (entropy measured in bits)
- Handles 0 × log(0) as 0 (mathematical convention)
- Empty columns (all gaps) have entropy = 0

### Parsimony Features

Fitch parsimony algorithm is used to calculate minimum substitution counts:

1. **Uses tree topology only** (ignores branch lengths)
2. **Excludes gap-containing sites** (only scores ungapped columns)
3. **Unrooted Fitch algorithm**: Standard phylogenetic parsimony
4. **Returns minimum number of substitutions** per site

Properties:
- Captures phylogenetic signal
- Correlates with evolutionary conservation
- Tree-aware (unlike entropy which is alignment-only)

### Autocorrelation Features

Lag-1 autocorrelation captures spatial patterns:
- Measures correlation between adjacent sites
- Theoretically linked to rho parameter (Markov process correlation)
- Calculated for both entropy and parsimony values

## Configuration

Edit `config.py` to change:
- `N_TRAIN_TREES`: Number of trees for training (rest for testing)
- `SIMULATED_DATA_DIR`: Path to simulated data
- `RESULTS_DIR`: Output directory

## Output Files

- `results/features.csv`: 8 features per MSA
- `results/alpha_model.pkl`: Trained RandomForest for alpha
- `results/rho_model.pkl`: Trained RandomForest for rho
- `results/ml_evaluation.csv`: ML model performance metrics
- `results/tree_split.csv`: Train/test tree assignments
- `results/method_comparison.csv`: ML vs codeml metrics
- `results/detailed_predictions.csv`: Per-sample predictions

## Notes

- The pipeline is fully integrated with the main PAML simulation workflow
- Tree files must exist in `../trees/` directory
- Gap-containing sites are excluded from parsimony calculations
- Feature extraction may take longer than entropy-only due to parsimony computation
- ML models use all 8 features for prediction