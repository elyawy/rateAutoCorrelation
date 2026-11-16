# Rate autocorrelation project

This repository contains the code and figures for the rate-autocorrelation analyses described in the paper. The goal is to keep the paper workflow reproducible: small committed result tables are generated from the same scripts that regenerate the main figures from local or Zenodo-backed data.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

The project expects PAML in `bins/paml-4.10.10-linux-x86_64` and data in `data/` (or a custom directory exported as `RHO_DATA_DIR`).

## Pipeline overview

1. `pipeline/1_data_prep/` prepares trees, alignments, and renamed taxa.
2. `pipeline/2_training/` generates the simulation datasets and trains the regressor.
3. `pipeline/3_codeml/` runs codeml on simulations and real data.
4. `pipeline/4_inference/` evaluates the CNN on real MSAs.

## Main commands

- `python pipeline/2_training/generate_sims_amino_indel_v2.py`
- `python pipeline/2_training/train_rho_regressor_amino_indel_v2.py`
- `python pipeline/3_codeml/run_codeml.py --help`
- `python experiments/sim_benchmark/3_compare.py`
- `python experiments/real_data/3_compare_real.py`

## Results index

- `results/sim_benchmark/` — simulated benchmark comparison
- `results/real_data/` — real-data fit and correlation metrics
- `results/stripped_core/` — gap-column stripping analysis
- `results/indel_experiment/` — indel awareness benchmark
- `results/shuffle_control/` — shuffled-column control
- `results/alignment_effect/` — MAFFT / alignment sensitivity analysis

## Notes

- Data files that are not tracked in git live under `data/` or are reproduced by the scripts.
- `config.py` is the single source of truth for file locations, seeds, and codeml settings.
