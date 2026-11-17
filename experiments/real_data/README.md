# Real data benchmark

Question: how well does the CNN track codeml rho across the real 15k-gene dataset?

Inputs: OrthoMaM-derived alignments and real-data codeml runs.

Command: `python experiments/real_data/3_compare_real.py`

Output: `results/real_data/results_real.csv`.

Expected headline: on 15,182 genes with codeml rho, CNN versus codeml has
correlation 0.852, RMSE 0.104, and bias -0.005.

Paper link: real-data regression panel.
