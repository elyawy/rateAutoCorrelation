# Sim benchmark

Question: how closely does the CNN recover the true rho on held-out simulated alignments?

Inputs: generated simulation files, codeml outputs, and the trained model.

Command: `python experiments/sim_benchmark/3_compare.py`

Output: `results/sim_benchmark/results.csv`.

Expected headline: 139/150 usable codeml runs; CNN RMSE 0.052 and correlation
0.983; codeml RMSE 0.045 and correlation 0.988.

Paper link: simulated benchmark comparison.
