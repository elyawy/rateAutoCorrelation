# Indel experiment

Question: how much does codeml rho shift when indel-aware versus indel-blind models are used?

Inputs: paired simulations and codeml outputs.

Command: `python experiments/indel_experiment/1_simulate.py`

```
python experiments/indel_experiment/1_simulate.py --n 100 --indel-scale 5 --rate-model indel_aware --n-workers 8

python experiments/indel_experiment/1_simulate.py --n 100 --indel-scale 5 --rate-model simple      --n-workers 8
```

Output: bias estimates in `results/indel_experiment/`.

Expected headline: indel-aware codeml has an upward rho bias of about +0.017;
stripping gap columns lowers rho by about -0.027 for aware simulations and
-0.050 for simple gamma-rate simulations.

Paper link: indel-aware bias analysis.
