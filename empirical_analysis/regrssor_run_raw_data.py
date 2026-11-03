import re, sys, pathlib
import numpy as np, pandas as pd, torch
from torch import nn
import time

from common.features_calculator import calculate_cnn_input, calculate_scalar_features
from common.model import RhoRegressor

MSAS = "data/no_sync/msas"
WEIGHTS = "models/rho_regressor_amino_indel_w1000.pt"
OUT = "data/results_real.csv"



def read_fasta(path):
    seqs, cur = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    seqs.append("".join(cur))
    return seqs

# CNN

model = RhoRegressor()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

def nn_rho(g):
    seqs = read_fasta(f"{MSAS}/{g}.fasta")
    x = calculate_cnn_input(seqs, alphabet_size=20).float().unsqueeze(0)
    s = calculate_scalar_features(seqs, alphabet_size=20).float().unsqueeze(0)
    with torch.no_grad():
        return model(x, s).item()

# run the regressor on all the fastas in the folder  MSAS and log the time taken
# no gap/ungapped and stripped/gapped just load the fasta, creak into sequences, calculate the cnn input and scalar features, and run the model to get the rho value
# log time and rho into csv file

rows = []

for msa in sorted(pathlib.Path(MSAS).glob("*.fasta")):
    g = msa.stem
    start = time.time()
    rho = nn_rho(g)
    end = time.time()
    print(f"{g}: {rho:.3f} (time taken: {end - start:.2f} seconds)")
    rows.append({"gene": g, "time": end - start, "rho": rho})

df = pd.DataFrame(rows, columns=["gene", "time", "rho"])
df.to_csv(OUT, index=False)