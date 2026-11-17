import numpy as np, pandas as pd, os

MSA_DIR, SAMPLE, CUTOFF = "data/msas", "data/random_sample_200.csv", 0.5
OUT = {"gapped": "data/real_gapped", "stripped": "data/real_stripped"}
for d in OUT.values():
    os.makedirs(d, exist_ok=True)

def read_fasta(path):
    names, seqs, cur = [], [], []
    for line in open(path):
        if line.startswith(">"):
            if names and cur: seqs.append("".join(cur)); cur = []
            names.append(line[1:].strip())
        else:
            cur.append(line.strip())
    seqs.append("".join(cur))
    return names, np.array([list(s) for s in seqs])

def write_fasta(path, names, arr):
    with open(path, "w") as f:
        for n, row in zip(names, arr):
            f.write(f">{n}\n{''.join(row)}\n")

rows = []
for g in pd.read_csv(SAMPLE).gene:
    names, msa = read_fasta(f"{MSA_DIR}/{g}.fasta")
    gap = msa == "-"
    keep = gap.mean(axis=0) <= CUTOFF
    write_fasta(f"{OUT['gapped']}/{g}.fasta", names, msa)
    write_fasta(f"{OUT['stripped']}/{g}.fasta", names, msa[:, keep])
    rows.append((g, msa.shape[1], int(keep.sum()), gap.mean(), 1 - keep.mean()))

pd.DataFrame(rows, columns=["gene", "n_cols", "n_cols_stripped", "gap_frac", "stripped_frac"]) \
  .to_csv("data/real_strip_stats.csv", index=False)