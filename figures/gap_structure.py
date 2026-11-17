import numpy as np, pandas as pd

MSA_DIR, SAMPLE = "data/msas", "data/random_sample_200.csv"

def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip())
    seqs.append("".join(cur))
    return np.array([list(s) for s in seqs])

df = pd.read_csv(SAMPLE)
rows = []
for g in df.gene:
    gap = read_fasta(f"{MSA_DIR}/{g}.fasta") == "-"
    rows.append((g, gap.mean(), (gap.mean(axis=0) > 0.5).mean()))
df[["gene_", "gap_frac", "stripped_frac"]] = rows

win = df.gap_frac.between(0.40, 0.63)
alpha = df.alpha.between(0.3, 0.8)
print("in 40-63% gap window:", win.sum())
print("also alpha 0.3-0.8:  ", (win & alpha).sum())
print("mean stripped frac:  ", round(df.stripped_frac.mean(), 3))
# print median stripped frac:  ", round(df.stripped_frac.median(), 3))
print("median stripped frac:  ", round(df.stripped_frac.median(), 3))