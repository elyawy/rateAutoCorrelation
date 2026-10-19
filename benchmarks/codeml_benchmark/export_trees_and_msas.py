"""
Write each gene's tree and cleaned MSA to separate folders with a shared base name:
    trees/{gene}.nwk   and   msas/{gene}.fasta
Reads protein_estimates.csv (trees) and protein_msas.parquet (the cleaned MSAs actually used).
"""

import csv
import re
from pathlib import Path

import pyarrow.parquet as pq

ESTIMATES_CSV = Path("protein_estimates.csv")
MSA_PARQUET = Path("protein_msas.parquet")
TREE_DIR = Path("trees")
MSA_DIR = Path("msas")


def main():
    TREE_DIR.mkdir(exist_ok=True)
    MSA_DIR.mkdir(exist_ok=True)

    trees = {row["gene"]: row["newick"] for row in csv.DictReader(open(ESTIMATES_CSV))}

    table = pq.read_table(MSA_PARQUET).to_pydict()
    fastas = {}
    for gene, taxon, seq in zip(table["gene"], table["taxon"], table["sequence"]):
        fastas.setdefault(gene, []).append(f">{taxon}\n{seq}\n")

    genes = sorted(set(trees) & set(fastas))
    for gene in genes:
        # remove bootstrap support from the tree, which is not needed regex "\d+.\d+:" replace with ":".
        trees[gene] = re.sub(r"\d+\.\d+:", ":", trees[gene])

        (TREE_DIR / f"{gene}.nwk").write_text(trees[gene] + "\n")
        (MSA_DIR / f"{gene}.fasta").write_text("".join(fastas[gene]))

    print(f"wrote {len(genes)} tree/MSA pairs to {TREE_DIR}/ and {MSA_DIR}/")
    only_trees, only_msas = set(trees) - set(fastas), set(fastas) - set(trees)
    if only_trees or only_msas:
        print(f"WARNING: {len(only_trees)} genes only in the csv, {len(only_msas)} only in the parquet (skipped)")


if __name__ == "__main__":
    main()
