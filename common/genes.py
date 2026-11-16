"""Shared gene loading and held-out splitting helpers."""

from __future__ import annotations

import csv
import random
from pathlib import Path

SPLIT_SEED = 0
TEST_GENE_FRACTION = 0.1


def load_genes(estimates_csv, indel_csv):
    """Load observable genes shared by the estimates and indel parameter tables.

    The estimates CSV is expected to contain a ``gene`` column and a per-gene
    alpha-like column named either ``alpha`` or ``gamma_alpha``.
    """
    estimates_path = Path(estimates_csv)
    indel_path = Path(indel_csv)

    indel_by_gene = {}
    with indel_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("name") or row.get("gene") or row.get("id") or "").strip()
            if not name:
                continue
            indel_by_gene[name.removesuffix(".fasta")] = row

    genes = []
    with estimates_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gene_name = (row.get("gene") or row.get("name") or row.get("id") or "").strip()
            if not gene_name:
                continue
            gene_key = gene_name.removesuffix(".fasta")
            if gene_key not in indel_by_gene:
                continue

            alpha_value = row.get("alpha", row.get("gamma_alpha"))
            if alpha_value is None:
                raise KeyError(f"alpha or gamma_alpha not found in {estimates_path}")

            gene_entry = dict(row)
            gene_entry["gene"] = gene_key
            gene_entry["alpha"] = float(alpha_value)
            genes.append(gene_entry)

    return genes


def held_out_genes(genes, split_seed=SPLIT_SEED, test_gene_fraction=TEST_GENE_FRACTION):
    """Return train/test partitions with a deterministic held-out split."""
    gene_list = list(genes)
    if not gene_list:
        return [], []

    rng = random.Random(split_seed)
    rng.shuffle(gene_list)

    n_test = int(round(len(gene_list) * test_gene_fraction))
    n_test = max(1, min(n_test, len(gene_list) - 1)) if len(gene_list) > 1 else 0
    test_genes = gene_list[:n_test]
    train_genes = gene_list[n_test:]
    return train_genes, test_genes
