"""
Step 1: Simulate test MSAs for the rho regressor vs codeml comparison.

For each (rho, scale) combination, N_SIMS_PER_COMBO sims are made:
  - random gene from the HELD-OUT genes (same split as the training script, so no tree was seen in training)
  - the gene's own tree, alpha and indel model (same setup as train_rho_regressor_amino_indel_v2.py)
  - LG, 8 gamma categories, indel-aware rate correlation rho, alignment cropped/padded to MSA_LENGTH
  - branch lengths multiplied by scale

Output per sim in SIMULATED_DATA_DIR/sim_XXXX_a<alpha>_r<rho>_s<scale>/:
  alignment.phy, tree.newick
Plus truth.csv with all true parameters.
"""

import csv
import pathlib
import random
import re

from ete3 import Tree
from msasim import (
    ALPHABET_CODES,
    MODEL_CODES,
    ReplacementModelSpec,
    SimProtocol,
    Simulator,
    SiteRateModelSpec,
)
from msasim.constants import SITE_RATE_MODELS
from msasim.distributions import ZipfDistribution

import config

# gene split: keep in sync with train_rho_regressor_amino_indel_v2.py / generate_sims_amino_indel_v2.py
ESTIMATES_CSV = pathlib.Path("protein_estimates_train500.csv")  # gene, newick, alpha
INDEL_CSV = pathlib.Path("data/no_sync/indel_estimates_2.csv")  # per-gene indel model
SPLIT_SEED = 0
TEST_GENE_FRACTION = 0.1

RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCALES = [1.0]
N_SIMS_PER_COMBO = 30
N_LEAVES = None
MSA_LENGTH = 1000
GAMMA_CATEGORIES = 8
MAX_INDEL_LENGTH = 150  # truncation of the indel length distributions, as in training
SEED = 0


def load_genes():
    """[(gene, {newick, alpha, ins_rate, del_rate, len_ins, len_del, root})] for genes with a tree and an indel model.

    Same filtering and order as the training script, so the shuffle below gives the same split.
    """
    indel = {row["name"].removesuffix(".fasta"): row for row in csv.DictReader(open(INDEL_CSV))}
    genes = []
    for row in csv.DictReader(open(ESTIMATES_CSV)):
        ind = indel.get(row["gene"])
        if ind is None:
            continue
        alpha_col = next((c for c in ("alpha", "gamma_alpha") if c in row), None)
        if alpha_col is None:
            raise KeyError(f"no alpha column in {ESTIMATES_CSV} (columns: {list(row)})")
        genes.append((row["gene"], {
            "newick": re.sub(r"\d+\.\d+:", ":", row["newick"]),  # drop support values
            "alpha": float(row[alpha_col]),
            **{k: float(ind[k]) for k in ("ins_rate", "del_rate", "len_ins", "len_del", "root")},
        }))
    return genes


def held_out_genes():
    genes = load_genes()
    random.Random(SPLIT_SEED).shuffle(genes)
    return genes[:int(len(genes) * TEST_GENE_FRACTION)]


def pick_clade(genes):
    """Return (gene_key, entry, Tree) for a random gene and a random clade with exactly N_LEAVES leaves."""
    while True:
        gene_key, entry = random.choice(genes)
        tree = Tree(entry["newick"], format=1)
        # if N_LEAVES is None, pick the entire tree; otherwise pick a random clade with exactly N_LEAVES leaves
        if N_LEAVES is None:
            return gene_key, entry, tree.copy()
        clades = [n for n in tree.traverse() if not n.is_leaf() and len(n) == N_LEAVES]
        if clades:
            return gene_key, entry, random.choice(clades).copy()


def scale_tree(tree, scale):
    for node in tree.traverse():
        node.dist *= scale
    tree.dist = 0


def simulate(newick, rho, entry):
    """Return list of (name, sequence), each cropped/padded to MSA_LENGTH like in training."""
    site_rate_model = SiteRateModelSpec(
        gamma_alpha=entry["alpha"],
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
    replacement_model = ReplacementModelSpec(
        model=MODEL_CODES.LG,
        alphabet=ALPHABET_CODES.PROTEIN,
        site_rate_model=site_rate_model,
    )
    protocol = SimProtocol(
        tree=newick,
        # the gene's own root length, but never below the window, so the cropped window has no padding
        root_seq_size=max(int(round(entry["root"])), MSA_LENGTH),
        insertion_rate=entry["ins_rate"],
        deletion_rate=entry["del_rate"],
    )
    protocol.set_insertion_length_distributions(ZipfDistribution(p=entry["len_ins"], truncation=MAX_INDEL_LENGTH))
    protocol.set_deletion_length_distributions(ZipfDistribution(p=entry["len_del"], truncation=MAX_INDEL_LENGTH))
    msa = Simulator(protocol, replacement_model=replacement_model)()

    rows = []
    for i in range(msa.get_num_sequences()):
        header, seq = msa.get_msa_row(i).split("\n")[:2]
        seq = seq[:MSA_LENGTH].ljust(MSA_LENGTH, "-")
        rows.append((header.lstrip(">").strip(), seq))
    return rows


def write_phylip(path, rows):
    with open(path, "w") as f:
        f.write(f" {len(rows)} {MSA_LENGTH}\n")
        for name, seq in rows:
            f.write(f"{name}  {seq}\n")


def main():
    random.seed(SEED)
    genes = held_out_genes()
    print(f"{len(genes)} held-out genes")
    out_root = pathlib.Path(config.SIMULATED_DATA_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    truth = []
    idx = 0
    for scale in SCALES:
        for rho in RHOS:
            for _ in range(N_SIMS_PER_COMBO):
                idx += 1
                gene_key, entry, tree = pick_clade(genes)
                scale_tree(tree, scale)
                newick = tree.write(format=5)

                name = f"sim_{idx:04d}_a{entry['alpha']:.4f}_r{rho}_s{scale}"
                sim_dir = out_root / name
                sim_dir.mkdir(exist_ok=True)
                (sim_dir / "tree.newick").write_text(newick + "\n")
                write_phylip(sim_dir / "alignment.phy", simulate(newick, rho, entry))
                truth.append([name, rho, entry["alpha"], scale, gene_key])
            print(f"scale={scale} rho={rho} done ({idx} sims so far)")

    with open(out_root / "truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation", "rho", "alpha", "scale", "gene"])
        writer.writerows(truth)


if __name__ == "__main__":
    main()