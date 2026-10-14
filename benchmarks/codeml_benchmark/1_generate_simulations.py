"""
Step 1: Simulate test MSAs on 10-leaf clades from OrthoMaM trees.

For each (rho, scale) combination, N_SIMS_PER_COMBO sims are made:
  - random gene from the manifest, random 10-leaf clade from its tree
  - branch lengths multiplied by scale
  - LG, fixed alpha, 8 gamma categories, indels (same setup as the training script)

Output per sim in SIMULATED_DATA_DIR/sim_XXXX_a<alpha>_r<rho>_s<scale>/:
  alignment.phy, tree.newick
Plus truth.csv with all true parameters.
"""

import csv
import json
import pathlib
import random

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

MANIFEST_PATH = pathlib.Path("protein_manifest.json")
RHOS = [0.1, 0.5, 0.9]
SCALES = [1.0]
N_SIMS_PER_COMBO = 50
N_LEAVES = None
MSA_LENGTH = 1000
ALPHA = 0.2
GAMMA_CATEGORIES = 8
INDEL_RATE_MULTIPLIER = 1.0
INSERTION_RATE = 0.007 * INDEL_RATE_MULTIPLIER
DELETION_RATE = 0.035 * INDEL_RATE_MULTIPLIER
SEED = 0


def pick_clade(genes):
    """Return (gene_key, Tree) for a random clade with exactly N_LEAVES leaves."""
    while True:
        gene_key, entry = random.choice(genes)
        tree = Tree(entry["newick"], format=1)
        # if N_LEAVES is None, pick the entire tree; otherwise pick a random clade with exactly N_LEAVES leaves
        if N_LEAVES is None:
            return gene_key, tree.copy()
        clades = [n for n in tree.traverse() if not n.is_leaf() and len(n) == N_LEAVES]
        if clades:
            return gene_key, random.choice(clades).copy()


def scale_tree(tree, scale):
    for node in tree.traverse():
        node.dist *= scale
    tree.dist = 0


def simulate(newick, rho):
    """Return list of (name, sequence), each cropped/padded to MSA_LENGTH like in training."""
    site_rate_model = SiteRateModelSpec(
        gamma_alpha=ALPHA,
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
        root_seq_size=MSA_LENGTH,
        insertion_rate=INSERTION_RATE,
        deletion_rate=DELETION_RATE,
    )
    protocol.set_insertion_length_distributions(ZipfDistribution(p=1.53, truncation=50))
    protocol.set_deletion_length_distributions(ZipfDistribution(p=1.11, truncation=50))
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
    genes = list(json.loads(MANIFEST_PATH.read_text()).items())
    out_root = pathlib.Path(config.SIMULATED_DATA_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    truth = []
    idx = 0
    for scale in SCALES:
        for rho in RHOS:
            for _ in range(N_SIMS_PER_COMBO):
                idx += 1
                gene_key, tree = pick_clade(genes)
                scale_tree(tree, scale)
                newick = tree.write(format=5)

                name = f"sim_{idx:04d}_a{ALPHA}_r{rho}_s{scale}"
                sim_dir = out_root / name
                sim_dir.mkdir(exist_ok=True)
                (sim_dir / "tree.newick").write_text(newick + "\n")
                write_phylip(sim_dir / "alignment.phy", simulate(newick, rho))
                truth.append([name, rho, ALPHA, scale, gene_key])
            print(f"scale={scale} rho={rho} done ({idx} sims so far)")

    with open(out_root / "truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation", "rho", "alpha", "scale", "gene"])
        writer.writerows(truth)


if __name__ == "__main__":
    main()