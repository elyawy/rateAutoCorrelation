"""
Check whether the OrthoMaM (nucleotide) trees give realistic protein divergence.

For a random sample of genes, compare the mean pairwise divergence of the real amino acid
alignment with LG simulations on the manifest tree, at several branch length scales.
Pairs are matched by gene, so the plot shows y = x if the tree scale is right.
"""

import json
import random
import re
import sys
import zipfile
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "common"))
from features_calculator import calculate_scalar_features

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

AA_DIR = Path("/home/pupkolab/temp/orthomam_AA")  # amino acid *.fasta.zip files
MANIFEST_PATH = Path("/home/pupkolab/Dev/SailfishBenchmark/simulation_manifest.json")
N_GENES = 300
WINDOW = 250
SCALES = [0.5, 1.0, 2.0]
RHO = 0.9  # divergence barely depends on rho, any value works
GAMMA_CATEGORIES = 8
INSERTION_RATE = 0.007
DELETION_RATE = 0.035
SEED = 0


def gene_id(name: str) -> str:
    """'10000_NT_AL_AA.fasta.zip' or '10000_AKT3' -> '10000'."""
    return name.split("_")[0]


def read_zip_sequences(zip_path: Path) -> list:
    with zipfile.ZipFile(zip_path) as zf:
        fasta_name = next(f for f in zf.namelist() if f.endswith(".fasta"))
        content = zf.read(fasta_name).decode("utf-8")
    return [str(rec.seq) for rec in SeqIO.parse(StringIO(content), "fasta")]


def scale_newick(newick: str, scale: float) -> str:
    """Multiply every branch length in a newick string by `scale`."""
    return re.sub(
        r":(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)",
        lambda m: f":{float(m.group(1)) * scale:.8g}",
        newick,
    )


def log_divergence(sequences: list) -> float:
    return float(calculate_scalar_features(sequences, alphabet_size=20)[0])


def simulate_protein(gene_entry: dict, scale: float, window: int) -> list:
    protocol = SimProtocol(
        tree=scale_newick(gene_entry["newick"], scale),
        root_seq_size=window,
        insertion_rate=INSERTION_RATE,
        deletion_rate=DELETION_RATE,
    )
    protocol.set_insertion_length_distributions(ZipfDistribution(p=1.53, truncation=50))
    protocol.set_deletion_length_distributions(ZipfDistribution(p=1.11, truncation=50))

    rate_model = SiteRateModelSpec(
        gamma_alpha=gene_entry["gtr_model"]["gamma_alpha"],
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=RHO,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
    model = ReplacementModelSpec(
        model=MODEL_CODES.LG,
        alphabet=ALPHABET_CODES.PROTEIN,
        site_rate_model=rate_model,
    )
    msa = Simulator(protocol, replacement_model=model)()

    sequences = []
    for i in range(msa.get_num_sequences()):
        row = msa.get_msa_row(i).split("\n")[1][:window]
        sequences.append(row + "-" * (window - len(row)))
    return sequences


def main():
    random.seed(SEED)
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_by_id = {gene_id(key): entry for key, entry in manifest.items()}
    zips_by_id = {gene_id(p.name): p for p in AA_DIR.glob("*.fasta.zip")}

    shared = sorted(set(manifest_by_id) & set(zips_by_id))
    print(f"manifest genes: {len(manifest_by_id)}, amino acid files: {len(zips_by_id)}, matched: {len(shared)}")
    if not shared:
        print("No matching gene ids. Example manifest keys:", list(manifest)[:3])
        print("Example zip names:", [p.name for p in list(AA_DIR.glob('*.fasta.zip'))[:3]])
        return
    genes = random.sample(shared, min(N_GENES, len(shared)))

    rows = []
    for n, gid in enumerate(genes, 1):
        real_seqs = read_zip_sequences(zips_by_id[gid])
        window = min(WINDOW, len(real_seqs[0]))
        real_div = log_divergence([s[:window] for s in real_seqs])
        for scale in SCALES:
            sim_div = log_divergence(simulate_protein(manifest_by_id[gid], scale, window))
            rows.append({"gene": gid, "scale": scale, "real": real_div, "sim": sim_div})
        if n % 50 == 0:
            print(f"  {n}/{len(genes)} genes")

    df = pd.DataFrame(rows)
    df.to_csv("protein_divergence_check.csv", index=False)

    fig, axes = plt.subplots(1, len(SCALES), figsize=(5 * len(SCALES), 5), sharex=True, sharey=True)
    print("\nlog divergence, simulated minus real (0 = perfect match):")
    for ax, scale in zip(axes, SCALES):
        d = df[df.scale == scale]
        diff = d.sim - d.real
        print(f"  scale {scale}: median {diff.median():+.2f} (ratio {np.exp(diff.median()):.2f}x), "
              f"spread {diff.std():.2f}, corr {np.corrcoef(d.real, d.sim)[0, 1]:.2f}")
        ax.scatter(d.real, d.sim, s=8, alpha=0.5)
        lims = [min(df.real.min(), df.sim.min()), max(df.real.max(), df.sim.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_title(f"tree scale x{scale}")
        ax.set_xlabel("real log divergence")
    axes[0].set_ylabel("simulated log divergence")
    plt.tight_layout()
    plt.savefig("protein_divergence_check.png", dpi=200)
    print("Saved protein_divergence_check.csv and protein_divergence_check.png")


if __name__ == "__main__":
    main()
