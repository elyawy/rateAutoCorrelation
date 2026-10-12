"""
Step 1 (protein version): extract features from OrthoMaM amino acid alignments.

Cuts each alignment into non-overlapping WINDOW-site chunks and saves the chunk features to one npz
(one row per chunk). Per-gene info, including genes skipped for being too short, goes to a CSV.
"""

import pathlib
import sys
import zipfile
from io import StringIO

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import common.config as config

sys.path.insert(0, str(REPO_DIR / "common"))
from features_calculator import calculate_cnn_input, calculate_scalar_features

AA_DIR = pathlib.Path("/home/elyalab/temp/orthomam_AA")
WINDOW = 250  # amino acid sites
ALPHABET_SIZE = 20
FEATURES_NAME = f"orthomam_features_amino_w{WINDOW}.npz"
GENE_META_NAME = f"orthomam_gene_meta_amino_w{WINDOW}.csv"


def extract_fasta_from_zip(zip_path):
    """Return (filename, fasta_content_string) from a .fasta.zip, or (None, None) on error."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            fasta_files = [f for f in zip_ref.namelist() if f.endswith(".fasta")]
            if not fasta_files:
                print(f"  WARNING: No FASTA file found in {zip_path.name}")
                return None, None
            if len(fasta_files) > 1:
                print(f"  WARNING: Multiple FASTA files in {zip_path.name}, using first")
            with zip_ref.open(fasta_files[0]) as f:
                return pathlib.Path(fasta_files[0]).name, f.read().decode("utf-8")
    except Exception as e:
        print(f"  ERROR extracting {zip_path.name}: {e}")
        return None, None


def parse_fasta_content(fasta_content):
    return [str(record.seq) for record in SeqIO.parse(StringIO(fasta_content), "fasta")]


def calculate_alignment_metadata(sequences):
    """n_sequences, alignment_length, n_variable_sites (columns with >1 distinct non-gap residue)."""
    alignment_length = len(sequences[0])
    n_variable = 0
    for col_idx in range(alignment_length):
        unique_residues = set(seq[col_idx] for seq in sequences if seq[col_idx] != "-")
        if len(unique_residues) > 1:
            n_variable += 1
    return {
        "n_sequences": len(sequences),
        "alignment_length": alignment_length,
        "n_variable_sites": n_variable,
    }


def looks_like_dna(sequences, n_check=20):
    chars = "".join(sequences[:n_check]).upper().replace("-", "")
    return len(chars) == 0 or sum(chars.count(b) for b in "ACGT") / len(chars) >= 0.9


def main():
    zip_files = sorted(AA_DIR.glob("*.fasta.zip"))
    if not zip_files:
        print(f"ERROR: No .fasta.zip files found in {AA_DIR}")
        return
    print(f"Found {len(zip_files)} zip files")

    features, scalars, chunk_genes = [], [], []
    gene_rows = []

    for idx, zip_path in enumerate(zip_files, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{len(zip_files)}: {zip_path.name}")

        filename, content = extract_fasta_from_zip(zip_path)
        if filename is None:
            gene_rows.append({"filename": zip_path.name, "status": "error"})
            continue
        sequences = parse_fasta_content(content)
        if not sequences:
            gene_rows.append({"filename": filename, "status": "empty"})
            continue

        row = {"filename": filename, **calculate_alignment_metadata(sequences)}
        if looks_like_dna(sequences):
            row.update(status="dna", n_chunks=0)
        elif row["alignment_length"] < WINDOW:
            row.update(status="too_short", n_chunks=0)
        else:
            n_chunks = 0
            for start in range(0, row["alignment_length"] - WINDOW + 1, WINDOW):
                chunk = [seq[start:start + WINDOW] for seq in sequences]
                features.append(calculate_cnn_input(chunk, alphabet_size=ALPHABET_SIZE))
                scalars.append(calculate_scalar_features(chunk, alphabet_size=ALPHABET_SIZE))
                chunk_genes.append(filename)
                n_chunks += 1
            row.update(status="ok", n_chunks=n_chunks)
        gene_rows.append(row)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.RESULTS_DIR / FEATURES_NAME,
        features=torch.stack(features).numpy(),
        scalars=torch.stack(scalars).numpy(),
        gene=np.array(chunk_genes),
        window=WINDOW,
    )
    meta = pd.DataFrame(gene_rows)
    meta.to_csv(config.RESULTS_DIR / GENE_META_NAME, index=False)

    print(f"\nSaved {len(features)} chunks to {config.RESULTS_DIR / FEATURES_NAME}")
    print(f"Gene status counts: {meta['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
