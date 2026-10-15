"""
Step 1 (protein version): extract features from OrthoMaM amino acid alignments.

Computes the features on each whole alignment (no chunking, so each keeps its own length).
The alignments are split across N_WORKERS processes that each start once and work through their own share.
Features of all genes are concatenated along the site axis into one array, with a lengths array to split
them again. Per-gene info, including genes skipped for being too short, goes to a CSV.
"""

import pathlib
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from io import StringIO

import numpy as np
import pandas as pd
from Bio import SeqIO

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import common.config as config

sys.path.insert(0, str(REPO_DIR / "common"))
from features_calculator import calculate_cnn_input, calculate_scalar_features

AA_DIR = pathlib.Path("/home/pupkolab/temp/orthomam_AA")
MIN_LENGTH = 100  # amino acid sites; shorter alignments are skipped (shortest length the regressor was tested on)
ALPHABET_SIZE = 20
N_WORKERS = 4
FEATURES_NAME = "orthomam_features_amino_full.npz"
GENE_META_NAME = "orthomam_gene_meta_amino_full.csv"


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


def process_gene(zip_path):
    """Return (row, features, scalars); features and scalars are None unless the gene is ok."""
    filename, content = extract_fasta_from_zip(zip_path)
    if filename is None:
        return {"filename": zip_path.name, "status": "error"}, None, None
    sequences = parse_fasta_content(content)
    if not sequences:
        return {"filename": filename, "status": "empty"}, None, None

    row = {"filename": filename, **calculate_alignment_metadata(sequences)}
    if looks_like_dna(sequences):
        row.update(status="dna")
        return row, None, None
    if row["alignment_length"] < MIN_LENGTH:
        row.update(status="too_short")
        return row, None, None

    row.update(status="ok")
    return (
        row,
        calculate_cnn_input(sequences, alphabet_size=ALPHABET_SIZE).numpy(),
        calculate_scalar_features(sequences, alphabet_size=ALPHABET_SIZE).numpy(),
    )


def process_share(worker_id, indexed_paths):
    """Run in one worker process: handle this worker's share of (position, zip_path) pairs."""
    results = []
    for n, (idx, zip_path) in enumerate(indexed_paths, 1):
        results.append((idx, *process_gene(zip_path)))
        if n % 100 == 0:
            print(f"  worker {worker_id}: {n}/{len(indexed_paths)}")
    return results


def main():
    zip_files = sorted(AA_DIR.glob("*.fasta.zip"))
    if not zip_files:
        print(f"ERROR: No .fasta.zip files found in {AA_DIR}")
        return
    print(f"Found {len(zip_files)} zip files, using {N_WORKERS} workers")

    # interleaved shares (every N_WORKERS-th file), so big and small alignments spread evenly over the workers
    indexed = list(enumerate(zip_files))
    shares = [indexed[w::N_WORKERS] for w in range(N_WORKERS)]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_share, w, share) for w, share in enumerate(shares)]
        results = [r for future in futures for r in future.result()]
    results.sort(key=lambda r: r[0])  # back to the original file order

    gene_rows = [row for _, row, _, _ in results]
    ok = [(row["filename"], feat, scal) for _, row, feat, scal in results if feat is not None]
    genes = [g for g, _, _ in ok]
    features = [f for _, f, _ in ok]
    scalars = [s for _, _, s in ok]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.RESULTS_DIR / FEATURES_NAME,
        features=np.concatenate(features, axis=1),  # (2, total number of sites over all genes)
        lengths=np.array([f.shape[1] for f in features]),  # sites per gene, to split `features` again
        scalars=np.stack(scalars),
        gene=np.array(genes),
    )
    meta = pd.DataFrame(gene_rows)
    meta.to_csv(config.RESULTS_DIR / GENE_META_NAME, index=False)

    print(f"\nSaved features for {len(features)} alignments to {config.RESULTS_DIR / FEATURES_NAME}")
    print(f"Gene status counts: {meta['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()