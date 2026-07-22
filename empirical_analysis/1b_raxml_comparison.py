"""
Step 1b: Run RAxML-NG on a filtered subset of OrthoMaM alignments
and compare inferred alpha to SBI predictions.

For each sampled alignment:
  - Download tree from OrthoMaM server
  - Extract AA FASTA from local zip
  - Run RAxML-NG --evaluate with fixed tree topology (WAG+G4)
  - Parse alpha from RAxML output
  - Join with existing SBI predictions and save CSV
"""

import pathlib
import zipfile
import tempfile
import subprocess
import re
import requests
import pandas as pd
import numpy as np
from io import StringIO
from Bio import SeqIO

import config

# ==========================================
# CONSTANTS
# ==========================================
TREE_BASE_URL = "https://orthomam.mbb.cnrs.fr/orthomam_v12/cds/trees/"
MSA_FOLDER = pathlib.Path("/home/pupkolab/temp/orthomam_AA")
STATS_CSV = pathlib.Path("/home/pupkolab/Dev/rateAutoCorrelation/empirical_analysis/orthomam_stats.csv")
OUTPUT_CSV = config.RESULTS_DIR / "raxml_comparison.csv"
N_SAMPLE = 1000
RANDOM_SEED = 42


def gene_name_from_zip(zip_path: pathlib.Path) -> str:
    """Extract gene name from zip filename.
    e.g. '10000_NT_AL_AA.fasta.zip' -> '10000'
    """
    return zip_path.name.split("_")[0]


def download_tree(gene: str, dest: pathlib.Path) -> bool:
    """Download NT tree for gene from OrthoMaM. Returns True on success."""
    url = f"{TREE_BASE_URL}{gene}_NT_AL.rootree"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_text(r.text)
        return True
    except Exception as e:
        print(f"  WARNING: Could not download tree for {gene}: {e}")
        return False


def extract_aa_fasta(zip_path: pathlib.Path, dest: pathlib.Path) -> bool:
    """Extract AA fasta from zip to dest. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            fasta_files = [f for f in zf.namelist() if f.endswith('.fasta')]
            if not fasta_files:
                print(f"  WARNING: No fasta in {zip_path.name}")
                return False
            with zf.open(fasta_files[0]) as f:
                dest.write_bytes(f.read())
        return True
    except Exception as e:
        print(f"  WARNING: Could not extract {zip_path.name}: {e}")
        return False


def run_raxml(fasta: pathlib.Path, tree: pathlib.Path, workdir: pathlib.Path) -> float | None:
    """
    Run RAxML-NG --evaluate on fixed tree, return inferred alpha or None.
    """
    cmd = [
        config.RAXML_EXECUTABLE,
        "--evaluate",
        "--msa", str(fasta),
        "--tree", str(tree),
        "--model", config.RAXML_MODEL,
        "--prefix", str(workdir / "raxml_out"),
        "--redo",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  WARNING: RAxML failed:\n{result.stderr[-500:]}")
            return None
        return parse_raxml_alpha(workdir / "raxml_out.raxml.log")
    except subprocess.TimeoutExpired:
        print("  WARNING: RAxML timed out")
        return None
    except Exception as e:
        print(f"  WARNING: RAxML error: {e}")
        return None


def parse_raxml_alpha(log_path: pathlib.Path) -> float | None:
    """Parse alpha from RAxML-NG log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # RAxML-NG logs: "alpha: 0.XXXX" or "Gamma shape alpha: 0.XXXX"
    match = re.search(r"alpha[:\s]+([0-9]+\.[0-9]+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    print(f"  WARNING: Could not parse alpha from {log_path.name}")
    return None


def process_gene(gene: str, zip_path: pathlib.Path) -> dict | None:
    """Full pipeline for one gene. Returns result dict or None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        # Download tree
        print(f"Processing gene {gene}...")
        tree_path = tmpdir / f"{gene}.tree"
        if not download_tree(gene, tree_path):
            return None

        # Extract fasta
        fasta_path = tmpdir / f"{gene}.fasta"
        if not extract_aa_fasta(zip_path, fasta_path):
            return None

        # Run RAxML
        raxml_alpha = run_raxml(fasta_path, tree_path, tmpdir)
        if raxml_alpha is None:
            return None

        return {"gene": gene, "raxml_alpha": raxml_alpha}


def main():
    print("=" * 60)
    print("RAXML-NG COMPARISON")
    print("=" * 60)

    # Load stats, apply filters, sample
    print("Loading and filtering OrthoMaM stats...")
    stats_df = pd.read_csv(STATS_CSV)
    filtered = stats_df[
        (stats_df["msa_length"] > 500) &
        (stats_df["num_taxa"] > 50)
    ]
    print(f"  {len(filtered)} alignments after filtering")

    sample = filtered.sample(n=min(N_SAMPLE, len(filtered)), random_state=RANDOM_SEED)
    print(f"  Sampled {len(sample)} alignments")


    # Build gene -> zip_path map
    zip_files = {p.name.split("_")[0]: p for p in MSA_FOLDER.glob("*.fasta.zip")}
    # Process
    print("\nProcessing alignments...")
    print("-" * 60)

    results = []
    genes = sample["filename"].astype(str).tolist() if "filename" in sample.columns else sample.iloc[:, 0].astype(str).tolist()

    for idx, gene in enumerate(genes, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  {idx}/{len(genes)}: gene {gene}")

        zip_path = zip_files.get(gene.split("_")[0])  # Match gene name to zip filename
        if zip_path is None:
            print(f"  WARNING: No zip found for gene {gene}")
            continue

        result = process_gene(gene.split("_")[0], zip_path)
        if result:
            results.append(result)

    print(f"\nSuccessfully processed {len(results)}/{len(genes)}")

    # Build output dataframe and join with SBI predictions
    raxml_df = pd.DataFrame(results)
    # merged = raxml_df.merge(predictions, left_on="gene", right_on="filename", how="left")

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raxml_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"\nRows with both estimates: {raxml_df['raxml_alpha'].notna().sum()}")
    if "raxml_alpha" in raxml_df and "pred_alpha" in raxml_df:
        valid = raxml_df.dropna(subset=["raxml_alpha", "pred_alpha"])
        corr = valid["raxml_alpha"].corr(valid["pred_alpha"])
        print(f"Pearson r (raxml vs SBI alpha): {corr:.3f}")


if __name__ == "__main__":
    main()
