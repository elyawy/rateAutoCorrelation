"""
Re-estimate branch lengths and gamma alpha on the protein alignments with raxml-ng.

For each gene: take the topology from the simulation manifest, keep only taxa present in both
the tree and the amino acid alignment, then run `raxml-ng --evaluate` (LG+G8, fixed topology)
to get a protein-scaled tree and a protein alpha. Writes a new manifest with those values.
"""

import argparse
import csv
import json
import random
import re
import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

from Bio import Phylo, SeqIO

AA_DIR = Path("/home/pupkolab/temp/orthomam_AA")
MANIFEST_PATH = Path("/home/pupkolab/Dev/SailfishBenchmark/simulation_manifest.json")
RAXML_PATH = Path("/home/pupkolab/Apps/raxml-ng")  # the executable, or the folder holding it
OUTPUT_MANIFEST = Path("protein_manifest.json")
FAILURE_LOG = Path("protein_manifest_failures.csv")
MODEL = "LG+G8"
N_WORKERS = 7
SEED = 0
SAVE_EVERY = 500  # genes between saves of the output manifest
MIN_TAXA = 4
RAXML_TIMEOUT = 600  # seconds per gene


def gene_id(name: str) -> str:
    """'10000_NT_AL_AA.fasta.zip' or '10000_AKT3' -> '10000'."""
    return name.split("_")[0]


def find_raxml(path: Path) -> Path:
    if path.is_dir():
        for candidate in (path / "raxml-ng", path / "bin" / "raxml-ng"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no raxml-ng executable found in {path}")
    return path


def read_zip_alignment(zip_path: Path) -> dict:
    """Return {taxon: sequence} from the FASTA inside a zip."""
    with zipfile.ZipFile(zip_path) as zf:
        fasta_name = next(f for f in zf.namelist() if f.endswith(".fasta"))
        content = zf.read(fasta_name).decode("utf-8")
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(StringIO(content), "fasta")}


def prune_tree(newick: str, keep: set):
    """Drop tips not in `keep`. Returns (newick string, number of tips removed)."""
    tree = Phylo.read(StringIO(newick), "newick")
    removed = 0
    for tip in [t.name for t in tree.get_terminals()]:
        if tip not in keep:
            tree.prune(name=tip)
            removed += 1
    out = StringIO()
    Phylo.write(tree, out, "newick")
    return out.getvalue().strip(), removed


def parse_alpha(model_text: str) -> float:
    """'LG+G8m{0.719226}, noname1 = 1-500' -> 0.719226"""
    return float(re.search(r"\{([0-9.eE+-]+)\}", model_text).group(1))


def process_gene(key: str, newick: str, zip_path: str, raxml: str):
    """Returns (key, entry_dict, None) on success or (key, None, reason) on failure."""
    try:
        alignment = read_zip_alignment(Path(zip_path))
        tree_tips = {t.name for t in Phylo.read(StringIO(newick), "newick").get_terminals()}
        shared = tree_tips & set(alignment)
        if len(shared) < MIN_TAXA:
            return key, None, f"only {len(shared)} shared taxa"

        pruned_newick, n_tips_removed = prune_tree(newick, shared)
        n_rows_dropped = len(alignment) - len(shared)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            msa_file, tree_file = tmp / "aln.fasta", tmp / "tree.nwk"
            msa_file.write_text("".join(f">{t}\n{alignment[t]}\n" for t in sorted(shared)))
            tree_file.write_text(pruned_newick + "\n")

            result = subprocess.run(
                [raxml, "--evaluate", "--msa", str(msa_file), "--msa-format", "FASTA",
                 "--data-type", "AA", "--tree", str(tree_file), "--model", MODEL,
                 "--prefix", str(tmp / "out"), "--threads", "1", "--redo"],
                capture_output=True, text=True, timeout=RAXML_TIMEOUT,
            )
            if result.returncode != 0:
                return key, None, "raxml-ng failed: " + (result.stdout + result.stderr).strip()[-200:]

            new_tree = (tmp / "out.raxml.bestTree").read_text().strip()
            alpha = parse_alpha((tmp / "out.raxml.bestModel").read_text())

        entry = {
            "newick": new_tree,
            "gamma_alpha": alpha,
            "n_taxa": len(shared),
            "n_tree_tips_removed": n_tips_removed,
            "n_alignment_rows_dropped": n_rows_dropped,
        }
        return key, entry, None
    except Exception as e:
        return key, None, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=None, help="run a random subset of N genes")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    raxml = str(find_raxml(RAXML_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text())
    zips_by_id = {gene_id(p.name): p for p in AA_DIR.glob("*.fasta.zip")}

    jobs = [(key, entry["newick"], str(zips_by_id[gene_id(key)]), raxml)
            for key, entry in manifest.items() if gene_id(key) in zips_by_id]
    print(f"manifest genes: {len(manifest)}, with a protein alignment: {len(jobs)}")
    if args.n_genes:
        jobs = random.Random(SEED).sample(jobs, min(args.n_genes, len(jobs)))
        print(f"running a random subset of {len(jobs)} genes")

    def save():
        OUTPUT_MANIFEST.write_text(json.dumps(new_manifest))
        with open(FAILURE_LOG, "w", newline="") as f:
            csv.writer(f).writerows([("gene", "reason"), *failures])

    new_manifest, failures = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_gene, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            key, entry, reason = future.result()
            if entry is not None:
                new_manifest[key] = entry
            else:
                failures.append((key, reason))
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done ({len(failures)} failed)")
            if done % SAVE_EVERY == 0:
                save()

    save()

    pruned = sum(1 for e in new_manifest.values()
                 if e["n_tree_tips_removed"] or e["n_alignment_rows_dropped"])
    print(f"\nSaved {len(new_manifest)} genes to {OUTPUT_MANIFEST}, failures logged in {FAILURE_LOG}")
    print(f"genes where tree and alignment taxa did not match exactly: {pruned}")


if __name__ == "__main__":
    main()