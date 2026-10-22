"""
Re-estimate branch lengths and gamma alpha (LG+G8, fixed topology) with raxml-ng
for the genes listed in selected_msas.csv, on the cleaned unfiltered alignments (see msa_cleaning.py).
Writes one row per gene to protein_estimates.csv, and the alignments actually used (cleaned, restricted
to the taxa in the tree) to protein_msas.parquet, one row per sequence.
"""

import csv
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio import Phylo

from msa_cleaning import clean_alignment, find_alignments, read_records

SELECTED_CSV = Path("selected_msas.csv")
TREE_DIR = Path("data/raw/trees")  # {gene}_NT_AL.rootree
RAXML_PATH = Path("bins/raxml-ng")
OUTPUT = Path("protein_estimates.csv")
MSA_OUTPUT = Path("protein_msas.parquet")
FAILURE_LOG = Path("protein_estimates_failures.csv")
MODEL = "LG+G8"
N_WORKERS = 6
RAXML_TIMEOUT = 3600  # seconds per gene
FIELDS = ["gene", "alpha", "final_loglik", "n_taxa", "n_tips_removed", "n_special_dropped", "newick"]


def process_gene(gene: str, aln_path: str, raxml: str):
    """Returns ((row_dict, msa_rows), None) on success or (None, (gene, reason)) on failure."""
    try:
        tree = Phylo.read(TREE_DIR / f"{gene}_NT_AL.rootree", "newick")
        alignment, n_special, reason = clean_alignment(
            read_records(Path(aln_path)), allowed={t.name for t in tree.get_terminals()})
        if alignment is None:
            return None, (gene, reason)

        n_removed = 0
        for tip in [t.name for t in tree.get_terminals()]:
            if tip not in alignment:
                tree.prune(name=tip)
                n_removed += 1
        buf = StringIO()
        Phylo.write(tree, buf, "newick")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "aln.fasta").write_text("".join(f">{t}\n{alignment[t]}\n" for t in sorted(alignment)))
            (tmp / "tree.nwk").write_text(buf.getvalue())

            result = subprocess.run(
                [raxml, "--evaluate", "--fast", "--msa", str(tmp / "aln.fasta"), "--msa-format", "FASTA",
                 "--data-type", "AA", "--tree", str(tmp / "tree.nwk"), "--model", MODEL,
                 "--prefix", str(tmp / "out"), "--threads", "1", "--redo"],
                capture_output=True, text=True, timeout=RAXML_TIMEOUT,
            )
            if result.returncode != 0:
                return None, (gene, "raxml-ng failed: " + (result.stdout + result.stderr).strip()[-200:])

            new_tree = (tmp / "out.raxml.bestTree").read_text().strip()
            model_text = (tmp / "out.raxml.bestModel").read_text()
            alpha = float(re.search(r"\{([0-9.eE+-]+)\}", model_text).group(1))
            loglik = float(re.search(r"Final LogLikelihood:\s*(\S+)",
                                     (tmp / "out.raxml.log").read_text()).group(1))

        return ({"gene": gene, "alpha": alpha, "final_loglik": loglik, "n_taxa": len(alignment),
                "n_tips_removed": n_removed, "n_special_dropped": n_special,
                "newick": new_tree}, [(gene, t, alignment[t]) for t in sorted(alignment)]), None
    except Exception as e:
        return None, (gene, f"{type(e).__name__}: {e}")


def load_previous():
    """Reload the outputs of an interrupted run, so those genes are kept and not redone.
    Only genes present in both the csv and the parquet are kept. Delete the failure log to retry failed genes."""
    rows, msa_rows, failures = [], [], []
    if OUTPUT.exists() and MSA_OUTPUT.exists():
        with open(OUTPUT, newline="") as f:
            rows = list(csv.DictReader(f))
        t = pq.read_table(MSA_OUTPUT).to_pydict()
        msa_rows = list(zip(t["gene"], t["taxon"], t["sequence"]))
        both = {r["gene"] for r in rows} & set(t["gene"])
        rows = [r for r in rows if r["gene"] in both]
        msa_rows = [m for m in msa_rows if m[0] in both]
    if FAILURE_LOG.exists():
        with open(FAILURE_LOG, newline="") as f:
            failures = [tuple(r) for r in list(csv.reader(f))[1:]]
    return rows, msa_rows, failures


def main():
    genes = [row["gene"] for row in csv.DictReader(open(SELECTED_CSV))]
    raxml = str(RAXML_PATH)
    alignments = find_alignments()
    missing = [g for g in genes if g not in alignments]
    genes = [g for g in genes if g in alignments]
    print(f"{len(genes)} genes to process ({len(missing)} in selected_msas.csv have no alignment file)")

    rows, msa_rows, failures = load_previous()
    skip = {r["gene"] for r in rows} | {g for g, _ in failures}
    genes = [g for g in genes if g not in skip]
    print(f"resuming: {len(rows)} genes done, {len(failures)} failed earlier, {len(genes)} left")

    def save():
        with open(OUTPUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        genes_col, taxa_col, seqs_col = zip(*msa_rows) if msa_rows else ((), (), ())
        pq.write_table(pa.table({"gene": genes_col, "taxon": taxa_col, "sequence": seqs_col}),
                       MSA_OUTPUT, compression="zstd")
        with open(FAILURE_LOG, "w", newline="") as f:
            csv.writer(f).writerows([("gene", "reason"), *failures])

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_gene, g, str(alignments[g]), raxml) for g in genes]
        for done, fut in enumerate(as_completed(futures), 1):
            result, failure = fut.result()
            if result:
                rows.append(result[0])
                msa_rows.extend(result[1])
            else:
                failures.append(failure)
            if done % 50 == 0 or done == len(genes):
                print(f"  {done}/{len(genes)} done ({len(failures)} failed)")
                save()

    save()
    print(f"Saved {len(rows)} genes to {OUTPUT} and {MSA_OUTPUT}, failures in {FAILURE_LOG}")


if __name__ == "__main__":
    main()