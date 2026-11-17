"""
Core rho with optimization, 200 genes.

Per gene: strip columns with >50% gaps, drop all-gap taxa (alignment + tree),
raxml-ng --evaluate (LG+G8, fixed topology) to re-fit alpha + branch lengths on the core,
then codeml with alpha + branches fixed, only rho free.

Resumable: genes with an existing result.json are skipped.
Output: OUT_DIR/core_opt_results.csv
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO

# ---- EDIT THESE ----
DATA_DIR = Path("data")
GAPPED_DIR = DATA_DIR / "real_gapped"                      # <gene>.fasta
TREE_DIR = DATA_DIR / "codeml_runs_real_gapped"            # <gene>/tree.nwk
GENE_LIST = DATA_DIR / "random_sample_200.csv"             # column "gene"
OUT_DIR = DATA_DIR / "core_genes_codeml"
RAXML = Path("bins/raxml-ng")
CODEML = Path("bins/paml-4.10.10-linux-x86_64/bin/codeml")
LG_DAT = Path("bins/paml-4.10.10-linux-x86_64/dat/lg.dat")
# --------------------
CUTOFF = 0.5
MODEL = "LG+G8"
RAXML_TIMEOUT = 3600
CODEML_TIMEOUT = 7200

CTL = """seqfile = aln.phy
treefile = tree.nwk
outfile = mlb
noisy = 0
verbose = 0
runmode = 0
seqtype = 2
aaRatefile = {lg}
model = 2
Mgene = 0
fix_kappa = 1
kappa = 1
fix_alpha = 1
alpha = {alpha}
ncatG = 8
fix_rho = 0
rho = 0.5
Malpha = 0
clock = 0
getSE = 0
RateAncestor = 0
cleandata = 0
fix_blength = 2
"""


def run_gene(gene: str):
    work = OUT_DIR / gene
    result_file = work / "result.json"
    if result_file.exists():
        return gene, json.loads(result_file.read_text()), None
    work.mkdir(parents=True, exist_ok=True)
    try:
        # --- strip ---
        recs = list(SeqIO.parse(GAPPED_DIR / f"{gene}.fasta", "fasta"))
        names = [f"s{i+1:03d}" for i in range(len(recs))]
        arr = np.array([list(str(r.seq)) for r in recs])
        keep = (arr == "-").mean(axis=0) <= CUTOFF
        core = arr[:, keep]
        has_data = (core != "-").any(axis=1)
        names = [n for n, h in zip(names, has_data) if h]
        core = core[has_data]
        n_dropped = int((~has_data).sum())

        # --- prune tree ---
        tree = Phylo.read(TREE_DIR / gene / "tree.nwk", "newick")
        tips = {t.name for t in tree.get_terminals()}
        for t in [x for x in tree.get_terminals() if x.name not in set(names)]:
            tree.prune(t)
        shared = [n for n in names if n in tips]
        idx = [names.index(n) for n in shared]
        core = core[idx]
        out = StringIO()
        Phylo.write(tree, out, "newick")
        (work / "start.nwk").write_text(out.getvalue().strip() + "\n")
        (work / "core.fasta").write_text(
            "".join(f">{n}\n{''.join(row)}\n" for n, row in zip(shared, core)))

        # --- raxml-ng: re-fit alpha + branches ---
        r = subprocess.run(
            [str(RAXML.resolve()), "--evaluate", "--msa", "core.fasta", "--msa-format", "FASTA",
             "--data-type", "AA", "--tree", "start.nwk", "--model", MODEL,
             "--prefix", "out", "--threads", "1", "--redo"],
            cwd=work, capture_output=True, text=True, timeout=RAXML_TIMEOUT)
        if r.returncode != 0:
            return gene, None, "raxml failed: " + (r.stdout + r.stderr).strip()[-200:]
        alpha = float(re.search(r"\{([0-9.eE+-]+)\}", (work / "out.raxml.bestModel").read_text()).group(1))
        best_tree = (work / "out.raxml.bestTree").read_text().strip()
        (work / "tree.nwk").write_text(best_tree + "\n")
        tree_len = sum(c.branch_length or 0 for c in Phylo.read(StringIO(best_tree), "newick").find_clades())

        # --- codeml: alpha + branches fixed, rho free ---
        with open(work / "aln.phy", "w") as f:
            f.write(f" {len(shared)} {core.shape[1]}\n")
            for n, row in zip(shared, core):
                f.write(f"{n}  {''.join(row)}\n")
        (work / "c.ctl").write_text(CTL.format(lg=LG_DAT.resolve(), alpha=alpha))
        subprocess.run([str(CODEML.resolve()), "c.ctl"], cwd=work, capture_output=True,
                       text=True, timeout=CODEML_TIMEOUT)
        m = re.search(r"rho \(correlation\)\s*=\s*([0-9.]+)", (work / "mlb").read_text())
        if not m:
            return gene, None, "no rho in mlb"

        res = {"gene": gene, "alpha_core": alpha, "tree_length_core": tree_len,
               "n_cols_core": int(core.shape[1]), "n_taxa_core": len(shared),
               "n_taxa_dropped": n_dropped, "rho_codeml_core_opt": float(m.group(1))}
        result_file.write_text(json.dumps(res))
        return gene, res, None
    except Exception as e:
        import traceback
        return gene, None, traceback.format_exc()[-600:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n-genes", type=int, default=None, help="test on the first N genes")
    args = ap.parse_args()

    genes = pd.read_csv(GENE_LIST)["gene"].astype(str).tolist()
    if args.n_genes:
        genes = genes[:args.n_genes]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_gene, g) for g in genes]
        for i, fut in enumerate(as_completed(futs), 1):
            gene, res, err = fut.result()
            if res:
                rows.append(res)
            else:
                failures.append((gene, err))
                print(f"FAIL {gene}: {err}")
            if i % 10 == 0:
                print(f"{i}/{len(genes)}")

    pd.DataFrame(rows).sort_values("gene").to_csv(OUT_DIR / "core_opt_results.csv", index=False)
    if failures:
        pd.DataFrame(failures, columns=["gene", "reason"]).to_csv(OUT_DIR / "failures.csv", index=False)
    print(f"done: {len(rows)} ok, {len(failures)} failed")


if __name__ == "__main__":
    main()
