"""
Run codeml (fixed tree, fixed alpha, free rho) on the real OrthoMaM protein MSAs.

Input (from export_trees_and_msas.py, plus the CSV for alpha):
    trees/{gene}.nwk, msas/{gene}.fasta, protein_estimates.csv
Output:
    codeml_runs_real/{gene}/mlb, codeml_runs_real/timing.csv
Genes whose mlb already exists are skipped, so the script can be re-run to resume.
"""

import argparse
import csv
import os
import pathlib
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "codeml_template.ctl"
ESTIMATES_CSV = pathlib.Path("data/protein_estimates.csv")
TREE_DIR = pathlib.Path("data/trees")
MSA_DIR = pathlib.Path("data/msas")
RUNS_DIR = pathlib.Path("data/codeml_runs_real")


def read_fasta(path):
    records, name, cur = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(cur)))
            name, cur = line[1:].strip(), []
        else:
            cur.append(line.strip())
    records.append((name, "".join(cur)))
    return records


def run_gene(gene, alpha, tree_dir, msa_dir, runs_dir):
    run_dir = (runs_dir / gene).resolve()
    if (run_dir / "mlb").exists():
        return gene, True, 0.0
    run_dir.mkdir(parents=True, exist_ok=True)

    # short taxon names: codeml chokes on long ones
    records = read_fasta(msa_dir / f"{gene}.fasta")
    mapping = {name: f"s{i:03d}" for i, (name, _) in enumerate(records, start=1)}
    tree = (tree_dir / f"{gene}.nwk").read_text().strip()
    tree = re.sub(r"([(,])([^():,]+):", lambda m: m.group(1) + mapping[m.group(2)] + ":", tree)
    (run_dir / "tree.nwk").write_text(tree + "\n")

    with open(run_dir / "alignment.phy", "w") as f:
        f.write(f" {len(records)} {len(records[0][1])}\n")
        for name, seq in records:
            f.write(f"{mapping[name]}  {seq}\n")

    ctl = (TEMPLATE.read_text()
           .replace("SEQFILE_PLACEHOLDER", "alignment.phy")
           .replace("TREEFILE_PLACEHOLDER", "tree.nwk")
           .replace("WAGFILE_PLACEHOLDER", str(pathlib.Path(config.MODELDAT_FILE).resolve()))
           .replace("ALPHA_PLACEHOLDER", str(alpha)))
    (run_dir / "control.ctl").write_text(ctl)

    start = time.time()
    try:
        subprocess.run([str(pathlib.Path(config.CODEML_EXECUTABLE).resolve()), "control.ctl"], cwd=run_dir,
                       capture_output=True, text=True, timeout=86400)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    mlb = run_dir / "mlb"
    return gene, mlb.exists() and mlb.stat().st_size > 0, time.time() - start


def main():
    default_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=default_cores)
    parser.add_argument("--tree-dir", type=pathlib.Path, default=TREE_DIR)
    parser.add_argument("--msa-dir", type=pathlib.Path, default=MSA_DIR)
    parser.add_argument("--runs-dir", type=pathlib.Path, default=RUNS_DIR)
    args = parser.parse_args()

    alphas = {row["gene"]: row["alpha"] for row in csv.DictReader(open(ESTIMATES_CSV))}
    genes = sorted(g.stem for g in args.msa_dir.glob("*.fasta"))
    args.runs_dir.mkdir(exist_ok=True)
    print(f"{len(genes)} genes, {args.cores} cores")

    done = 0
    with open(args.runs_dir / "timing.csv", "a", newline="") as f, \
            ProcessPoolExecutor(max_workers=args.cores) as ex:
        writer = csv.writer(f)
        futures = [ex.submit(run_gene, g, alphas[g], args.tree_dir, args.msa_dir, args.runs_dir)
                   for g in genes]
        for fut in as_completed(futures):
            gene, ok, secs = fut.result()
            writer.writerow([gene, ok, f"{secs:.1f}"])
            f.flush()
            done += 1
            if done % 10 == 0 or done == len(genes):
                print(f"  {done}/{len(genes)}")


if __name__ == "__main__":
    main()