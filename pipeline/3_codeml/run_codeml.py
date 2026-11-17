#!/usr/bin/env python3
"""Run codeml for simulated, real, or stripped-core alignments."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config
from common.codeml import append_timing_csv, create_control_file, run_codeml


def parse_true_alpha(sim_name):
    match = re.search(r"_a([0-9.]+)_r", sim_name)
    if not match:
        raise ValueError(f"Could not parse alpha from simulation name: {sim_name}")
    return float(match.group(1))


def completed(run_dir):
    mlb = Path(run_dir) / "mlb"
    return mlb.exists() and "rho (correlation)" in mlb.read_text(errors="replace")


def read_fasta(path):
    records, name, current = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(current)))
            name, current = line[1:].strip(), []
        else:
            current.append(line.strip())
    if name is not None:
        records.append((name, "".join(current)))
    return records


def write_phylip(records, path):
    if not records:
        raise ValueError(f"No sequences found in {path}")
    length = len(records[0][1])
    with Path(path).open("w") as fh:
        fh.write(f" {len(records)} {length}\n")
        for name, sequence in records:
            fh.write(f"{name}  {sequence}\n")


def process_sim(sim_dir, runs_dir, template_path):
    sim_dir = Path(sim_dir)
    run_dir = Path(runs_dir) / sim_dir.name
    if completed(run_dir):
        return sim_dir.name, True, 0.0
    if (run_dir / "mlb").exists():
        (run_dir / "mlb").unlink()

    alignment = sim_dir / "alignment.phy"
    tree = sim_dir / "tree.newick"
    if not alignment.exists() or not tree.exists():
        missing = alignment if not alignment.exists() else tree
        return sim_dir.name, False, 0.0, f"Missing input: {missing}"

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(alignment, run_dir / "alignment.phy")
    control = create_control_file(
        template_path,
        run_dir,
        "alignment.phy",
        tree.resolve(),
        parse_true_alpha(sim_dir.name),
    )
    success, elapsed = run_codeml(control, run_dir, timeout=config.CODEML_TIMEOUT)
    return sim_dir.name, success and completed(run_dir), elapsed, None


def process_real(gene, alpha, tree_dir, msa_dir, runs_dir, template_path, timeout):
    run_dir = Path(runs_dir) / gene
    if completed(run_dir):
        return gene, True, 0.0
    if (run_dir / "mlb").exists():
        (run_dir / "mlb").unlink()

    msa_path = Path(msa_dir) / f"{gene}.fasta"
    tree_path = Path(tree_dir) / f"{gene}.nwk"
    records = read_fasta(msa_path)
    mapping = {name: f"s{i:03d}" for i, (name, _) in enumerate(records, start=1)}
    tree = tree_path.read_text().strip()
    tree = re.sub(
        r"([(,])([^():,]+):",
        lambda match: match.group(1) + mapping[match.group(2)] + ":",
        tree,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tree.nwk").write_text(tree + "\n")
    write_phylip([(mapping[name], sequence) for name, sequence in records], run_dir / "alignment.phy")
    control = create_control_file(
        template_path,
        run_dir,
        "alignment.phy",
        "tree.nwk",
        alpha,
    )
    success, elapsed = run_codeml(control, run_dir, timeout=timeout)
    return gene, success and completed(run_dir), elapsed


def run_sim(args):
    sim_root = Path(config.SIMULATED_DATA_DIR)
    runs_root = Path(config.CODEML_RUNS_DIR)
    sims = [sim_root / args.sim] if args.sim else sorted(p for p in sim_root.iterdir() if p.is_dir())
    runs_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.cores) as pool:
        futures = [pool.submit(process_sim, sim, runs_root, config.CODEML_TEMPLATE) for sim in sims]
        for future in as_completed(futures):
            result = future.result()
            name, success, elapsed, *error = result
            append_timing_csv(runs_root / "timing.csv", name, success, elapsed)
            if error and error[0]:
                print(f"WARNING [{name}]: {error[0]}")


def run_real(args):
    estimates = Path(config.REAL_ESTIMATES_CSV)
    alphas = {row["gene"]: float(row["alpha"]) for row in csv.DictReader(estimates.open())}
    msa_dir = Path(args.msa_dir)
    genes = sorted(path.stem for path in msa_dir.glob("*.fasta"))
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.cores) as pool:
        futures = [
            pool.submit(
                process_real,
                gene,
                alphas[gene],
                args.tree_dir,
                msa_dir,
                runs_dir,
                config.CODEML_TEMPLATE,
                args.timeout,
            )
            for gene in genes
            if gene in alphas
        ]
        for future in as_completed(futures):
            gene, success, elapsed = future.result()
            append_timing_csv(runs_dir / "timing.csv", gene, success, elapsed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "real", "core"), default="sim")
    parser.add_argument("--sim")
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--tree-dir", type=Path, default=Path(config.DATA_DIR) / "trees")
    parser.add_argument("--msa-dir", type=Path, default=Path(config.REAL_MSA_DIR))
    parser.add_argument("--runs-dir", type=Path, default=Path(config.REAL_CODEML_RUNS_DIR))
    parser.add_argument("--timeout", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timeout is None:
        args.timeout = config.CODEML_TIMEOUT if args.mode == "sim" else config.CODEML_TIMEOUT_REAL
    if args.mode == "sim":
        run_sim(args)
    else:
        if args.mode == "core":
            args.msa_dir = Path(config.REAL_STRIPPED_MSA_DIR)
            args.runs_dir = Path(config.DATA_DIR) / "no_sync" / "codeml_runs_real_stripped"
        run_real(args)


if __name__ == "__main__":
    main()
