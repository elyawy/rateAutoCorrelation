"""Paired no-indel experiment: does ignoring indels bias codeml's rho?

For each sampled rho, simulate three MSAs with the same tree, alpha and rho:
  run1  no indels, root length 500
  run2  with indels (indel-aware gamma rates)
  run3  run2 without gaps (same length, same rate categories, residues unmasked)
Run 2 is run 3 with the skeleton's gap pattern masked on top.
codeml (alpha and branches fixed, rho free) is run on each; results go to one CSV.
"""
import argparse
import csv
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from msasim import (ALPHABET_CODES, MODEL_CODES, ReplacementModelSpec,
                    SimProtocol, SiteRateModelSpec, Simulator)
from msasim.constants import SITE_RATE_MODELS
from msasim.distributions import ZipfDistribution

ROOT_LEN = 500
GAMMA_CATEGORIES = 8
INS_RATE, INS_ZIPF = 0.0051, 2.15
DEL_RATE, DEL_ZIPF = 0.017, 1.52
ZIPF_TRUNCATION = 150
INDEL_SCALE = 1.0  # multiplies both indel rates, set by --indel-scale


def make_simulator(tree, rho, alpha, seed, indels):
    protocol = SimProtocol(
        tree,
        root_seq_size=ROOT_LEN,
        insertion_rate=INS_RATE * INDEL_SCALE if indels else 0.0,
        deletion_rate=DEL_RATE * INDEL_SCALE if indels else 0.0,
        insertion_dist=ZipfDistribution(INS_ZIPF, ZIPF_TRUNCATION),
        deletion_dist=ZipfDistribution(DEL_ZIPF, ZIPF_TRUNCATION),
        seed=seed,
    )
    rate_model = SiteRateModelSpec(
        gamma_alpha=alpha,
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
    model = ReplacementModelSpec(
        model=MODEL_CODES.LG, alphabet=ALPHABET_CODES.PROTEIN, site_rate_model=rate_model
    )
    return Simulator(protocol, model)


def msa_rows(msa):
    """[(name, sequence)] from a filled Msa."""
    rows = []
    for i in range(msa.get_num_sequences()):
        header, seq = msa.get_msa_row(i).split("\n", 1)
        rows.append((header.lstrip(">").strip(), seq.replace("\n", "")))
    return rows


def sparse_to_residue_mask(sparse, length):
    """Sparse blocks (positive = residues, negative = gap run) -> True where residue."""
    mask = []
    for blocks in sparse:
        row = []
        for b in blocks:
            row += [True] * b if b > 0 else [False] * (-b)
        assert len(row) == length
        mask.append(row)
    return mask


def simulate_triplet(tree, rho, alpha, seed):
    """Return (run1_rows, run2_rows, run3_rows, gap_features)."""
    # run 1: no indels
    run1 = msa_rows(make_simulator(tree, rho, alpha, seed, indels=False)())

    # indel skeleton: gap pattern + indel-aware rate categories, no residues yet
    sim = make_simulator(tree, rho, alpha, seed, indels=True)
    skeleton = sim._build_msa_with_indels()
    length = skeleton.get_length()
    cats = skeleton.get_per_site_rate_categories()
    root_pos = skeleton.get_root_positions_in_msa()
    mask = sparse_to_residue_mask(skeleton.get_sparse_msa(), length)

    # run 3: residues for every column, no gap map given to the substitution sim
    names = [n for n, _ in run1]
    subs = sim._substitution_simulator
    subs.set_per_site_rate_categories(cats)
    full = subs.simulate_substitutions(length, "", root_pos)
    run3 = [(names[i], full[i]) for i in range(len(names))]

    # run 2: run 3 with the skeleton's gaps masked on top
    run2 = [
        (n, "".join(c if mask[i][k] else "-" for k, c in enumerate(seq)))
        for i, (n, seq) in enumerate(run3)
    ]

    n_seq = len(mask)
    col_gap_frac = [1 - sum(mask[i][k] for i in range(n_seq)) / n_seq for k in range(length)]
    gaps = {
        "n_columns": length,
        "gap_fraction": float(np.mean(col_gap_frac)),
        "frac_cols_over90_gap": float(np.mean([g > 0.9 for g in col_gap_frac])),
    }
    return run1, run2, run3, gaps


def write_phylip(rows, path):
    with open(path, "w") as f:
        f.write(f" {len(rows)} {len(rows[0][1])}\n")
        for name, seq in rows:
            f.write(f"{name}  {seq}\n")


def run_codeml(codeml, template, lg_dat, tree_file, rows, alpha, timeout):
    """Run codeml in a temp dir (deleted afterwards); return (rho estimate, seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_phylip(rows, run_dir / "aln.phy")
        (run_dir / "tree.nwk").write_text(Path(tree_file).read_text())
        ctl = (
            Path(template).read_text()
            .replace("SEQFILE_PLACEHOLDER", "aln.phy")
            .replace("TREEFILE_PLACEHOLDER", "tree.nwk")
            .replace("WAGFILE_PLACEHOLDER", str(Path(lg_dat).resolve()))
            .replace("ALPHA_PLACEHOLDER", str(alpha))
        )
        (run_dir / "codeml.ctl").write_text(ctl)
        t0 = time.time()
        subprocess.run([str(Path(codeml).resolve()), "codeml.ctl"], cwd=run_dir,
                       capture_output=True, timeout=timeout)
        elapsed = time.time() - t0
        mlb = run_dir / "mlb"
        m = re.search(r"rho \(correlation\)\s*=\s*([-\d.eE+]+)", mlb.read_text()) if mlb.exists() else None
        return (float(m.group(1)) if m else float("nan")), elapsed


def main():
    ap = argparse.ArgumentParser()
    # defaults are relative to the repo root (run the script from there)
    ap.add_argument("--tree", default="data/reference_tree.nwk")
    ap.add_argument("--codeml", default="bins/paml-4.10.10-linux-x86_64/bin/codeml")
    ap.add_argument("--lg-dat", default="bins/paml-4.10.10-linux-x86_64/dat/lg.dat")
    ap.add_argument("--template", default="benchmarks/codeml_benchmark/codeml_template.ctl")
    ap.add_argument("--out-dir", default="data/no_indel_experiment")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--indel-scale", type=float, default=5.0)
    args = ap.parse_args()
    global INDEL_SCALE
    INDEL_SCALE = args.indel_scale

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tree_str = Path(args.tree).read_text().strip()
    rng = np.random.default_rng(args.seed)
    rhos = rng.uniform(0.01, 0.99, args.n)

    rows = []
    for k, rho in enumerate(rhos):
        sim_seed = int(rng.integers(1, 2**31 - 1))
        run1, run2, run3, gaps = simulate_triplet(tree_str, float(rho), args.alpha, sim_seed)
        row = {"sample": k, "seed": sim_seed, "true_rho": float(rho), "alpha": args.alpha, **gaps}
        for label, aln in (("no_indel", run1), ("indel", run2), ("indel_gapless", run3)):
            est, secs = run_codeml(args.codeml, args.template, args.lg_dat, args.tree, aln,
                                   args.alpha, args.timeout)
            row[f"est_{label}"] = est
            row[f"secs_{label}"] = round(secs, 1)
        rows.append(row)
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()