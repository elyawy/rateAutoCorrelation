"""
Evaluate codeml on freshly simulated MSAs.

Mirrors 3_evaluate.py so results are directly comparable.
Runs codeml on 100 MSAs (10 trees x 10 MSAs), fixed to OrthoMaM-like
conditions (150 taxa, 200 sites). Temp files are not persisted.
"""

import math
import pathlib
import re
import subprocess
import tempfile
import random
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from io import StringIO

import config
from utils.simulation import SimulationParams, generate_random_tree, setup_sim, simulate_msa


# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
VALIDATION_SEED = 42
N_TREES = 10
N_MSAS_PER_TREE = 1

SIM_PARAMS = SimulationParams(
    min_taxa=25,
    max_taxa=25,
    min_seq_length=200,
    max_seq_length=200,
)

CODEML_EXECUTABLE = "codeml"
CODEML_TIMEOUT = 3600  # seconds per run


CONTROL_TEMPLATE = """\
      seqfile = {seqfile}
      treefile = {treefile}
      outfile = mlb

      noisy = 0
      verbose = 0
      runmode = 0
      seqtype = 2
      aaRatefile = {wagfile}
      model = 2
      Mgene = 0

   fix_kappa = 1
       kappa = 1

   fix_alpha = 0
       alpha = 0.5
       ncatG = 8

     fix_rho = 0
         rho = 0.1

      Malpha = 0
       clock = 0
       getSE = 0
 RateAncestor = 0

   cleandata = 1
"""


def find_wag_dat():
    """Find wag.dat by running codeml and inspecting its default control file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([CODEML_EXECUTABLE], cwd=tmpdir, capture_output=True)
        ctl = pathlib.Path(tmpdir) / "codeml.ctl"
        if ctl.exists():
            match = re.search(r'aaRatefile\s*=\s*(.+\.dat)', ctl.read_text())
            if match:
                return match.group(1).replace('jones.dat', 'wag.dat').strip()
    return "wag.dat"


def write_phylip(sequences, path):
    """Write a list of sequence strings to a PHYLIP sequential file."""
    n = len(sequences)
    length = len(sequences[0])
    with open(path, 'w') as f:
        f.write(f" {n} {length}\n")
        for i, seq in enumerate(sequences):
            f.write(f"seq_{i:<10} {seq}\n")

def write_newick(tree, path):
    """Write an ete3 tree to a newick file."""
    tree = tree.copy()
    for i, leaf in enumerate(tree.get_leaves()):
        leaf.name = f"seq_{i}"
    with open(path, 'w') as f:
        f.write(tree.write(format=1))

def parse_codeml_output(mlb_path):
    """
    Parse alpha and rho from a codeml mlb output file.
    Returns (alpha, rho) or (None, None) if parsing fails.
    """
    try:
        content = mlb_path.read_text()
        alpha_match = re.search(r'alpha\s*\([^)]*gamma[^)]*\)\s*=\s*([\d.]+)', content, re.IGNORECASE)
        rho_match = re.search(r'rho\s*\(correlation\)\s*=\s*([\d.]+)', content, re.IGNORECASE)
        alpha = float(alpha_match.group(1)) if alpha_match else None
        rho = float(rho_match.group(1)) if rho_match else None
        return alpha, rho
    except Exception:
        return None, None


def run_codeml_on_msa(sequences, tree, wag_dat):
    """
    Run codeml on a single MSA in a temp directory.
    Returns (inferred_alpha, inferred_rho) or (None, None) on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path("debug_codeml")
        tmpdir.mkdir(exist_ok=True)
        seq_file = tmpdir / "seq.phy"
        tree_file = tmpdir / "tree.nwk"
        ctl_file = tmpdir / "control.ctl"

        write_phylip(sequences, seq_file)
        write_newick(tree, tree_file)

        ctl_content = CONTROL_TEMPLATE.format(
            seqfile=seq_file.name,
            treefile=tree_file.name,
            wagfile=wag_dat,
        )
        ctl_file.write_text(ctl_content)

        try:
            result = subprocess.run(
                [CODEML_EXECUTABLE, ctl_file.name],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=CODEML_TIMEOUT,
            )
            print(result.stdout[-500:])  # last 500 chars
            print(result.stderr[-500:])
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, None

        mlb_file = tmpdir / "mlb"
        if not mlb_file.exists():
            return None, None

        return parse_codeml_output(mlb_file)


def save_scatter_plots(df, results, plots_dir):
    """Save true-vs-predicted scatter plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    for param in ['alpha', 'rho']:
        true_col = f'true_{param}'
        pred_col = f'inferred_{param}'

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df[true_col], df[pred_col], alpha=0.5, s=20)

        min_val = min(df[true_col].min(), df[pred_col].min())
        max_val = max(df[true_col].max(), df[pred_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

        r2 = np.corrcoef(df[true_col], df[pred_col])[0, 1] ** 2
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        ax.set_xlabel(f'True {param}', fontsize=12)
        ax.set_ylabel(f'Inferred {param} (codeml)', fontsize=12)
        ax.set_title(
            f'codeml — {param.upper()}\nMSE: {results[f"mse_{param}"]:.6f}',
            fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_file = plots_dir / f"codeml_{param}_scatter.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file}")


def main():
    output_dir = pathlib.Path("results/codeml_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CODEML EVALUATION")
    print("=" * 60)
    print(f"Trees: {N_TREES}, MSAs/tree: {N_MSAS_PER_TREE}, seed: {VALIDATION_SEED}")
    print(f"Taxa: {SIM_PARAMS.min_taxa}, seq length: {SIM_PARAMS.min_seq_length} aa")
    print(f"Total MSAs: {N_TREES * N_MSAS_PER_TREE}")
    print()

    wag_dat = find_wag_dat()
    print(f"WAG dat file: {wag_dat}")
    print()

    random.seed(VALIDATION_SEED)
    np.random.seed(VALIDATION_SEED)

    all_data = []
    total = N_TREES * N_MSAS_PER_TREE
    completed = 0
    failed = 0

    for tree_idx in range(N_TREES):
        n_taxa = SIM_PARAMS.min_taxa
        tree_seed = VALIDATION_SEED + tree_idx * 1000
        random_scale = 10 ** random.uniform(math.log10(config.SCALE_MIN), math.log10(config.SCALE_MAX))

        tree = generate_random_tree(n_taxa, random_scale, tree_seed)
        tree_name = f"tree_{tree_idx:03d}_n{n_taxa}"

        random.seed(tree_seed)
        np.random.seed(tree_seed)
        simulator = setup_sim(tree, sim_seed=tree_seed)

        print(f"Tree {tree_idx + 1}/{N_TREES}: {tree_name}")

        for msa_idx in range(N_MSAS_PER_TREE):
            sequences, true_alpha, true_rho = simulate_msa(simulator, SIM_PARAMS)

            inferred_alpha, inferred_rho = run_codeml_on_msa(sequences, tree, wag_dat)
            completed += 1

            if inferred_alpha is None or inferred_rho is None:
                print(f"  [{completed}/{total}] FAILED (msa {msa_idx})")
                failed += 1
                continue

            print(f"  [{completed}/{total}] alpha: {true_alpha:.3f} -> {inferred_alpha:.3f} | "
                  f"rho: {true_rho:.3f} -> {inferred_rho:.3f}")

            all_data.append({
                'tree': tree_name,
                'msa_idx': msa_idx,
                'true_alpha': true_alpha,
                'true_rho': true_rho,
                'inferred_alpha': inferred_alpha,
                'inferred_rho': inferred_rho,
            })

    print()
    print(f"Completed: {len(all_data)}/{total} successful ({failed} failed)")

    if not all_data:
        print("No results to save.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / "codeml_validation.csv", index=False)

    results = {
        'mse_alpha': float(np.mean((df['true_alpha'] - df['inferred_alpha']) ** 2)),
        'mse_rho':   float(np.mean((df['true_rho']   - df['inferred_rho'])   ** 2)),
    }

    print()
    print("RESULTS")
    print("=" * 60)
    print(f"Alpha — MSE: {results['mse_alpha']:.6f}, "
          f"R²: {np.corrcoef(df['true_alpha'], df['inferred_alpha'])[0,1]**2:.4f}")
    print(f"Rho   — MSE: {results['mse_rho']:.6f}, "
          f"R²: {np.corrcoef(df['true_rho'], df['inferred_rho'])[0,1]**2:.4f}")

    save_scatter_plots(df, results, output_dir / "plots")
    print(f"\nDone — results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
