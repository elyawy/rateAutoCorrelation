"""Codeml helpers shared across simulated and real-data workflows."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import config


def _absolute_path(path):
    return str(Path(path).expanduser().resolve())


def create_control_file(template_path, run_dir, seq_file, tree_file, true_alpha, *,
                       aa_ratefile=None, model=2, ncatg=8, fix_blength=2,
                       rho_start=0.5, cleandata=0):
    """Create a codeml control file with absolute paths.

    The codeml runner executes with the working directory set to the run directory,
    so the sequence/tree inputs and the LG matrix file must be absolute.
    """
    template = Path(template_path).read_text()
    run_dir = Path(run_dir)
    seq_path = Path(seq_file)
    tree_path = Path(tree_file)

    replacements = {
        "SEQFILE_PLACEHOLDER": str(seq_path if seq_path.is_absolute() else (run_dir / seq_path).resolve()),
        "TREEFILE_PLACEHOLDER": str(tree_path if tree_path.is_absolute() else (run_dir / tree_path).resolve()),
        "WAGFILE_PLACEHOLDER": _absolute_path(aa_ratefile or config.MODELDAT_FILE),
        "ALPHA_PLACEHOLDER": str(true_alpha),
        "MODEL_PLACEHOLDER": str(model),
        "NCATG_PLACEHOLDER": str(ncatg),
        "FIX_BLENGTH_PLACEHOLDER": str(fix_blength),
        "RHO_START_PLACEHOLDER": str(rho_start),
        "CLEANDATA_PLACEHOLDER": str(cleandata),
    }

    for key, value in replacements.items():
        template = template.replace(key, value)

    control_path = run_dir / "control.ctl"
    control_path.write_text(template)
    return control_path


def run_codeml(control_file, run_dir, *, timeout=None):
    """Run codeml with an explicit timeout and return success/elapsed time."""
    run_dir = Path(run_dir)
    control_file = Path(control_file)
    cmd = [os.path.abspath(shutil.which(config.CODEML_EXECUTABLE) or config.CODEML_EXECUTABLE), control_file.name]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=float(timeout if timeout is not None else config.CODEML_TIMEOUT),
        )
        elapsed = time.time() - start
        return result.returncode == 0, elapsed
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, time.time() - start


def parse_codeml_rho(mlb_path):
    """Extract rho from a codeml mlb output file."""
    mlb_path = Path(mlb_path)
    if not mlb_path.exists():
        return None

    match = re.search(r"rho[^=\n]*=\s*([-0-9.eE+]+)", mlb_path.read_text(), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def append_timing_csv(path, sim_name, success, elapsed):
    """Append timing rows without clobbering prior results."""
    timing_path = Path(path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not timing_path.exists() or timing_path.stat().st_size == 0
    with timing_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["simulation", "success", "time_seconds"])
        writer.writerow([sim_name, bool(success), f"{float(elapsed):.3f}"])
    return timing_path
