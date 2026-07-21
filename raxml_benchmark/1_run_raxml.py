"""
Step 1: Run raxml-ng on all simulated MSAs.

For each simulation directory in simulated_data/:
  - Run raxml-ng in --evaluate mode (fixed tree, optimize branch lengths + alpha)
  - Log execution time to raxml_runs/timing.csv
"""

import pathlib
import subprocess
import time
import csv
import argparse
import config


def run_raxml(sim_dir, raxml_runs_dir):
    """Run raxml-ng --evaluate on a single simulation. Returns (sim_name, success, elapsed)."""
    sim_name = sim_dir.name
    alignment_file = sim_dir / "alignment.phy"
    tree_file = sim_dir / "tree.newick"

    if not alignment_file.exists():
        return sim_name, False, 0.0, f"Missing alignment: {alignment_file}"
    if not tree_file.exists():
        return sim_name, False, 0.0, f"Missing tree: {tree_file}"

    run_dir = raxml_runs_dir / sim_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.RAXML_EXECUTABLE,
        "--evaluate",
        "--msa", str(alignment_file.resolve()),
        "--tree", str(tree_file.resolve()),
        "--model", config.RAXML_MODEL,
        "--prefix", str((run_dir / "raxml").resolve()),
        "--redo",
    ]

    start_time = time.time()
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        elapsed = time.time() - start_time
    except subprocess.TimeoutExpired:
        return sim_name, False, time.time() - start_time, "Timeout"
    except FileNotFoundError:
        return sim_name, False, 0.0, f"Executable not found: {config.RAXML_EXECUTABLE}"

    # Success = bestModel file exists and is non-empty
    best_model_file = run_dir / "raxml.raxml.bestModel"
    success = best_model_file.exists() and best_model_file.stat().st_size > 0
    return sim_name, success, elapsed, None


def main():
    parser = argparse.ArgumentParser(description="Run raxml-ng on simulated data")
    parser.add_argument("--sim", type=str, help="Process only this specific sim directory")
    args = parser.parse_args()

    simulated_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    raxml_runs_dir = pathlib.Path(config.RAXML_RUNS_DIR)

    if not simulated_dir.exists():
        print(f"Error: {simulated_dir} does not exist.")
        return

    if args.sim:
        sim_dirs = [simulated_dir / args.sim]
        if not sim_dirs[0].exists():
            print(f"Error: {sim_dirs[0]} does not exist.")
            return
    else:
        sim_dirs = sorted([d for d in simulated_dir.iterdir() if d.is_dir()])

    if not sim_dirs:
        print(f"Error: No sim directories found in {simulated_dir}/")
        return

    raxml_runs_dir.mkdir(parents=True, exist_ok=True)
    timing_log = raxml_runs_dir / "timing.csv"

    print(f"Found {len(sim_dirs)} simulation(s)")
    print("=" * 50)

    total_success = 0
    total_count = len(sim_dirs)

    with open(timing_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation", "success", "time_seconds"])

        for completed, sim_dir in enumerate(sim_dirs, 1):
            sim_name, success, elapsed, error = run_raxml(sim_dir, raxml_runs_dir)
            if error:
                print(f"  WARNING [{sim_name}]: {error}")
            writer.writerow([sim_name, success, f"{elapsed:.3f}"])
            f.flush()
            if success:
                total_success += 1
            if completed % 10 == 0 or completed == total_count:
                print(f"  Completed {completed}/{total_count}")

    print("\n" + "=" * 50)
    print(f"Done: {total_success}/{total_count} successful")
    print(f"Timing log: {timing_log}")
    print(f"Output directory: {raxml_runs_dir}/")


if __name__ == "__main__":
    main()