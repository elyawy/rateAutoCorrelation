"""
Step 2: Run codeml on all simulated MSAs (Multiprocessing & SLURM Optimized).

For each simulation directory in simulated_data/:
  - Create a matching run directory in codeml_runs/
  - Copy the alignment file
  - Create control file pointing to alignment and tree
  - Run codeml
  - Log execution time

Can be run for all sims or a single sim (for SLURM parallelization).
"""

import pathlib
import shutil
import subprocess
import time
import csv
import argparse
import config
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_control_file(template_path, run_dir, seq_file, tree_file):
    """Create a control file with proper paths."""
    with open(template_path, 'r') as f:
        template = f.read()

    control_content = template.replace('SEQFILE_PLACEHOLDER', str(seq_file))
    control_content = control_content.replace('TREEFILE_PLACEHOLDER', str(tree_file))
    control_content = control_content.replace('WAGFILE_PLACEHOLDER', str(config.WAGDAT_FILE))

    control_path = run_dir / 'control.ctl'
    with open(control_path, 'w') as f:
        f.write(control_content)

    return control_path


def run_codeml(control_file, run_dir):
    """Execute codeml in the run directory and return success status and elapsed time."""
    start_time = time.time()

    try:
        result = subprocess.run(
            [config.CODEML_EXECUTABLE, str(control_file.name)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=1800
        )
        elapsed_time = time.time() - start_time
        return result.returncode == 0, elapsed_time

    except subprocess.TimeoutExpired:
        return False, time.time() - start_time

    except FileNotFoundError:
        return False, time.time() - start_time


def process_single_sim(sim_dir, codeml_runs_dir, template_path):
    """
    Worker function to process a single simulation directory.
    Must be at module level for multiprocessing pickling.
    """
    sim_name = sim_dir.name
    alignment_file = sim_dir / "alignment.phy"
    tree_file = sim_dir / "tree.newick"

    if not alignment_file.exists():
        return sim_name, False, 0.0, f"Missing alignment: {alignment_file}"
    if not tree_file.exists():
        return sim_name, False, 0.0, f"Missing tree: {tree_file}"

    # Create run directory
    run_dir = codeml_runs_dir / sim_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy alignment to run directory (codeml needs it local)
    shutil.copy(alignment_file, run_dir / "alignment.phy")

    # Create control file
    control_file = create_control_file(
        template_path,
        run_dir,
        "alignment.phy",          # relative path within run_dir
        tree_file.resolve()       # absolute path to tree
    )

    _, elapsed_time = run_codeml(control_file, run_dir)
    mlb = run_dir / 'mlb'
    success = mlb.exists() and mlb.stat().st_size > 0
    return sim_name, success, elapsed_time, None


def main():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    default_cores = int(slurm_cpus) if slurm_cpus else os.cpu_count()

    parser = argparse.ArgumentParser(description='Run codeml on simulated data')
    parser.add_argument('--sim', type=str, help='Process only this specific sim directory (for SLURM parallelization)')
    parser.add_argument('--cores', type=int, default=default_cores,
                        help=f'Number of cores to use (default: {default_cores} [detected from env])')
    args = parser.parse_args()

    simulated_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    codeml_runs_dir = pathlib.Path(config.CODEML_RUNS_DIR)
    template_path = pathlib.Path(config.CODEML_TEMPLATE)

    if not simulated_dir.exists():
        print(f"Error: {simulated_dir}/ does not exist. Run step 1 first.")
        return

    if not template_path.exists():
        print(f"Error: {template_path} not found.")
        return

    # Get sim directories to process
    if args.sim:
        sim_dirs = [simulated_dir / args.sim]
        if not sim_dirs[0].exists():
            print(f"Error: Sim directory {sim_dirs[0]} does not exist.")
            return
    else:
        sim_dirs = sorted([d for d in simulated_dir.iterdir() if d.is_dir()])

    if not sim_dirs:
        print(f"Error: No sim directories found in {simulated_dir}/")
        return

    print(f"Found {len(sim_dirs)} simulation(s) to process")
    print(f"Using {args.cores} cores")
    print("=" * 50)

    timing_log_path = codeml_runs_dir / 'timing.csv'
    codeml_runs_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_count = len(sim_dirs)

    with open(timing_log_path, 'w', newline='') as timing_file:
        timing_writer = csv.writer(timing_file)
        timing_writer.writerow(['simulation', 'success', 'time_seconds'])

        with ProcessPoolExecutor(max_workers=args.cores) as executor:
            future_to_sim = {
                executor.submit(process_single_sim, sim_dir, codeml_runs_dir, template_path): sim_dir
                for sim_dir in sim_dirs
            }

            completed = 0
            for future in as_completed(future_to_sim):
                try:
                    sim_name, success, elapsed_time, error = future.result()

                    if error:
                        print(f"  WARNING [{sim_name}]: {error}")

                    timing_writer.writerow([sim_name, success, f'{elapsed_time:.3f}'])

                    if success:
                        total_success += 1

                    completed += 1
                    if completed % 10 == 0 or completed == total_count:
                        print(f"  Completed {completed}/{total_count} runs")

                except Exception as e:
                    print(f"  ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"All runs complete: {total_success}/{total_count} successful")
    print(f"Timing log: {timing_log_path}")
    print(f"Output directory: {codeml_runs_dir}/")


if __name__ == "__main__":
    main()