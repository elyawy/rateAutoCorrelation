"""
Step 2: Extract inferred alpha from raxml-ng output files.

Parses raxml.raxml.bestModel for each run and saves results to
results/inferred_parameters.csv.

bestModel line looks like:
  WAG+G4{0.531247}
"""

import pathlib
import re
import csv

import config


def parse_alpha(best_model_file):
    """Extract alpha from a raxml.raxml.bestModel file."""
    try:
        content = best_model_file.read_text().strip()
        # e.g. WAG+G4{0.531247}
        match = re.search(r'\{([0-9.]+)\}', content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"  Error reading {best_model_file}: {e}")
    return None


def main():
    raxml_runs_dir = pathlib.Path(config.RAXML_RUNS_DIR)
    results_dir = pathlib.Path(config.RESULTS_DIR)

    if not raxml_runs_dir.exists():
        print(f"Error: {raxml_runs_dir}/ does not exist. Run step 1 first.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "inferred_parameters.csv"

    rows = []
    sim_dirs = sorted([d for d in raxml_runs_dir.iterdir() if d.is_dir()])

    print(f"Extracting parameters from {len(sim_dirs)} run directories...")

    for run_dir in sim_dirs:
        sim_name = run_dir.name
        best_model_file = run_dir / "raxml.raxml.bestModel"

        if not best_model_file.exists():
            print(f"  WARNING: No bestModel file for {sim_name}")
            continue

        alpha = parse_alpha(best_model_file)
        if alpha is None:
            print(f"  WARNING: Could not parse alpha for {sim_name}")
            continue

        rows.append({"sim_name": sim_name, "inferred_alpha": alpha})

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sim_name", "inferred_alpha"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtracted {len(rows)} results → {output_file}")


if __name__ == "__main__":
    main()
