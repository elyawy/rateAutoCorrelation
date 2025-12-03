"""
Step 2: Run codeml on all simulated MSAs.

For each tree and each simulation:
  - Create a run directory in codeml_runs/
  - Copy the alignment file
  - Create control file with correct paths
  - Run codeml
  - Log execution time
"""

import pathlib
import shutil
import subprocess
import time
import csv
import config



def create_control_file(template_path, run_dir, seq_file, tree_file):
    """Create a control file with proper paths."""
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    control_content = template.replace('SEQFILE_PLACEHOLDER', str(seq_file))
    control_content = control_content.replace('TREEFILE_PLACEHOLDER', str(tree_file))
    control_content = control_content.replace('WAGFILE_PLACEHOLDER', str(config.WAGDAT_FILE))


    # Write to run directory
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
            timeout=300  # 5 minute timeout per run
        )
        elapsed_time = time.time() - start_time
        return result.returncode == 0, elapsed_time
    
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"    WARNING: Timeout exceeded")
        return False, elapsed_time
    
    except FileNotFoundError:
        elapsed_time = time.time() - start_time
        print(f"    ERROR: {config.CODEML_EXECUTABLE} not found. Is it in your PATH?")
        return False, elapsed_time

def process_tree(tree_name, simulated_dir, trees_dir, codeml_runs_dir, template_path):
    """Process all simulations for a single tree."""
    tree_sim_dir = simulated_dir / tree_name
    tree_file = trees_dir / f"{tree_name}.newick"
    
    if not tree_file.exists():
        print(f"  WARNING: Tree file not found: {tree_file}")
        return 0, 0
    
    # Find all simulation files
    sim_files = sorted(tree_sim_dir.glob("sim_*.phy"))
    
    if not sim_files:
        print(f"  WARNING: No simulation files found in {tree_sim_dir}")
        return 0, 0
    
    success_count = 0
    total_count = len(sim_files)
    
    # Prepare timing log
    timing_log_path = codeml_runs_dir / tree_name / 'timing.csv'
    timing_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(timing_log_path, 'w', newline='') as timing_file:
        timing_writer = csv.writer(timing_file)
        timing_writer.writerow(['simulation', 'success', 'time_seconds'])
        
        for sim_file in sim_files:
            sim_name = sim_file.stem  # e.g., sim_001_a0.5_r0.3
            
            # Create run directory
            run_dir = codeml_runs_dir / tree_name / sim_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy alignment file to run directory
            shutil.copy(sim_file, run_dir / sim_file.name)
            
            # Create control file
            control_file = create_control_file(
                template_path,
                run_dir,
                sim_file.name,  # relative path within run_dir
                tree_file.resolve()  # absolute path to tree
            )
            
            # Run codeml and get timing
            success, elapsed_time = run_codeml(control_file, run_dir)
            
            # Log timing
            timing_writer.writerow([sim_name, success, f'{elapsed_time:.3f}'])
            
            if success:
                success_count += 1
            
            # Progress indicator
            if (sim_files.index(sim_file) + 1) % 10 == 0:
                print(f"    Completed {sim_files.index(sim_file) + 1}/{total_count} runs")
    
    print(f"  Timing log saved to: {timing_log_path}")
    return success_count, total_count

def main():
    """Main function to run codeml on all simulations."""
    simulated_dir = pathlib.Path(config.SIMULATED_DATA_DIR)
    trees_dir = pathlib.Path(config.TREES_DIR)
    codeml_runs_dir = pathlib.Path(config.CODEML_RUNS_DIR)
    template_path = pathlib.Path(config.CODEML_TEMPLATE)
    
    if not simulated_dir.exists():
        print(f"Error: {simulated_dir}/ does not exist. Run step 1 first.")
        return
    
    if not template_path.exists():
        print(f"Error: {template_path} not found.")
        return
    
    # Get all tree directories
    tree_dirs = sorted([d for d in simulated_dir.iterdir() if d.is_dir()])
    
    if not tree_dirs:
        print(f"Error: No tree directories found in {simulated_dir}/")
        return
    
    print(f"Found {len(tree_dirs)} tree(s) to process")
    print("=" * 50)
    
    total_success = 0
    total_runs = 0
    
    # Process each tree
    for tree_dir in tree_dirs:
        tree_name = tree_dir.name
        print(f"\nProcessing tree: {tree_name}")
        
        success, total = process_tree(
            tree_name,
            simulated_dir,
            trees_dir,
            codeml_runs_dir,
            template_path
        )
        
        total_success += success
        total_runs += total
        
        print(f"  Completed: {success}/{total} successful runs")
    
    print("\n" + "=" * 50)
    print(f"All runs complete: {total_success}/{total_runs} successful")
    print(f"Output directory: {codeml_runs_dir}/")

if __name__ == "__main__":
    main()