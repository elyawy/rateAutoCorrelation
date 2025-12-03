#!/usr/bin/env python3
"""
Submit SLURM jobs to run codeml on each tree in parallel.

Creates one SLURM job per tree, allowing all trees to be processed simultaneously.
"""

import subprocess
import os
from pathlib import Path
import config

def create_slurm_script(job_name: str,
                       account: str,
                       tree_name: str,
                       script_dir: Path,
                       time_limit: str = "03:00:00",
                       memory: str = "4G",
                       num_cores: int = 1,
                       email: str = None,
                       partition: str = "pupko-pool") -> str:
    """
    Creates a SLURM job submission script for a single tree.
    """
    script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={num_cores}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --output=slurm_logs/%x_%j.out",
        f"#SBATCH --error=slurm_logs/%x_%j.err"
    ]
    
    if email:
        script.extend([
            f"#SBATCH --mail-user={email}",
            "#SBATCH --mail-type=END,FAIL"
        ])
    
    script.append("\nmodule purge  # Clear all modules")
    script.append("module load paml-4.10.7")
    
    # Add commands
    script.append("\n# Job commands")
    script.append(f"cd {script_dir}")
    script.append(f"python 2_run_codeml.py --tree {tree_name}")
    
    return "\n".join(script)

def submit_job(script_content: str, script_name: str) -> str:
    """
    Submits a job to the SLURM queue.
    
    Returns:
        Job ID if successful, raises Exception otherwise
    """
    # Write the script to a file
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_name, 0o755)
    
    # Submit the job
    try:
        result = subprocess.run(['sbatch', script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        raise Exception(f"Job submission failed: {e.stderr}")

def main():
    """Submit a SLURM job for each tree."""
    simulated_dir = Path(config.SIMULATED_DATA_DIR)
    
    if not simulated_dir.exists():
        print(f"Error: {simulated_dir}/ does not exist. Run step 1 first.")
        return
    
    # Get all tree directories
    tree_dirs = sorted([d for d in simulated_dir.iterdir() if d.is_dir()])
    
    if not tree_dirs:
        print(f"Error: No tree directories found in {simulated_dir}/")
        return
    
    print(f"Found {len(tree_dirs)} trees to process")
    print("=" * 50)
    
    # Create directories for logs and scripts
    Path("slurm_logs").mkdir(exist_ok=True)
    Path("slurm_scripts").mkdir(exist_ok=True)
    
    # Job submission parameters
    params = {
        "account": "pupko-users",
        "time_limit": "03:00:00",
        "memory": "4G",
        "num_cores": 1,
        "email": None,  # Set to your email if you want notifications
        "partition": "pupko-pool",
        "script_dir": Path.cwd().resolve()
    }
    
    submitted_jobs = []
    
    # Submit each tree as a separate job
    for tree_dir in tree_dirs:
        tree_name = tree_dir.name
        job_name = f"codeml_{tree_name}"
        
        script_content = create_slurm_script(
            job_name=job_name,
            tree_name=tree_name,
            **params
        )
        
        script_name = f"slurm_scripts/{job_name}.sh"
        
        try:
            job_id = submit_job(script_content, script_name)
            submitted_jobs.append((tree_name, job_id))
            print(f"Submitted {tree_name} (Job ID: {job_id})")
        except Exception as e:
            print(f"Failed to submit {tree_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Successfully submitted {len(submitted_jobs)}/{len(tree_dirs)} jobs")
    print("\nTo monitor jobs, use:")
    print("  squeue -u $USER")
    print("\nTo check job status:")
    print("  sacct -j <job_id>")
    print("\nLogs will be in: slurm_logs/")

if __name__ == "__main__":
    main()