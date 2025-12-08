#!/bin/bash
#SBATCH --job-name=codeml_array
#SBATCH -A pupko-users_v2
#SBATCH -p pupko-pool
#SBATCH --qos=owner
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module purge
module load paml-4.10.7

SIMDIR="simulated_data"

TREE_DIRS=($(ls -d ${SIMDIR}/*/))
TREE_DIR=${TREE_DIRS[$SLURM_ARRAY_TASK_ID]}
TREE_NAME=$(basename "$TREE_DIR")

echo "Processing tree: $TREE_NAME"

python 2_run_codeml.py --tree "$TREE_NAME"

# to run this script as an array job, use the following command:
# Make sure to create the slurm_logs directory before running the job:
# mkdir -p slurm_logs
# Then submit the job with:
# sbatch --array=0-$(($(ls -d simulated_data/*/ | wc -l) - 1)) run_codeml_array.sh