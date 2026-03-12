#!/bin/bash
#SBATCH --job-name=train_neural_net
#SBATCH -A pupko-users_v2
#SBATCH -p gpu-pupko-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:A100:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

# Load modules
module purge
module load python/python-anaconda_3.7
# module load cuda/11.8  # Add if needed - check with: module avail cuda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create log directory if needed
mkdir -p slurm_logs

# Run training
cd inference_pipeline
nvidia-smi
python 2_train_models.py