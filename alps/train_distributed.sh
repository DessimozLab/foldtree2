#!/bin/bash
#SBATCH --job-name=pl_ddp_multi_node    # Job name
#SBATCH --nodes=2                       # Request 2 nodes
#SBATCH --gres=gpu:2                    # Request 2 GPUs per node
#SBATCH --time=02:00:00                 # Set a time limit
#SBATCH --output=logs/%x_%j.out          # Standard output log
#SBATCH --error=logs/%x_%j.err           # Standard error log

# Load modules (adjust as necessary)
module load cuda/11.7
module load python/3.8

# Activate your environment
source ~/envs/myenv/bin/activate

# Launch the training script using srun; SLURM handles process distribution
srun python train.py
