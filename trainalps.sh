#!/bin/bash
#SBATCH --job-name=ft2-ScaleExp
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=A-prep01
#SBATCH --array=0-12
#SBATCH --mem=32G
#SBATCH --output=ft2_scaling_%A_%a.out
#SBATCH --error=ft2_scaling_%A_%a.err

source /users/dmoi/miniforge3/etc/profile.d/conda.sh
source /users/dmoi/miniforge3/etc/profile.d/mamba.sh

mamba activate ft2

# Define an array of hidden sizes
hidden_sizes=(100 300 500 700 900 1100 1300 1500 1700 1900 2100 2300 2500)

# Get the hidden size for this array job
HIDDEN_SIZE=${hidden_sizes[$SLURM_ARRAY_TASK_ID]}

# Create output directory for this specific hidden size
OUTPUT_DIR="/capstor/store/cscs/swissai/prep01/foldtree2/results/hidden_${HIDDEN_SIZE}"
mkdir -p $OUTPUT_DIR

echo "Starting job with hidden size: ${HIDDEN_SIZE}"
echo "Output will be saved to: ${OUTPUT_DIR}"

# Activate virtual environment
source /capstor/store/cscs/swissai/prep01/venv/bin/activate

# Run experiment with the specified hidden size
python /capstor/store/cscs/swissai/prep01/foldtree2/foldtree2/scaling_experiment.py \
    --dataset /capstor/scratch/cscs/dmoi/structs_trainingalpstest.h5 \
    --hidden-size ${HIDDEN_SIZE} \
    --dataset-fractions 0.5 1.0 \
    --epochs 30 \
    --learning-rate 0.0001 \
    --batch-size 16 \
    --output-dir ${OUTPUT_DIR}

echo "Job completed for hidden size: ${HIDDEN_SIZE}"