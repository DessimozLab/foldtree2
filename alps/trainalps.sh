#!/bin/bash
#SBATCH --job-name=ft2-ScaleExp
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=A-prep01
#SBATCH --array=0-12
#SBATCH --output=ft2_scaling_%A_%a.out
#SBATCH --error=ft2_scaling_%A_%a.err
#SBATCH --environment=torch

source ${VENV_PATH}/bin/activate

# Define an array of hidden sizes
hidden_sizes=(100 200 400 800 1600 3200)

# Get the hidden size for this array job
HIDDEN_SIZE=${hidden_sizes[$SLURM_ARRAY_TASK_ID]}

# Create output directory for this specific hidden size
OUTPUT_DIR="/capstor/store/cscs/swissai/prep01/foldtree2/results/hidden_${HIDDEN_SIZE}"
mkdir -p $OUTPUT_DIR

echo "Starting job with hidden size: ${HIDDEN_SIZE}"
echo "Output will be saved to: ${OUTPUT_DIR}"

cd /capstor/store/cscs/swissai/prep01/foldtree2/foldtree2

pip install --no-cache-dir  -e .

# Run experiment with the specified hidden size using learn_monodecoder.py
python /capstor/store/cscs/swissai/prep01/foldtree2/foldtree2/learn_monodecoder.py \
    --dataset /capstor/store/cscs/swissai/prep01/structs_traininffttest.h5 \
    --hidden-size ${HIDDEN_SIZE} \
    --epochs 100 \
    --learning-rate 0.0001 \
    --batch-size 16 \
    --output-dir ${OUTPUT_DIR} \
    --model-name monodecoder_model_${HIDDEN_SIZE} \
    --output-fft \

echo "Job completed for hidden size: ${HIDDEN_SIZE}"