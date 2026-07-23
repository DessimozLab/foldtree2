#!/bin/bash
#SBATCH --job-name=ft2-se3-dot-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Dedicated Alps launcher for the frozen-encoder geometry-dot-contact SE3 run.
# Override any setting with an environment variable before submitting with sbatch.

module load cuda/11.7 || true
module load python/3.8 || true

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p logs

PROJECT_ROOT=${PROJECT_ROOT:-/capstor/store/cscs/swissai/prep01/foldtree2/foldtree2}
DATASET=${DATASET:-/capstor/store/cscs/swissai/prep01/structs_training_mk2.h5}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/capstor/store/cscs/swissai/prep01/foldtree2/models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/prep01/foldtree2/results/geometry/se3_dot_large_${SLURM_JOB_ID}}

# The decoder input must remain three-dimensional for the dot-product contact sketch.
TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-3}
TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-1}
TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-1}

# Larger SE3 capacity than the local validation run.
SE3_HIDDEN=${SE3_HIDDEN:-64}
SE3_DEPTH=${SE3_DEPTH:-6}
SE3_HEADS=${SE3_HEADS:-8}
SE3_DIM_HEAD=${SE3_DIM_HEAD:-16}

TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SE3_MAX_NODES=${SE3_MAX_NODES:-384}

cd "${PROJECT_ROOT}"
pip install --no-cache-dir -e .
mkdir -p "${CHECKPOINT_DIR}"

CMD=(
  python learn_geometry_lightning.py
  --dataset "${DATASET}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --target-effective-batch-size "${TARGET_EFFECTIVE_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --num-workers "${NUM_WORKERS:-0}"
  --accelerator "${ACCELERATOR:-cuda}"
  --devices "${DEVICES:-1}"
  --strategy "${STRATEGY:-ddp_find_unused_parameters_true}"
  --precision "${PRECISION:-32-true}"
  --pretrained-encoder-path "${PRETRAINED_ENCODER}"
  --pretrained-encoder-full-path "${PRETRAINED_ENCODER}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --save-top-k "${SAVE_TOP_K:-3}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-16}"
  --transformer-width "${TRANSFORMER_WIDTH}"
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-nheads "${TRANSFORMER_NHEADS}"
  --use-se3
  --train-se3-only
  --se3-input-source geometry_dot_contacts
  --se3-hidden "${SE3_HIDDEN}"
  --se3-depth "${SE3_DEPTH}"
  --se3-heads "${SE3_HEADS}"
  --se3-dim-head "${SE3_DIM_HEAD}"
  --se3-contact-coord-scale "${SE3_CONTACT_COORD_SCALE:-20}"
  --se3-contact-local-window "${SE3_CONTACT_LOCAL_WINDOW:-1}"
  --se3-contact-sketch-top-k "${SE3_CONTACT_SKETCH_TOP_K:-8}"
  --se3-contact-sketch-threshold "${SE3_CONTACT_SKETCH_THRESHOLD:-0.0}"
  --se3-contact-sketch-min-seq-sep "${SE3_CONTACT_SKETCH_MIN_SEQ_SEP:-3}"
  --se3-max-nodes "${SE3_MAX_NODES}"
  --no-use-frame-fape-loss
  --no-use-quat-geodesic-loss
  --no-use-decoder-angle-loss
  --no-use-coarse-ca-loss
  --no-use-coarse-backbone-loss
  --no-use-se3-atom-refine
  --use-se3-angle-loss
  --no-use-uncertainty-weighting
  --sanitize-nonfinite-grads
  --skip-empty-loss-batches
)

# Residue frame losses are optional because the first scale-up should remain
# directly comparable to the validated angle-only experiment.
if [[ "${USE_SE3_RESIDUE_LOSS:-0}" == "1" ]]; then
  CMD+=(--use-se3-residue-loss)
else
  CMD+=(--no-use-se3-residue-loss)
fi

if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  CMD+=(--limit-val-batches "${LIMIT_VAL_BATCHES}")
fi

echo "Launching frozen-encoder geometry-dot-contact SE3 training"
echo "SE3 setup: hidden=${SE3_HIDDEN}, depth=${SE3_DEPTH}, heads=${SE3_HEADS}, dim_head=${SE3_DIM_HEAD}"
echo "Input setup: transformer_width=${TRANSFORMER_WIDTH}, max_nodes=${SE3_MAX_NODES}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"

echo "Completed frozen-encoder geometry-dot-contact SE3 training"
