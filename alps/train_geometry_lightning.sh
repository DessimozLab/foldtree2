#!/bin/bash
#SBATCH --job-name=ft2-geom-lightning
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=a0117
#SBATCH --array=0-5
#SBATCH --output=ft2_geom_%A_%a.out
#SBATCH --error=ft2_geom_%A_%a.err
#SBATCH --environment=pygmk3

set -euo pipefail

# Optional: export VENV_PATH before submitting.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

# Sweep transformer width like the original hidden-size sweep.
transformer_widths=(64 96 128 192 256 384)
TRANSFORMER_WIDTH=${transformer_widths[$SLURM_ARRAY_TASK_ID]}

PROJECT_ROOT=${PROJECT_ROOT:-/users/dmoi/foldtree2/}
DATASET=${DATASET:-/capstor/store/cscs/swissai/a0117/structalnfinal.h5}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/users/dmoi/foldtree2/foldtree2/models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt}

RUN_TAG="tw${TRANSFORMER_WIDTH}_bs2_lr5e4"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/a0117/chkpts/results/geometry/${RUN_TAG}}
mkdir -p "${CHECKPOINT_DIR}"

cd "${PROJECT_ROOT}"
pip install --no-cache-dir -e .

echo "Starting geometry Lightning run"
echo "  transformer_width=${TRANSFORMER_WIDTH}"
echo "  dataset=${DATASET}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"

CMD=(
  python learn_geometry_lightning.py
  --dataset "${DATASET}"
  --epochs "${EPOCHS:-100}"
  --batch-size "${BATCH_SIZE:-2}"
  --target-effective-batch-size "${TARGET_EFFECTIVE_BATCH_SIZE:-10}"
  --learning-rate "${LEARNING_RATE:-5e-4}"
  --num-workers "${NUM_WORKERS:-0}"
  --accelerator "${ACCELERATOR:-cuda}"
  --devices "${DEVICES:-1}"
  --precision "${PRECISION:-32-true}"
  --pretrained-encoder-path "${PRETRAINED_ENCODER}"
  --pretrained-encoder-full-path "${PRETRAINED_ENCODER}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --save-top-k "${SAVE_TOP_K:--1}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-10}"
  --transformer-width "${TRANSFORMER_WIDTH}"
  --transformer-layers "${TRANSFORMER_LAYERS:-2}"
  --transformer-nheads "${TRANSFORMER_NHEADS:-4}"
  --rt-hidden "${RT_HIDDEN:-64,32,16}"
  --rotation-hidden "${ROTATION_HIDDEN:-64,32,16}"
  --ca-step-hidden "${CA_STEP_HIDDEN:-64,32,16}"
  --angles-hidden "${ANGLES_HIDDEN:-64,32,16}"
  --ss-hidden "${SS_HIDDEN:-32,16,8}"
  --use-coarse-ca-loss
  --use-coarse-backbone-loss
  --coarse-ca-weight "${COARSE_CA_WEIGHT:-1.0}"
  --coarse-ca-step-frame "${COARSE_CA_STEP_FRAME:-prev}"
  --rotation-target-frame "${ROTATION_TARGET_FRAME:-local}"
  --coarse-backbone-angle-weight "${COARSE_BACKBONE_ANGLE_WEIGHT:-0.05}"
)

USE_SE3=${USE_SE3:-1}
USE_SE3_ATOM_REFINE=${USE_SE3_ATOM_REFINE:-1}

if [[ "${USE_SE3}" == "1" ]]; then
  CMD+=(
    --use-se3
    --se3-input-source "${SE3_INPUT_SOURCE:-coarse_ca}"
    --se3-hidden "${SE3_HIDDEN:-16}"
    --se3-depth "${SE3_DEPTH:-2}"
    --se3-heads "${SE3_HEADS:-4}"
    --se3-dim-head "${SE3_DIM_HEAD:-8}"
  )

  if [[ "${USE_SE3_ATOM_REFINE}" == "1" ]]; then
    CMD+=(
      --use-se3-atom-refine
      --se3-atom-weight "${SE3_ATOM_WEIGHT:-0.05}"
      --se3-atom-fape-weight "${SE3_ATOM_FAPE_WEIGHT:-0.25}"
    )
  fi
else
  CMD+=(--no-use-se3)
fi

if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Completed geometry Lightning run: ${RUN_TAG}"
