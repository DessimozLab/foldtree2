#!/bin/bash
#SBATCH --job-name=ft2-geom-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --environment=pygmk3


set -euo pipefail

# Load modules if your cluster requires them.
module load cuda/11.7 || true
module load python/3.8 || true

# Optional: export VENV_PATH before submitting.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p logs

PROJECT_ROOT=${PROJECT_ROOT:-/capstor/store/cscs/swissai/prep01/foldtree2/foldtree2}
DATASET=${DATASET:-/capstor/store/cscs/swissai/prep01/structs_training_mk2.h5}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/capstor/store/cscs/swissai/prep01/foldtree2/models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/prep01/foldtree2/results/geometry/gh200_${SLURM_JOB_ID}}
FAPE_PAIR_SAMPLE_SIZE=${FAPE_PAIR_SAMPLE_SIZE:-1024}
mkdir -p "${CHECKPOINT_DIR}"

TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE:-64}
TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-192}
TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-4}
TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-8}
SE3_HIDDEN=${SE3_HIDDEN:-32}
SE3_DEPTH=${SE3_DEPTH:-4}
SE3_HEADS=${SE3_HEADS:-8}
SE3_DIM_HEAD=${SE3_DIM_HEAD:-16}

cd "${PROJECT_ROOT}"
pip install --no-cache-dir -e .

CMD=(
  python learn_geometry_lightning.py
  --dataset "${DATASET}"
  --epochs "${EPOCHS:-100}"
  --batch-size "${BATCH_SIZE:-4}"
  --target-effective-batch-size "${TARGET_EFFECTIVE_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE:-5e-4}"
  --num-workers "${NUM_WORKERS:-0}"
  --accelerator "${ACCELERATOR:-cuda}"
  --devices "${DEVICES:-1}"
  --strategy "${STRATEGY:-ddp_find_unused_parameters_true}"
  --precision "${PRECISION:-32-true}"
  --pretrained-encoder-path "${PRETRAINED_ENCODER}"
  --pretrained-encoder-full-path "${PRETRAINED_ENCODER}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --save-top-k "${SAVE_TOP_K:--1}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-10}"
  --transformer-width "${TRANSFORMER_WIDTH}"
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-nheads "${TRANSFORMER_NHEADS}"
  --rt-hidden "${RT_HIDDEN:-128,64,32}"
  --rotation-hidden "${ROTATION_HIDDEN:-128,64,32}"
  --ca-step-hidden "${CA_STEP_HIDDEN:-128,64,32}"
  --angles-hidden "${ANGLES_HIDDEN:-128,64,32}"
  --ss-hidden "${SS_HIDDEN:-64,32,16}"
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
    --se3-hidden "${SE3_HIDDEN}"
    --se3-depth "${SE3_DEPTH}"
    --se3-heads "${SE3_HEADS}"
    --se3-dim-head "${SE3_DIM_HEAD}"
  )

  if [[ "${USE_SE3_ATOM_REFINE}" == "1" ]]; then
    CMD+=(
      --use-se3-atom-refine
      --se3-atom-weight "${SE3_ATOM_WEIGHT:-0.05}"
      --se3-atom-fape-weight "${SE3_ATOM_FAPE_WEIGHT:-0.25}"
      --fape-pair-sample-size "${FAPE_PAIR_SAMPLE_SIZE}"
    )
  fi
else
  CMD+=(--no-use-se3)
fi

if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi

echo "Launching GH200 single-node data-parallel geometry training"
echo "GH200 setup: nodes=1, tasks_per_node=4, gpus_per_task=1"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"

echo "Completed GH200 geometry training"