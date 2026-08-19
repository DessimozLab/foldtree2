#!/bin/bash
#SBATCH --job-name=ft2-prod-staged-gh200
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=a0117
#SBATCH --output=ft2_prod_staged_%j.out
#SBATCH --error=ft2_prod_staged_%j.err
#SBATCH --environment=pygmk3

set -euo pipefail

SCRIPT_START_EPOCH=$(date +%s)
SCRIPT_NAME=$(basename "$0")

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*"
}

print_kv() {
  printf '  %-36s %s\n' "$1" "$2"
}

on_exit() {
  local exit_code=$?
  local elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
  if [[ ${exit_code} -eq 0 ]]; then
    log "Script finished successfully in ${elapsed}s"
  else
    log "Script failed with exit code ${exit_code} after ${elapsed}s"
  fi
}
trap on_exit EXIT

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR=${LOG_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${SCRIPT_NAME%.sh}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "${LOG_FILE}") 2>&1

log "Logging initialized"
print_kv "log_file" "${LOG_FILE}"
print_kv "script" "${0}"
print_kv "host" "$(hostname)"

if [[ -n "${VENV_PATH:-}" ]]; then
  log "Activating virtual environment from VENV_PATH=${VENV_PATH}"
  source "${VENV_PATH}/bin/activate"
fi

PROJECT_ROOT=${PROJECT_ROOT:-/users/dmoi/foldtree2/}
log "Installing editable package from PROJECT_ROOT=${PROJECT_ROOT}"
pip install --no-cache-dir --no-deps -e "${PROJECT_ROOT}"

DATASET=${DATASET:-/capstor/store/cscs/swissai/a0117/structalnfinal.h5}
MODEL_TAG=${MODEL_TAG:-40char_staged_transformer}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-${PROJECT_ROOT}/models/production/40char_minimal_decoder/final_40char_mk2_contactsfix_aa_encoder_full_epoch_41.pt}
PRETRAINED_GEOMETRY_DECODER=${PRETRAINED_GEOMETRY_DECODER:-${PROJECT_ROOT}/models/production/40char_minimal_decoder/final_40char_mk2_contactsfix_aa_decoder_full_epoch_41.pt}

BATCH_SIZE=${BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE:-16}
TRAIN_PRECISION=${PRECISION:-32-true}
DEVICES=${DEVICES:-1}
STRATEGY=${STRATEGY:-auto}

STAGED_HIDDEN=${STAGED_HIDDEN:-128}
STAGED_HEADS=${STAGED_HEADS:-4}
STAGED_LAYERS=${STAGED_LAYERS:-2}
STAGED_DROPOUT=${STAGED_DROPOUT:-0.05}
STAGED_MAX_STEP=${STAGED_MAX_STEP:-4.0}
STAGED_MAX_REFINE_DELTA=${STAGED_MAX_REFINE_DELTA:-2.0}
STAGE_LOSS_WEIGHTS=${STAGE_LOSS_WEIGHTS:-0.25,0.5,1.0}

RUN_TAG="prod_staged_${MODEL_TAG}_bs${BATCH_SIZE}_eff${TARGET_EFFECTIVE_BATCH_SIZE}"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/a0117/chkpts/results/geometry/${RUN_TAG}}
mkdir -p "${CHECKPOINT_DIR}"

cd "${PROJECT_ROOT}"

log "Starting production staged transformer geometry run"
print_kv "dataset" "${DATASET}"
print_kv "checkpoint_dir" "${CHECKPOINT_DIR}"
print_kv "batch_size" "${BATCH_SIZE}"
print_kv "val_batch_size" "${VAL_BATCH_SIZE}"
print_kv "target_effective_batch_size" "${TARGET_EFFECTIVE_BATCH_SIZE}"
print_kv "devices" "${DEVICES}"
print_kv "strategy" "${STRATEGY}"
print_kv "precision" "${TRAIN_PRECISION}"
print_kv "staged_hidden" "${STAGED_HIDDEN}"
print_kv "staged_heads" "${STAGED_HEADS}"
print_kv "staged_layers" "${STAGED_LAYERS}"
print_kv "staged_max_step" "${STAGED_MAX_STEP}"
print_kv "staged_max_refine_delta" "${STAGED_MAX_REFINE_DELTA}"

CMD=(
  python foldtree2/learn_production_staged_transformer_geometry.py
  --dataset "${DATASET}"
  --epochs "${EPOCHS:-100}"
  --batch-size "${BATCH_SIZE}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --target-effective-batch-size "${TARGET_EFFECTIVE_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --num-workers "${NUM_WORKERS:-0}"
  --accelerator "${ACCELERATOR:-cuda}"
  --devices "${DEVICES}"
  --strategy "${STRATEGY}"
  --precision "${TRAIN_PRECISION}"
  --pretrained-encoder-path "${PRETRAINED_ENCODER}"
  --pretrained-encoder-full-path "${PRETRAINED_ENCODER}"
  --pretrained-geometry-decoder-path "${PRETRAINED_GEOMETRY_DECODER}"
  --production-coordinate-source "${PRODUCTION_COORDINATE_SOURCE:-auto}"
  --production-coordinate-scale "${PRODUCTION_COORDINATE_SCALE:-1.0}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --save-top-k "${SAVE_TOP_K:--1}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-10}"
  --transformer-width "${TRANSFORMER_WIDTH:-3}"
  --transformer-layers "${TRANSFORMER_LAYERS:-2}"
  --transformer-nheads "${TRANSFORMER_NHEADS:-1}"
  --se3-num-atom-types "${SE3_NUM_ATOM_TYPES:-40}"
  --se3-contact-sketch-top-k "${CONTACT_SKETCH_TOP_K:-8}"
  --se3-contact-sketch-threshold "${CONTACT_SKETCH_THRESHOLD:-0.0}"
  --se3-contact-sketch-min-seq-sep "${CONTACT_SKETCH_MIN_SEQ_SEP:-3}"
  --se3-contact-local-window "${CONTACT_LOCAL_WINDOW:-1}"
  --staged-hidden "${STAGED_HIDDEN}"
  --staged-heads "${STAGED_HEADS}"
  --staged-layers "${STAGED_LAYERS}"
  --staged-dropout "${STAGED_DROPOUT}"
  --staged-max-step "${STAGED_MAX_STEP}"
  --staged-max-refine-delta "${STAGED_MAX_REFINE_DELTA}"
  --stage-loss-weights "${STAGE_LOSS_WEIGHTS}"
  --no-use-frame-fape-loss
  --no-use-quat-geodesic-loss
  --no-use-decoder-angle-loss
  --use-coarse-ca-loss
  --use-coarse-backbone-loss
  --use-coarse-backbone-atom-loss
  --use-coarse-c-loss
  --use-coarse-cb-loss
  --use-coarse-n-loss
  --use-coarse-backbone-fape-loss
  --use-coarse-backbone-angle-loss
  --coarse-backbone-angle-weight "${COARSE_BACKBONE_ANGLE_WEIGHT:-0.05}"
  --no-use-se3
  --no-use-uncertainty-weighting
  --sanitize-nonfinite-grads
)

if [[ -n "${CONFIG:-}" ]]; then
  CMD+=(--config "${CONFIG}")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  CMD+=(--limit-val-batches "${LIMIT_VAL_BATCHES}")
fi

CMD_STRING=$(printf '%q ' "${CMD[@]}")
log "Launch command"
echo "  ${CMD_STRING}"

if "${CMD[@]}"; then
  log "Completed production staged transformer geometry run: ${RUN_TAG}"
else
  exit_code=$?
  log "Training process failed with exit code ${exit_code}"
  exit "${exit_code}"
fi
