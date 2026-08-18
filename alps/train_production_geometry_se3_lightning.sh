#!/bin/bash
#SBATCH --job-name=ft2-prod-se3-gh200
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=a0117
#SBATCH --output=ft2_prod_se3_%j.out
#SBATCH --error=ft2_prod_se3_%j.err
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

dump_vars() {
  local var
  for var in "$@"; do
    print_kv "${var}" "${!var-<unset>}"
  done
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

TRACE_SCRIPT=${TRACE_SCRIPT:-0}
if [[ "${TRACE_SCRIPT}" == "1" ]]; then
  export PS4='+ [$(date +%Y-%m-%dT%H:%M:%S%z)] ${BASH_SOURCE##*/}:${LINENO}: '
  set -x
fi

LOG_DIR=${LOG_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${SCRIPT_NAME%.sh}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "${LOG_FILE}") 2>&1

log "Logging initialized"
print_kv "log_file" "${LOG_FILE}"
print_kv "script" "${0}"
print_kv "host" "$(hostname)"
print_kv "start_time_utc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

log "SLURM metadata"
dump_vars \
  SLURM_JOB_ID \
  SLURM_JOB_NAME \
  SLURM_JOB_NODELIST \
  SLURM_NNODES \
  SLURM_NTASKS \
  SLURM_NTASKS_PER_NODE \
  SLURM_CPUS_PER_TASK \
  SLURM_GPUS \
  SLURM_GPUS_PER_NODE \
  SLURM_GPUS_PER_TASK \
  SLURM_SUBMIT_DIR

log "Runtime environment"
dump_vars NCCL_DEBUG PYTHONFAULTHANDLER CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS

# Optional: export VENV_PATH before submitting.
if [[ -n "${VENV_PATH:-}" ]]; then
  log "Activating virtual environment from VENV_PATH=${VENV_PATH}"
  source "${VENV_PATH}/bin/activate"
else
  log "VENV_PATH not set; using current environment"
fi

TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-3}
TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-2}
TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-1}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DEVICES=${DEVICES:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}
else
  DEVICES=${DEVICES:-${SLURM_GPUS_PER_NODE:-1}}
fi
DEVICES=${DEVICES:-1}

if [[ -z "${STRATEGY:-}" ]]; then
  if [[ "${DEVICES}" == "1" ]]; then
    STRATEGY=auto
  else
    STRATEGY=ddp_find_unused_parameters_true
  fi
fi

PROJECT_ROOT=${PROJECT_ROOT:-/users/dmoi/foldtree2/}

log "Installing editable package from PROJECT_ROOT=${PROJECT_ROOT}"
pip install --no-cache-dir --no-deps -e "${PROJECT_ROOT}"

log "Python and GPU runtime probes"
print_kv "python" "$(command -v python || echo not_found)"
print_kv "python_version" "$(python --version 2>&1 || echo unavailable)"
if python - <<'PY'
try:
    import torch
    print(f"  torch_version                         {torch.__version__}")
    print(f"  torch_cuda_available                  {torch.cuda.is_available()}")
    print(f"  torch_cuda_device_count               {torch.cuda.device_count()}")
except Exception as exc:
    print(f"  torch_probe_failed                    {exc}")
PY
then
  :
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi -L"
  nvidia-smi -L || true
  log "nvidia-smi utilization snapshot"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader || true
else
  log "nvidia-smi not available on PATH"
fi

DATASET=${DATASET:-/capstor/store/cscs/swissai/a0117/structalnfinal.h5}
MODEL_TAG=${MODEL_TAG:-40char}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-${PROJECT_ROOT}/models/production/40char_minimal_decoder/final_40char_mk2_contactsfix_aa_encoder_full_epoch_41.pt}
PRETRAINED_GEOMETRY_DECODER=${PRETRAINED_GEOMETRY_DECODER:-${PROJECT_ROOT}/models/production/40char_minimal_decoder/final_40char_mk2_contactsfix_aa_decoder_full_epoch_41.pt}

BATCH_SIZE=${BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE:-128}
RUN_TAG="prod_se3_${MODEL_TAG}_gh200_bs${BATCH_SIZE}_eff${TARGET_EFFECTIVE_BATCH_SIZE}"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/a0117/chkpts/results/geometry/${RUN_TAG}}
VISUALIZATION_DIR=${VISUALIZATION_DIR:-${CHECKPOINT_DIR}/visualizations}
VISUALIZATION_SAMPLE_INDEX=${VISUALIZATION_SAMPLE_INDEX:-0}
VISUALIZATION_MAX_RESIDUES=${VISUALIZATION_MAX_RESIDUES:-256}
mkdir -p "${CHECKPOINT_DIR}" "${VISUALIZATION_DIR}"

cd "${PROJECT_ROOT}"

log "Starting production geometry SE3 Lightning run"
print_kv "transformer_width" "${TRANSFORMER_WIDTH}"
print_kv "transformer_layers" "${TRANSFORMER_LAYERS}"
print_kv "transformer_nheads" "${TRANSFORMER_NHEADS}"
print_kv "dataset" "${DATASET}"
print_kv "checkpoint_dir" "${CHECKPOINT_DIR}"
print_kv "visualization_dir" "${VISUALIZATION_DIR}"
print_kv "batch_size" "${BATCH_SIZE}"
print_kv "val_batch_size" "${VAL_BATCH_SIZE}"
print_kv "target_effective_batch_size" "${TARGET_EFFECTIVE_BATCH_SIZE}"
print_kv "run_tag" "${RUN_TAG}"
print_kv "devices" "${DEVICES}"
print_kv "strategy" "${STRATEGY}"

CMD=(
  python foldtree2/learn_production_geometry_se3_lightning.py
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
  --precision "${PRECISION:-32-true}"
  --pretrained-encoder-path "${PRETRAINED_ENCODER}"
  --pretrained-encoder-full-path "${PRETRAINED_ENCODER}"
  --pretrained-geometry-decoder-path "${PRETRAINED_GEOMETRY_DECODER}"
  --production-coordinate-source "${PRODUCTION_COORDINATE_SOURCE:-auto}"
  --production-coordinate-scale "${PRODUCTION_COORDINATE_SCALE:-1.0}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --enable-epoch-visualizations
  --visualization-dir "${VISUALIZATION_DIR}"
  --visualization-sample-index "${VISUALIZATION_SAMPLE_INDEX}"
  --visualization-max-residues "${VISUALIZATION_MAX_RESIDUES}"
  --save-top-k "${SAVE_TOP_K:--1}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-10}"
  --transformer-width "${TRANSFORMER_WIDTH}"
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-nheads "${TRANSFORMER_NHEADS}"
  --use-se3
  --train-se3-only
  --se3-input-source geometry_dot_contacts
  --se3-use-codebook-vectors
  --se3-use-distance-contacts
  --se3-distance-contact-cutoff "${SE3_DISTANCE_CONTACT_CUTOFF:-8.0}"
  --se3-hidden "${SE3_HIDDEN:-128}"
  --se3-depth "${SE3_DEPTH:-8}"
  --se3-heads "${SE3_HEADS:-8}"
  --se3-dim-head "${SE3_DIM_HEAD:-32}"
  --se3-residue-hidden "${SE3_RESIDUE_HIDDEN:-128}"
  --se3-residue-depth "${SE3_RESIDUE_DEPTH:-8}"
  --se3-residue-heads "${SE3_RESIDUE_HEADS:-8}"
  --se3-residue-dim-head "${SE3_RESIDUE_DIM_HEAD:-32}"
  --se3-atom-hidden "${SE3_ATOM_HIDDEN:-128}"
  --se3-atom-depth "${SE3_ATOM_DEPTH:-6}"
  --se3-atom-heads "${SE3_ATOM_HEADS:-8}"
  --se3-atom-dim-head "${SE3_ATOM_DIM_HEAD:-32}"
  --se3-num-atom-types "${SE3_NUM_ATOM_TYPES:-0}"
  --se3-contact-coord-scale "${SE3_CONTACT_COORD_SCALE:-1.0}"
  --se3-contact-local-window "${SE3_CONTACT_LOCAL_WINDOW:-1}"
  --se3-contact-sketch-top-k "${SE3_CONTACT_SKETCH_TOP_K:-16}"
  --se3-contact-sketch-threshold "${SE3_CONTACT_SKETCH_THRESHOLD:-0.0}"
  --se3-contact-sketch-min-seq-sep "${SE3_CONTACT_SKETCH_MIN_SEQ_SEP:-3}"
  --se3-max-nodes "${SE3_MAX_NODES:-0}"
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
  --use-se3-coarse-geometry-losses
  --use-se3-residue-loss
  --use-se3-angle-loss
  --use-se3-atom-refine
  --use-se3-atom-loss
  --use-se3-atom-fape-loss
  --no-use-uncertainty-weighting
  --sanitize-nonfinite-grads
  --skip-empty-loss-batches
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

RUN_START_EPOCH=$(date +%s)
log "Launching training process directly (no srun)"
if "${CMD[@]}"; then
  RUN_EXIT_CODE=0
else
  RUN_EXIT_CODE=$?
fi
RUN_DURATION=$(( $(date +%s) - RUN_START_EPOCH ))
if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
  log "Training process failed with exit code ${RUN_EXIT_CODE} after ${RUN_DURATION}s"
  exit "${RUN_EXIT_CODE}"
fi
log "Training process completed successfully in ${RUN_DURATION}s"

log "Completed production geometry SE3 Lightning run: ${RUN_TAG}"
