#!/bin/bash
#SBATCH --job-name=ft2-prod-se3-gh200
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=a0117
#SBATCH --output=ft2_prod_se3_%j.out
#SBATCH --error=ft2_prod_se3_%j.err
#SBATCH --environment=pygmk3

set -euo pipefail

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}

# Optional: export VENV_PATH before submitting.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-3}
TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-2}
TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-1}

PROJECT_ROOT=${PROJECT_ROOT:-/users/dmoi/foldtree2/}

pip install --no-cache-dir --no-deps -e "${PROJECT_ROOT}"

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
mkdir -p "${CHECKPOINT_DIR}"

cd "${PROJECT_ROOT}"

echo "Starting production geometry SE3 Lightning run"
echo "  transformer_width=${TRANSFORMER_WIDTH}"
echo "  dataset=${DATASET}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  visualization_dir=${VISUALIZATION_DIR}"
echo "  GH200 launch: tasks_per_node=4, batch_size=${BATCH_SIZE}, effective_batch_size=${TARGET_EFFECTIVE_BATCH_SIZE}"

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
  --devices 1
  --strategy "${STRATEGY:-ddp_find_unused_parameters_true}"
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
  --use-se3-distance-contacts
  --se3-distance-contact-cutoff "${SE3_DISTANCE_CONTACT_CUTOFF:-8.0}"
  --se3-hidden "${SE3_HIDDEN:-128}"
  --se3-depth "${SE3_DEPTH:-8}"
  --se3-heads "${SE3_HEADS:-8}"
  --se3-dim-head "${SE3_DIM_HEAD:-32}"
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
  --no-use-coarse-ca-loss
  --use-coarse-backbone-loss
  --use-coarse-backbone-atom-loss
  --use-coarse-c-loss
  --use-coarse-cb-loss
  --use-coarse-n-loss
  --use-coarse-backbone-fape-loss
  --no-use-coarse-backbone-angle-loss
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
echo "Command: ${CMD[*]}"
srun --ntasks-per-node=4 "${CMD[@]}"

echo "Completed production geometry SE3 Lightning run: ${RUN_TAG}"
