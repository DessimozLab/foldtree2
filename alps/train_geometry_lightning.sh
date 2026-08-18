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
#SBATCH --output=ft2_geom_%j.out
#SBATCH --error=ft2_geom_%j.err
#SBATCH --environment=pygmk3

set -euo pipefail

# Optional: export VENV_PATH before submitting.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

# The geometry decoder must emit three channels for the dot-product contact
# sketch. The SE3 decoder is the only trainable decoder in this launcher.
TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-3}
TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-2}
TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-1}

PROJECT_ROOT=${PROJECT_ROOT:-/users/dmoi/foldtree2/}

pip install --no-cache-dir --no-deps -e "${PROJECT_ROOT}"

DATASET=${DATASET:-/capstor/store/cscs/swissai/a0117/structalnfinal.h5}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/users/dmoi/foldtree2/foldtree2/models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt}

RUN_TAG="se3_dot_contacts_only_${MODEL_TAG:-30char}_bs${BATCH_SIZE:-1}_lr${LEARNING_RATE:-1e-5}"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/capstor/store/cscs/swissai/a0117/chkpts/results/geometry/${RUN_TAG}}
mkdir -p "${CHECKPOINT_DIR}"

cd "${PROJECT_ROOT}"

echo "Starting geometry Lightning run"
echo "  transformer_width=${TRANSFORMER_WIDTH}"
echo "  dataset=${DATASET}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"

CMD=(
  python foldtree2/learn_geometry_lightning.py
  --dataset "${DATASET}"
  --epochs "${EPOCHS:-100}"
  --batch-size "${BATCH_SIZE:-1}"
  --val-batch-size "${VAL_BATCH_SIZE:-1}"
  --target-effective-batch-size "${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
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
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-nheads "${TRANSFORMER_NHEADS}"
  --use-se3
  --train-se3-only
  --se3-input-source geometry_dot_contacts
  --se3-hidden "${SE3_HIDDEN:-64}"
  --se3-depth "${SE3_DEPTH:-6}"
  --se3-heads "${SE3_HEADS:-8}"
  --se3-dim-head "${SE3_DIM_HEAD:-16}"
  --se3-contact-coord-scale "${SE3_CONTACT_COORD_SCALE:-20}"
  --se3-contact-local-window "${SE3_CONTACT_LOCAL_WINDOW:-1}"
  --se3-contact-sketch-top-k "${SE3_CONTACT_SKETCH_TOP_K:-8}"
  --se3-contact-sketch-threshold "${SE3_CONTACT_SKETCH_THRESHOLD:-0.0}"
  --se3-contact-sketch-min-seq-sep "${SE3_CONTACT_SKETCH_MIN_SEQ_SEP:-3}"
  --se3-max-nodes "${SE3_MAX_NODES:-512}"
  --no-use-frame-fape-loss
  --no-use-quat-geodesic-loss
  --no-use-decoder-angle-loss
  --no-use-coarse-ca-loss
  --no-use-coarse-backbone-loss
  --no-use-se3-residue-loss
  --use-se3-angle-loss
  --no-use-se3-atom-refine
  --no-use-uncertainty-weighting
  --sanitize-nonfinite-grads
  --skip-empty-loss-batches
)

if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  CMD+=(--limit-val-batches "${LIMIT_VAL_BATCHES}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Completed geometry Lightning run: ${RUN_TAG}"
