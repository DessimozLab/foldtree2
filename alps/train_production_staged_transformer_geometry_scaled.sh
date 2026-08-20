#!/bin/bash
# Scaled architecture variant of train_production_staged_transformer_geometry.sh.
# Override any defaults below with environment variables at submission time.
#
# Example:
#   sbatch --export=ALL,STAGED_HIDDEN=384,STAGED_LAYERS=6 \
#     alps/train_production_staged_transformer_geometry_scaled.sh

#SBATCH --job-name=ft2-prod-staged-scaled-gh200
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --account=a0117
#SBATCH --output=ft2_prod_staged_scaled_%j.out
#SBATCH --error=ft2_prod_staged_scaled_%j.err
#SBATCH --environment=pygmk3

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The three-stage refiner is widened and deepened, while the frozen production
# geometry transformer also gets a larger representation and more attention.
export STAGED_HIDDEN=${STAGED_HIDDEN:-256}
export STAGED_HEADS=${STAGED_HEADS:-8}
export STAGED_LAYERS=${STAGED_LAYERS:-4}
export STAGED_DROPOUT=${STAGED_DROPOUT:-0.05}
export STAGED_MAX_STEP=${STAGED_MAX_STEP:-4.0}
export STAGED_MAX_REFINE_DELTA=${STAGED_MAX_REFINE_DELTA:-2.0}
export STAGE_LOSS_WEIGHTS=${STAGE_LOSS_WEIGHTS:-0.25,0.5,1.0}

export TRANSFORMER_WIDTH=${TRANSFORMER_WIDTH:-6}
export TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS:-4}
export TRANSFORMER_NHEADS=${TRANSFORMER_NHEADS:-4}

# Keep the scaled run isolated from checkpoints produced by the baseline.
export MODEL_TAG=${MODEL_TAG:-40char_staged_transformer_scaled}

exec "${SCRIPT_DIR}/train_production_staged_transformer_geometry.sh" "$@"