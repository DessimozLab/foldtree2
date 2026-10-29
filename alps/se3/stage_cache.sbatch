#!/bin/bash
#SBATCH --job-name=ft2-se3-stage
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/se3-stage-%j.out
set -euo pipefail
: "${TRAIN_WORKDIR:?isolated checkout}"
: "${EDF_PATH:?pinned verified EDF}"
: "${TRAIN_VENV:?verified ARM environment}"
: "${SOURCE_CACHE:?durable projectstore cache}"
: "${TARGET_CACHE:?scratch cache}"
cd "$TRAIN_WORKDIR"
[[ "$TARGET_CACHE" == /iopsstor/scratch/cscs/dmoi/se3/* ]] || exit 2
mkdir -p "$TARGET_CACHE"
rsync -a "$SOURCE_CACHE/" "$TARGET_CACHE/"
srun --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none bash -c '
    set -euo pipefail
    cd "$TRAIN_WORKDIR"; source "$TRAIN_VENV/bin/activate"
    python -c "import os; from foldtree2.se3_cache import open_bundle; print(open_bundle(os.environ[\"TARGET_CACHE\"])[\"bundle_sha256\"])"
'
