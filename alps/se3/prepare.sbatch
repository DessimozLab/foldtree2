#!/bin/bash
#SBATCH --job-name=ft2-se3-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/se3-env-%j.out
set -euo pipefail
: "${TRAIN_WORKDIR:?absolute isolated checkout}"
: "${EDF_PATH:?absolute EDF}"
: "${TRAIN_VENV:?persistent ARM container-specific environment}"
cd "$TRAIN_WORKDIR"
test -s alps/se3/requirements.txt || { echo "Missing release dependency file" >&2; exit 2; }
: "${SOURCE_CACHE:?durable verified frozen bundle}"
: "${TARGET_CACHE:?scratch frozen bundle}"
[[ "$TARGET_CACHE" == /iopsstor/scratch/cscs/dmoi/se3/* ]] || exit 2
mkdir -p "$TARGET_CACHE"
rsync -a "$SOURCE_CACHE/" "$TARGET_CACHE/"
# Install once, outside torchrun. Keep the image's torch/CUDA packages fixed.
srun --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none --cpu-bind=none \
    --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    cd "$TRAIN_WORKDIR"
    [[ ! -e "$TRAIN_VENV" ]] || { echo "Refusing to overwrite existing environment" >&2; exit 2; }
    python -m venv --system-site-packages "$TRAIN_VENV"
    source "$TRAIN_VENV/bin/activate"
    python -c "import importlib.metadata as m; print(\"torch==\"+m.version(\"torch\"))" > "$TRAIN_VENV/torch-constraint.txt"
    python -m pip install -c "$TRAIN_VENV/torch-constraint.txt" -r alps/se3/requirements.txt
    python -m pip freeze > "$TRAIN_VENV/resolved-requirements.txt"
    python alps/se3/inspect_environment.py
    python -c "import os; from foldtree2.se3_cache import open_bundle; print(open_bundle(os.environ[\"TARGET_CACHE\"])[\"bundle_sha256\"])"
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
' > "logs/environment-${SLURM_JOB_ID}.jsonl" 2>&1
