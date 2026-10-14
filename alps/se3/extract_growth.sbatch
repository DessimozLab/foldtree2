#!/bin/bash
#SBATCH --job-name=ft2-se3-extract
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/extract-growth-%j.out
set -euo pipefail
ulimit -c 0
: "${TRAIN_WORKDIR:?pinned checkout}"
: "${TRAIN_VENV:?verified environment}"
: "${EDF_PATH:?pinned image}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$TRAIN_WORKDIR"
srun --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none --cpu-bind=none --kill-on-bad-exit=1 bash -c '
set -euo pipefail
cd "$TRAIN_WORKDIR"
source "$TRAIN_VENV/bin/activate"
export MPLCONFIGDIR=/tmp/se3-extract-matplotlib
# Isolate extraction-only dependencies from the verified training environment.
prep_deps=/capstor/store/cscs/swissai/a0117/foldtree2/se3/environments/growth-extraction-deps-v2
python -m pip install --no-deps --target "$prep_deps" gemmi==0.7.3 h5py==3.14.0
export PYTHONPATH="$prep_deps:${PYTHONPATH:-}"
# Catch production-loader dependencies before costly cache inventory validation.
python -c "import gemmi, h5py; from foldtree2 import learn_geometry_lightning; from foldtree2.learn_production_geometry_se3_lightning import load_production_decoder; print(\"Production extraction imports verified\")"
python -m scripts.extract_se3_growth_clariden \
    --inputs /capstor/store/cscs/swissai/a0117/foldtree2/se3/inputs/growth_graph_inputs \
    --previous-bundle /iopsstor/scratch/cscs/dmoi/se3/bundle_4096 \
    --output /capstor/store/cscs/swissai/a0117/foldtree2/se3/cache/bundle_growth_8192 \
    --report /capstor/store/cscs/swissai/a0117/foldtree2/se3/artifacts/growth-extraction.json \
    --accept-runtime-drift
rsync -a /capstor/store/cscs/swissai/a0117/foldtree2/se3/cache/bundle_growth_8192/ /iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192/
python -c "from foldtree2.se3_cache import open_bundle; print(open_bundle(\"/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192\")[\"bundle_sha256\"])"
'
# This exact follow-on trial is already authorized. Submit once only after
# extraction, frozen-state checks, durable publication and scratch verification.
python3 -m scripts.submit_se3_growth_trial --available-credit 256 --submit
