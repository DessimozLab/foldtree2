#!/usr/bin/env bash
set -eo pipefail
cd /home/dmoi/projects/foldtree2
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=/tmp/foldtree2-se3-deps:$PWD
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec python -u -m scripts.train_se3_revised_pilot \
  --config configs/se3_revised_pilot.yaml \
  --resume-from runs/se3_revised_pilot_20261009/last.pt \
  >> runs/se3_revised_pilot_20261009/console.log 2>&1
