#!/usr/bin/env bash
set -eo pipefail
cd /home/dmoi/projects/foldtree2
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=/tmp/foldtree2-se3-deps:$PWD
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec >> runs/se3_larger_local_20261013/console.log 2>&1
python -u -m scripts.prepare_se3_larger_cohort --config configs/se3_larger_local.yaml
if [[ -f runs/se3_larger_local_20261013/last.pt ]]; then
  exec python -u -m scripts.train_se3_revised_pilot --config configs/se3_larger_local.yaml --resume-from runs/se3_larger_local_20261013/last.pt
elif [[ -f runs/se3_larger_local_20261013/initial.pt ]]; then
  exec python -u -m scripts.train_se3_revised_pilot --config configs/se3_larger_local.yaml --resume-from runs/se3_larger_local_20261013/initial.pt
else
  exec python -u -m scripts.train_se3_revised_pilot --config configs/se3_larger_local.yaml
fi
