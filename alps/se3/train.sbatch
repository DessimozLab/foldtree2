#!/bin/bash
#SBATCH --job-name=ft2-se3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/se3-%j.out
#SBATCH --open-mode=append
set -euo pipefail
ulimit -c 0
: "${TRAIN_WORKDIR:?absolute isolated code checkout}"
: "${EDF_PATH:?absolute EDF path}"
cd "$TRAIN_WORKDIR"
hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${hosts%%$'\n'*}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none \
    --cpu-bind=none --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    cd "$TRAIN_WORKDIR"
    if [[ -n "${TRAIN_VENV:-}" ]]; then source "$TRAIN_VENV/bin/activate"; fi
    exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" \
        --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        --max-restarts=0 -m foldtree2.train_se3_lightning "$@"
' bash "$@"
