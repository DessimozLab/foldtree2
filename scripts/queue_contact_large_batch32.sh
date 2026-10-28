#!/usr/bin/env bash
cd /home/dmoi/projects/foldtree2 || exit 1
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$1" in
  0) model=baseline ;;
  1) model=pair ;;
  *) printf 'Expected GPU index 0 or 1\n' >&2; exit 2 ;;
esac
out="runs/contact_objective_large256_batch32_20261027_${model}"
if [[ -e "$out" ]]; then
    printf 'Refusing to overwrite %s\n' "$out" >&2
    exit 1
fi
python scripts/train_contact_levers.py --output "$out" --model "$model" --device "cuda:$1" --hidden 256 --batch-size 32
status=$?
if (( status != 0 )); then
    mkdir -p "$out"
    printf '{"exit_code": %d}\n' "$status" > "$out/failure.json"
fi
exit "$status"
