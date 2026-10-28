#!/usr/bin/env bash
cd /home/dmoi/projects/foldtree2 || exit 1
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
out=runs/contact_objective_xsa384_batch32_20261027_pair_adamw
if [[ -e "$out" ]]; then
    printf 'Refusing to overwrite %s\n' "$out" >&2
    exit 1
fi
python scripts/contact/train_contact_levers.py --output "$out" --model pair_xsa --device cuda:0 \
    --hidden 384 --context-layers 3 --context-heads 12 --batch-size 32 \
    --optimizer adamw --learning-rate 0.001
status=$?
if (( status != 0 )); then
    mkdir -p "$out"
    printf '{"exit_code": %d}\n' "$status" > "$out/failure.json"
fi
exit "$status"
