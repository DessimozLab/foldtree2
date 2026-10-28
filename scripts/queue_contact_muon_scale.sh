#!/usr/bin/env bash
cd /home/dmoi/projects/foldtree2 || exit 1
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_case() {
    local name=$1 model=$2 gpu=$3 optimizer=$4 lr=$5
    local out="runs/contact_objective_muon384_batch32_20261027_${name}"
    if [[ -e "$out" ]]; then
        printf 'Refusing to overwrite %s\n' "$out" >&2
        return 1
    fi
    python scripts/train_contact_levers.py --output "$out" --model "$model" --device "cuda:$gpu" \
        --hidden 384 --context-layers 3 --context-heads 12 --batch-size 32 \
        --optimizer "$optimizer" --muon-lr 0.02 --learning-rate "$lr"
    local status=$?
    if (( status != 0 )); then
        mkdir -p "$out"
        printf '{"exit_code": %d}\n' "$status" > "$out/failure.json"
    fi
    return "$status"
}
case "$1" in
  0) run_case baseline_muon baseline 0 muon 0.0003 || exit 1
     run_case pair_adamw_control pair 0 adamw 0.001 || exit 1 ;;
  1) run_case pair_muon pair 1 muon 0.0003 || exit 1 ;;
  *) printf 'Expected GPU index 0 or 1\n' >&2; exit 2 ;;
esac
