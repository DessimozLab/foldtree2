#!/usr/bin/env bash
cd /home/dmoi/projects/foldtree2 || exit 1
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
set -u
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_case() {
    local name=$1
    shift
    local out="runs/contact_objective_levers_batch8_20261027_${name}"
    if [[ -e "$out" ]]; then
        printf 'Refusing to overwrite %s\n' "$out" >&2
        return 1
    fi
    python scripts/train_contact_levers.py --output "$out" "$@"
    local status=$?
    if (( status != 0 )); then
        mkdir -p "$out"
        printf '{"exit_code": %d}\n' "$status" > "$out/failure.json"
        printf 'Failed %s with exit code %d\n' "$name" "$status" >&2
    fi
    return "$status"
}

case "$1" in
    0)
        run_case full_baseline --model baseline --device cuda:0 || exit 1
        run_case screen_pair32_control --model pair --device cuda:0 --train-count 512 --train-selection stratified --epochs 6 || exit 1
        run_case screen_pair64_rank --model pair --device cuda:0 --train-count 512 --train-selection stratified --epochs 6 --ranking-weight 0.1 --correction-hidden 64 || exit 1
        ;;
    1)
        run_case full_pair32 --model pair --device cuda:1 || exit 1
        run_case screen_pair32_rank --model pair --device cuda:1 --train-count 512 --train-selection stratified --epochs 6 --ranking-weight 0.1 || exit 1
        run_case screen_pair32_rank_l2 --model pair --device cuda:1 --train-count 512 --train-selection stratified --epochs 6 --ranking-weight 0.1 --correction-l2 0.01 || exit 1
        ;;
    *) printf 'Expected GPU index 0 or 1\n' >&2; exit 2 ;;
esac
