#!/usr/bin/env bash
# Fixed, reviewed inactive dataset targets; preserve all existing consumer paths.
set -euo pipefail
project_root=/home/dmoi/projects/foldtree2
data_root=/mnt/data1/foldtree2-workstation-data
targets=(
  foldtree2/structalnfinal.h5
  foldtree2/notebooks/structs_training_mk2.h5
  foldtree2/notebooks/structs_train_final_big.h5
  benchmark_output/information_content_benchmark_dataset.h5
)

assert_unused() {
  local usage_status=0
  fuser "$1" >/dev/null 2>&1 || usage_status=$?
  if [[ "$usage_status" != 1 ]]; then
    echo "Refusing relocation: file open or usage check failed ($usage_status): $1"
    exit 1
  fi
}

for relative in "${targets[@]}"; do
  source_path="$project_root/$relative"
  destination="$data_root/$relative"
  temporary="$destination.relocation-pending"
  link_path="$source_path.relocation-link"
  if [[ -L "$source_path" ]]; then
    [[ "$(readlink "$source_path")" == "$destination" && -f "$destination" ]] || exit 1
    echo "Already relocated: $source_path"
    continue
  fi
  [[ -f "$source_path" && ! -e "$destination" && ! -L "$destination" ]] || exit 1
  [[ ! -e "$link_path" && ! -L "$link_path" && ! -L "$temporary" ]] || exit 1
  assert_unused "$source_path"
  initial_stat=$(stat -c '%s:%Y:%Z' "$source_path")
  mkdir -p "$(dirname "$destination")"
  echo "COPY $(date --iso-8601=seconds) $source_path -> $destination"
  # Resume interrupted copies, but do not saturate the disk running analyses.
  ionice -c 3 nice -n 15 rsync -a --partial --info=progress2 --bwlimit=75000 \
    "$source_path" "$temporary"
  echo "VERIFY $(date --iso-8601=seconds) $relative"
  ionice -c 3 nice -n 15 cmp -- "$source_path" "$temporary"
  [[ "$(stat -c '%s:%Y:%Z' "$source_path")" == "$initial_stat" ]] || exit 1
  assert_unused "$source_path"
  mv -T -- "$temporary" "$destination"
  ln -s -- "$destination" "$link_path"
  # Atomic path replacement only after the complete copy passes comparison.
  # The original data remains recoverable at destination; no directory is deleted.
  mv -T -- "$link_path" "$source_path"
  echo "DONE $(date --iso-8601=seconds) $source_path -> $(readlink "$source_path")"
  df -h "$project_root" "$data_root"
done
echo "All four reviewed datasets relocated and verified."
