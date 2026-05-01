#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs}"
PATTERN="${2:-config_*.yaml}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true
EXTRA_ARGS=("$@")

MEM_USED_MAX_MB="${MEM_USED_MAX_MB:-2000}"
UTIL_MAX="${UTIL_MAX:-10}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Cannot detect free GPUs." >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "$PATTERN" | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files found in $CONFIG_DIR matching pattern $PATTERN" >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No GPUs detected by nvidia-smi." >&2
  exit 1
fi

mkdir -p runs

declare -A GPU_PID

auto_find_free_gpu() {
  local idx=0
  while IFS=',' read -r used total util; do
    used=$(echo "$used" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    util=$(echo "$util" | tr -d ' ')
    if [[ -z "$used" || -z "$util" ]]; then
      idx=$((idx + 1))
      continue
    fi
    if (( used < MEM_USED_MAX_MB && util < UTIL_MAX )); then
      echo "$idx"
      return 0
    fi
    idx=$((idx + 1))
  done < <(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

reap_finished() {
  for g in "${!GPU_PID[@]}"; do
    local pid="${GPU_PID[$g]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || echo "Job on GPU $g failed." >&2
      unset GPU_PID["$g"]
    fi
  done
}

launch_job() {
  local cfg="$1"
  local gpu="$2"
  local base
  base=$(basename "$cfg")
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  local log_file="runs/${base%.yaml}_gpu${gpu}_${ts}.log"

  echo "Starting $cfg on GPU $gpu (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python foldtree2/learn_monodecoder.py --config "$cfg" --device cuda:0 "${EXTRA_ARGS[@]}" \
    > "$log_file" 2>&1 &

  GPU_PID["$gpu"]=$!
}

idx=0
while [[ $idx -lt ${#CONFIGS[@]} || ${#GPU_PID[@]} -gt 0 ]]; do
  reap_finished

  if [[ $idx -lt ${#CONFIGS[@]} ]]; then
    free_gpu=""
    if free_gpu=$(auto_find_free_gpu); then
      if [[ -z "${GPU_PID[$free_gpu]:-}" ]]; then
        launch_job "${CONFIGS[$idx]}" "$free_gpu"
        idx=$((idx + 1))
        continue
      fi
    fi
  fi

  sleep "$SLEEP_SECONDS"
done

echo "All training jobs completed."
