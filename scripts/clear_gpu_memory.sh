#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found. NVIDIA drivers may not be installed." >&2
  exit 1
fi

echo "Checking GPU compute processes..."

# Query only compute apps so we do not kill display processes such as Xorg.
mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF {print $1}' | sort -u)

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No GPU compute processes found."
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  exit 0
fi

echo "Found ${#pids[@]} GPU compute process(es): ${pids[*]}"
echo "Sending SIGTERM..."
for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 2

remaining=()
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    remaining+=("$pid")
  fi
done

if [[ ${#remaining[@]} -gt 0 ]]; then
  echo "Still running after SIGTERM: ${remaining[*]}"
  echo "Sending SIGKILL..."
  for pid in "${remaining[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
fi

sleep 1

echo
echo "GPU memory usage after cleanup:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo
echo "Active GPU compute processes after cleanup:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
