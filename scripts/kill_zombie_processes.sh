#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Kill or clean up zombie processes by acting on their parents.

Usage:
  scripts/kill_zombie_processes.sh [options]

Options:
  --dry-run             Print actions without sending signals.
  --terminate-parents   Send SIGTERM to parents that still own zombies after reap attempt.
  --kill-parents        After SIGTERM pass, send SIGKILL to stubborn parents.
  --include-init        Also target PPID 1 (not recommended).
  -h, --help            Show this help message.

Notes:
  - Zombies cannot be killed directly; only their parent can reap them.
  - Safe default behavior sends SIGCHLD to parent processes only.
  - Use --terminate-parents (and optionally --kill-parents) for aggressive cleanup.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

get_zombies() {
  ps -eo pid=,ppid=,stat=,comm= | awk '$3 ~ /Z/ {print $1" "$2" "$3" "$4}'
}

list_parent_pids() {
  awk '{print $2}' | sort -u
}

has_zombies_for_parent() {
  local parent_pid="$1"
  local zombies_text="$2"
  awk -v ppid="$parent_pid" '$2 == ppid {found=1} END {exit found ? 0 : 1}' <<<"$zombies_text"
}

print_zombies() {
  local zombies_text="$1"
  if [[ -z "$zombies_text" ]]; then
    echo "No zombie processes found."
    return 0
  fi

  echo "Zombie processes (PID PPID STAT COMMAND):"
  echo "$zombies_text"
}

DRY_RUN=false
TERMINATE_PARENTS=false
KILL_PARENTS=false
INCLUDE_INIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --terminate-parents)
      TERMINATE_PARENTS=true
      ;;
    --kill-parents)
      KILL_PARENTS=true
      ;;
    --include-init)
      INCLUDE_INIT=true
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
  shift
done

require_cmd ps
require_cmd awk
require_cmd kill

zombies_before="$(get_zombies)"
print_zombies "$zombies_before"

if [[ -z "$zombies_before" ]]; then
  exit 0
fi

mapfile -t parent_pids < <(printf '%s\n' "$zombies_before" | list_parent_pids)

if [[ ${#parent_pids[@]} -eq 0 ]]; then
  echo "No parent PIDs found for zombie list."
  exit 0
fi

echo
echo "Step 1: Request parent reap via SIGCHLD"
for ppid in "${parent_pids[@]}"; do
  if [[ "$ppid" == "1" && "$INCLUDE_INIT" != "true" ]]; then
    echo "Skipping PPID 1 (use --include-init to override)."
    continue
  fi

  if ! kill -0 "$ppid" 2>/dev/null; then
    echo "Parent not running: PPID=$ppid"
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] kill -SIGCHLD $ppid"
  else
    kill -SIGCHLD "$ppid" 2>/dev/null || true
    echo "Sent SIGCHLD to PPID=$ppid"
  fi
done

zombies_after_reap="$(get_zombies)"

echo
echo "After SIGCHLD reap attempt:"
print_zombies "$zombies_after_reap"

if [[ -z "$zombies_after_reap" ]]; then
  echo "Cleanup successful."
  exit 0
fi

if [[ "$TERMINATE_PARENTS" != "true" ]]; then
  echo
  echo "Zombies remain. Re-run with --terminate-parents for aggressive cleanup."
  exit 2
fi

echo
echo "Step 2: Terminate parents still holding zombies (SIGTERM)"
mapfile -t parent_pids_after_reap < <(printf '%s\n' "$zombies_after_reap" | list_parent_pids)
for ppid in "${parent_pids_after_reap[@]}"; do
  if [[ "$ppid" == "1" && "$INCLUDE_INIT" != "true" ]]; then
    echo "Skipping PPID 1 (use --include-init to override)."
    continue
  fi

  if ! kill -0 "$ppid" 2>/dev/null; then
    echo "Parent already exited: PPID=$ppid"
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] kill -SIGTERM $ppid"
  else
    kill -SIGTERM "$ppid" 2>/dev/null || true
    echo "Sent SIGTERM to PPID=$ppid"
  fi
done

zombies_after_term="$(get_zombies)"

echo
echo "After SIGTERM pass:"
print_zombies "$zombies_after_term"

if [[ -z "$zombies_after_term" ]]; then
  echo "Cleanup successful."
  exit 0
fi

if [[ "$KILL_PARENTS" != "true" ]]; then
  echo
  echo "Zombies remain. Re-run with --kill-parents to escalate to SIGKILL."
  exit 3
fi

echo
echo "Step 3: Force kill remaining parents (SIGKILL)"
mapfile -t parent_pids_after_term < <(printf '%s\n' "$zombies_after_term" | list_parent_pids)
for ppid in "${parent_pids_after_term[@]}"; do
  if [[ "$ppid" == "1" && "$INCLUDE_INIT" != "true" ]]; then
    echo "Skipping PPID 1 (use --include-init to override)."
    continue
  fi

  if ! kill -0 "$ppid" 2>/dev/null; then
    echo "Parent already exited: PPID=$ppid"
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] kill -SIGKILL $ppid"
  else
    kill -SIGKILL "$ppid" 2>/dev/null || true
    echo "Sent SIGKILL to PPID=$ppid"
  fi
done

zombies_final="$(get_zombies)"

echo
echo "Final zombie state:"
print_zombies "$zombies_final"

if [[ -z "$zombies_final" ]]; then
  echo "Cleanup successful."
  exit 0
fi

echo "Some zombies still remain (likely kernel-owned or critical parent relationships)."
exit 4
