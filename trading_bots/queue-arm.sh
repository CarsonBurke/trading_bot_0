#!/usr/bin/env bash
# Waits OUTSIDE the mlq queue for the card, then submits ONE arm at the shared default priority.
#
# Usage: trading_bots/queue-arm.sh <run-name> <pinned-binary> <subcommand> [args...]
#   TIME_LIMIT (default 40m)  the honest measured cost of the arm, not a padded reservation
#   PRIORITY   (default 0)    the same priority every other tenant of this queue uses
#   WAIT_FREE_MIB (default 20500)
#
# The waiting happens here rather than inside a lease because a job that squats a lease to wait
# starves every other tenant of the queue: job 5591 did exactly that to seven siblings. Out here
# it costs nobody a lease, an admission slot or a priority inversion, and the submit only happens
# once the card has actually been free for three minutes - which by construction means no large
# tenant is mid-run.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run="${1:?usage: queue-arm.sh <run-name> <binary> <subcommand> [args...]}"; shift
binary="${1:?usage: queue-arm.sh <run-name> <binary> <subcommand> [args...]}"; shift
need_free_mib="${WAIT_FREE_MIB:-20500}"
stable=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$((total - used))
  if [ "$free" -ge "$need_free_mib" ]; then stable=$((stable + 1)); else stable=0; fi
  if [ "$stable" -ge 6 ]; then
    echo "card free ${free} MiB of ${total} held for ${stable} samples; submitting ${run}"
    break
  fi
  sleep 30
done
exec mlq submit --name "$run" --priority "${PRIORITY:-0}" --max-parallel-runs 1 \
  --max-attempts 1 --time-limit "${TIME_LIMIT:-40m}" --cwd "$repo_root" -- \
  "$repo_root/trading_bots/run-arm.sh" "$run" "$binary" "$@"
