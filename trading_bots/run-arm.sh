#!/usr/bin/env bash
# The command an mlq arm actually runs: a bounded headroom gate, a self-cleaning run dir, then
# the pinned binary.
#
# Usage: trading_bots/run-arm.sh <run-name> <pinned-binary> <subcommand> [args...]
#
# Two rules are enforced here because both were violated this session:
#
# 1. A lease is NEVER squatted to wait. The gate is bounded at GATE_MINUTES (default 3) and then
#    exits 75, handing the lease straight back. Job 5591 held a 6-hour lease at priority 6
#    polling nvidia-smi and blocked seven sibling jobs from admission while doing no work.
# 2. The run dir is removed here, because mlq retries are otherwise eaten by the (correct)
#    "run dir already exists" guard, which makes --max-attempts useless.
#
# The gate's only job is refusing a start into a tenant that is visibly mid-allocation: a
# captured training step needs 18,044 MiB reserved plus ~780 MiB for the evaluation pass at
# --eval-batch-size 64, and job 5585 passed a one-shot 25,860 MiB check then died inside graph
# capture when a foreign tenant reclaimed the card during the ~90 s of startup and arming.
# Sampling cannot close that race; it only refuses the obviously bad start.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run="${1:?usage: run-arm.sh <run-name> <binary> <subcommand> [args...]}"; shift
binary="${1:?usage: run-arm.sh <run-name> <binary> <subcommand> [args...]}"; shift
need_free_mib="${NEED_FREE_MIB:-19200}"
stable_samples="${STABLE_SAMPLES:-4}"
gate_seconds=$(( ${GATE_MINUTES:-3} * 60 ))
stable=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$((total - used))
  if [ "$free" -ge "$need_free_mib" ]; then stable=$((stable + 1)); else stable=0; fi
  if [ "$stable" -ge "$stable_samples" ]; then
    echo "card headroom ${free} MiB of ${total} held for ${stable} samples: starting ${run} on ${binary}"
    break
  fi
  if [ "$SECONDS" -ge "$gate_seconds" ]; then
    echo "releasing the lease: ${need_free_mib} MiB not free for ${stable_samples} samples within $((gate_seconds / 60)) min, last free ${free} MiB" >&2
    exit 75
  fi
  sleep 15
done
rm -rf "training/runs/$run"
# Only the train-* subcommands take --run; the eval/ceiling/probe instruments are pointed at an
# explicit --checkpoint and --output instead, and reject it.
case "$1" in
  train-*) exec "$repo_root/torch-env.sh" "$binary" "$@" --run "$run" ;;
  *) exec "$repo_root/torch-env.sh" "$binary" "$@" ;;
esac
