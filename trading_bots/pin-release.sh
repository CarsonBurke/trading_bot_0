#!/usr/bin/env bash
# Builds the release binary and pins an immutable, content-named copy under training/binaries/.
#
# Pinning is not bureaucracy: `run-release-cuda.sh` rebuilds from the CURRENT tree, so a queued
# or running job whose command names the cargo target directory silently changes body the moment
# anyone lands an edit. That has already cost this project two false results - a job measured on
# an instrumented build nobody meant to queue, and a stamp overwritten under a running arm. A
# content-named copy is the only way a result can be attributed to a body afterwards.
#
# Usage: trading_bots/pin-release.sh <label>
# Prints the pinned path on stdout; everything else goes to stderr so it composes.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
label="${1:?usage: pin-release.sh <label>}"
cd "$repo_root"
"$repo_root/torch-env.sh" cargo build --release -p trading_bot_0 >&2
target="$("$repo_root/torch-env.sh" cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r .target_directory)/release/trading_bot_0"
sha="$(sha256sum "$target" | cut -c1-12)"
mkdir -p "$repo_root/training/binaries"
pinned="$repo_root/training/binaries/${label}-${sha}"
if [ ! -e "$pinned" ]; then
  cp "$target" "$pinned"
  chmod a-w "$pinned"
fi
echo "pinned $pinned (sha256 $(sha256sum "$target" | cut -c1-64))" >&2
echo "$pinned"
