#!/usr/bin/env bash
# Release probe launcher, the `run-release-cuda.sh` pattern for this crate's own binary.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/torch-env.sh" cargo run --release -p fused_kernels --bin fused_kernel_probe -- "$@"
