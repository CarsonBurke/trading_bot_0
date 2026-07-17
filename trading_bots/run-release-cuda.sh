#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${FA4_VENV:-$repo_root/.venv-fa4}"
if [[ ! -x "$venv/bin/python" ]]; then
    echo "FA4 environment missing; run $repo_root/setup-fa4.sh" >&2
    exit 1
fi
site_packages="$($venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"

unset LIBTORCH
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"
export LIBTORCH_USE_PYTORCH=1

cd "$repo_root"
exec cargo run --release -p trading_bot_0 -- "$@"
