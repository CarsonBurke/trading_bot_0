#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${FA4_VENV:-$repo_root/.venv-fa4}"

expected_torch="$(tr -d '[:space:]' < "$repo_root/.pytorch-version")"
if [[ ! -x "$venv/bin/python" ]] || ! "$venv/bin/python" -c \
    "import torch; assert torch.__version__ == '$expected_torch'" 2>/dev/null; then
    python3 -m venv --clear "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==$expected_torch"
"$venv/bin/python" -m pip install --pre 'flash-attn-4[cu13]==4.0.0b21'
EXPECTED_TORCH="$expected_torch" "$venv/bin/python" - <<'PY'
import os
import torch
from flash_attn.cute import flash_attn_func

expected = os.environ["EXPECTED_TORCH"]
assert torch.__version__ == expected, torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert (major, minor) == (12, 0), (major, minor)
print(f"FA4 ready: torch={torch.__version__} cuda={torch.version.cuda} sm={major}{minor}")
print(f"kernel={flash_attn_func.__module__}.{flash_attn_func.__name__}")
PY
FA4_VENV="$venv" "$repo_root/torch-env.sh" check
