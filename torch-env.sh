#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${FA4_VENV:-$repo_root/.venv-fa4}"
expected_torch="$(tr -d '[:space:]' < "$repo_root/.pytorch-version")"

fail() {
    echo "PyTorch toolchain error: $*" >&2
    echo "Run $repo_root/setup-fa4.sh, then invoke Cargo through $repo_root/torch-env.sh." >&2
    exit 1
}

[[ -x "$venv/bin/python" ]] || fail "missing environment at $venv"
venv="$(cd "$venv" && pwd)"
python="$venv/bin/python"

if ! toolchain="$($python - "$expected_torch" <<'PY'
import pathlib
import site
import sys

try:
    import torch
except Exception as error:
    raise SystemExit(f"could not import torch: {error}")

expected = sys.argv[1]
if torch.__version__ != expected:
    raise SystemExit(f"expected torch {expected}, found {torch.__version__}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"expected CUDA build 13.0, found {torch.version.cuda}")
torch_root = pathlib.Path(torch.__file__).resolve().parent
torch_lib = (torch_root / "lib").resolve()
for library in ("libtorch.so", "libtorch_cpu.so", "libc10.so"):
    if not (torch_lib / library).is_file():
        raise SystemExit(f"missing {torch_lib / library}")
print(site.getsitepackages()[0])
print(torch_lib)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)"; then
    fail "$toolchain"
fi

site_packages="$(sed -n '1p' <<<"$toolchain")"
torch_lib="$(sed -n '2p' <<<"$toolchain")"
cxx11_abi="$(sed -n '3p' <<<"$toolchain")"
[[ -n "$site_packages" && -n "$torch_lib" && -n "$cxx11_abi" ]] \
    || fail "validator returned incomplete environment details"

unset LIBTORCH LIBTORCH_BYPASS_VERSION_CHECK LIBTORCH_INCLUDE LIBTORCH_LIB
unset LIBTORCH_CXX11_ABI LIBTORCH_STATIC PYTHONHOME
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export PYO3_PYTHON="$python"
export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Some hosts provide a different libtorch in /usr/lib. pyo3 also adds that
# directory for libpython, so make the supported wheel the first native link
# search path as well as the first runtime search path.
export RUSTFLAGS="-Lnative=$torch_lib${RUSTFLAGS:+ $RUSTFLAGS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${1:-check}" in
    check)
        echo "PyTorch toolchain ready: torch=$expected_torch lib=$torch_lib cxx11_abi=$cxx11_abi"
        ;;
    ldd)
        [[ $# -eq 2 ]] || fail "usage: torch-env.sh ldd <binary>"
        [[ -x "$2" ]] || fail "binary is not executable: $2"
        mixed="$({ ldd "$2" || true; } | awk -v root="$torch_lib/" '
            /lib(torch|c10|shm)/ && /=>/ {
                resolved=$3
                if (index(resolved, root) != 1) print resolved
            }
        ')"
        [[ -z "$mixed" ]] || fail "mixed libtorch linkage detected outside $torch_lib: $mixed"
        echo "libtorch linkage is confined to $torch_lib"
        ;;
    *)
        cd "$repo_root"
        exec "$@"
        ;;
esac
