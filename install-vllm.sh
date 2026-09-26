#!/usr/bin/env bash
# Install vLLM into its own venv for LexiPanel's vLLM instances (engine "vllm").
#   bash install-vllm.sh           NVIDIA / CUDA  -> ~/vllm/venv-cu
#   bash install-vllm.sh --rocm    AMD / ROCm     -> ~/vllm/venv-rocm   (Python 3.12 only)
# Never touches the system Python. Needs Python 3.10-3.13 (3.12 for ROCm): set VLLM_PYTHON to
# pick one, otherwise the first of python3.12, python3.13, python3.11, python3.10 on PATH.
# uv does the install because it prefers the extra index over PyPI; plain pip can silently
# take the CUDA wheel of the same version when ROCm was asked for.
set -euo pipefail
FLAVOUR=cu; [ "${1:-}" = "--rocm" ] && FLAVOUR=rocm
DEST="${VLLM_VENV:-$HOME/vllm/venv-$FLAVOUR}"
PY="${VLLM_PYTHON:-}"
if [ -z "$PY" ]; then
    for c in python3.12 python3.13 python3.11 python3.10; do
        if command -v "$c" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
    done
fi
[ -n "$PY" ] || { echo "no Python 3.10-3.13 found: set VLLM_PYTHON=/path/to/python3.12"; exit 1; }
VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$FLAVOUR:$VER" in
    rocm:3.12|cu:3.10|cu:3.11|cu:3.12|cu:3.13) ;;
    *) echo "Python $VER cannot install the $FLAVOUR wheels (CUDA: 3.10-3.13, ROCm: 3.12)"; exit 1 ;;
esac
mkdir -p "$(dirname "$DEST")"
FREE=$(df -Pk "$(dirname "$DEST")" | awk 'NR==2{print int($4/1048576)}')
[ "${FREE:-0}" -ge 25 ] || { echo "need ~15 GB for the venv and room to spare (${FREE} GB free)"; exit 1; }
"$PY" -m venv "$DEST"
"$DEST/bin/pip" install -q --upgrade pip uv
if [ "$FLAVOUR" = cu ]; then
    "$DEST/bin/uv" pip install --python "$DEST/bin/python" vllm --extra-index-url https://download.pytorch.org/whl/cu129
else
    "$DEST/bin/uv" pip install --python "$DEST/bin/python" vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
fi
"$DEST/bin/python" - <<'PY'
import importlib.metadata as m, torch
print("vllm", m.version("vllm"), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| hip", getattr(torch.version, "hip", None), "| devices", torch.cuda.device_count())
PY
echo "installed: $DEST   (LexiPanel starts every vLLM instance with VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1)"
