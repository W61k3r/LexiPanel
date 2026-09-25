#!/usr/bin/env bash
# ONNX Runtime GenAI for LexiPanel's "onnx" engine, in its own venv (~/onnxrt/venv).
#   bash install-onnx.sh              CPU (works everywhere; also the base the providers build on)
#   bash install-onnx.sh --cuda       NVIDIA: onnxruntime-genai-cuda
#   bash install-onnx.sh --openvino   Intel NPU / GPU / CPU: adds the OpenVINO runtime
#   bash install-onnx.sh --qnn        Qualcomm Hexagon NPU: adds onnxruntime-qnn
# AMD Ryzen AI (VitisAI) is not on PyPI: install AMD's Ryzen AI Software, then set the
# instance's "Runtime Python" to the Python it creates.
set -euo pipefail
venv="$HOME/onnxrt/venv"; pkg=onnxruntime-genai; extra=()
case "${1:-}" in
  --cuda) pkg=onnxruntime-genai-cuda;;
  --openvino) extra=(openvino);;
  --qnn) extra=(onnxruntime-qnn);;
  "") ;;
  *) echo "usage: $0 [--cuda|--openvino|--qnn]"; exit 2;;
esac
python3 -m venv "$venv"
"$venv/bin/pip" install --upgrade pip >/dev/null
"$venv/bin/pip" install --upgrade "$pkg" "${extra[@]}"
"$venv/bin/python" -c "import onnxruntime_genai as og; print('onnxruntime-genai', og.__version__, 'ready in $venv')"
echo "NPUs the kernel sees:"; ls -l /sys/class/accel/ 2>/dev/null | sed 's/^/  /' || echo "  none (/sys/class/accel is empty)"
