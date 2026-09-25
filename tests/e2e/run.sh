#!/usr/bin/env bash
# Auto-fit end to end, on a TEMPORARY COPY of this panel's code, with a stand-in llama-server
# (tests/e2e/fake_llama.py) on <port>+100 and the copy's UI on http://127.0.0.1:<port>/.
# Your settings, instances, servers and GPU are not touched; the copy is deleted afterwards.
#   bash tests/e2e/run.sh confirm|regress|interrupt [port]      (about 3 minutes each)
#   HOLD=120 bash tests/e2e/run.sh confirm    keeps the copy's UI up 2 minutes at the end
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
panel=$(cd "$here/../.." && pwd)
sc=${1:-confirm}; port=${2:-18290}
case "$sc" in confirm|regress|interrupt) ;; *) echo "usage: $0 confirm|regress|interrupt [port]"; exit 2;; esac
tmp=$(mktemp -d /tmp/lexipanel-e2e.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
# code only: none of the runtime state .gitignore lists comes along
(cd "$panel" && tar --exclude=.git --exclude=__pycache__ --exclude='params*.env' --exclude=builds.env \
    --exclude='engine-*.json' --exclude='gpu-power.json*' --exclude=main-devices.json --exclude=instances \
    --exclude=profiles --exclude=curves --exclude=optimize --exclude=refusal --exclude=power-state \
    --exclude=gpu-tune --exclude=gpu-bios --exclude=workload --exclude='*.gguf' -cf - .) | (cd "$tmp" && tar -xf -)
python3 "$tmp/tests/e2e/autofit_e2e.py" "$tmp" "$port" "$sc"
