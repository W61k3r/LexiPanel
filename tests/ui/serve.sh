#!/usr/bin/env bash
# Run the panel from a temporary copy of this folder's code on a port, for the browser checks.
#   serve.sh start <port> [--seed-workload]     prints the copy's folder
#   serve.sh stop <port>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); panel=$(cd "$here/../.." && pwd)
cmd=$1; port=$2; dir="${TMPDIR:-/tmp}/lexipanel-ui-$port"
if [ "$cmd" = stop ]; then
    [ -f "$dir/panel.pid" ] && kill "$(cat "$dir/panel.pid")" 2>/dev/null || true
    rm -rf "$dir"; exit 0
fi
"$0" stop "$port"
mkdir -p "$dir"
(cd "$panel" && tar --exclude=.git --exclude=__pycache__ --exclude=node_modules --exclude='params*.env' \
    --exclude=builds.env --exclude='engine-*.json' --exclude='gpu-power.json*' --exclude=main-devices.json \
    --exclude=instances --exclude=profiles --exclude=curves --exclude=optimize --exclude=refusal \
    --exclude=power-state --exclude=gpu-tune --exclude=gpu-bios --exclude=workload --exclude='*.gguf' -cf - .) \
    | (cd "$dir" && tar -xf -)
[ "${3:-}" = --seed-workload ] && python3 "$dir/tests/e2e/seed_workload.py" "$dir" >/dev/null
INF01_PANEL_DIR="$dir" PANEL_PORT="$port" nohup python3 "$dir/panel.py" > "$dir/panel.log" 2>&1 &
echo $! > "$dir/panel.pid"
for i in $(seq 1 80); do
    curl -sf "http://127.0.0.1:$port/api/servers" >/dev/null 2>&1 && { echo "$dir"; exit 0; }
    kill -0 "$(cat "$dir/panel.pid")" 2>/dev/null || { echo "panel died:" >&2; tail -30 "$dir/panel.log" >&2; exit 1; }
    sleep 0.5
done
echo "panel did not answer on :$port" >&2; tail -30 "$dir/panel.log" >&2; exit 1
