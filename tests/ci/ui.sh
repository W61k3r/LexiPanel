#!/usr/bin/env bash
# Browser and HTTP checks against a temporary copy of the panel: scripts parse, every tab works,
# no GET route answers 500, GG's tests, the Workload tab on seeded traffic.
# Needs node, and playwright + jsdom (cd tests/ui && npm install && npx playwright install chromium).
set -uo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd); ui=$root/tests/ui
export NODE_PATH="$ui/node_modules${NODE_PATH:+:$NODE_PATH}"
command -v node >/dev/null || { echo "SKIP ui checks: node is not installed"; exit 2; }
node -e "require('playwright'); require('jsdom')" 2>/dev/null || {
    echo "SKIP ui checks: run  cd tests/ui && npm install && npx playwright install chromium"; exit 2; }
mkdir -p /home/smbadmin/llama /home/smbadmin/models /home/smbadmin/llama_logs 2>/dev/null || true
p1=${UI_PORT:-18190}; p2=$((p1 + 1)); fail=0
step() { echo; echo "== $1"; }
trap '"$ui/serve.sh" stop $p1; "$ui/serve.sh" stop $p2' EXIT
step "inline scripts parse";   python3 "$ui/check_scripts.py" || fail=1
step "GG engine";              node "$root/gg/tests/engine.test.js" || fail=1
"$ui/serve.sh" start $p1 >/dev/null || exit 1
step "every tab";              node "$ui/tabs.test.js" $p1 || fail=1
step "GET routes";             python3 "$ui/routes.py" $p1 || fail=1
step "GG in the page";         node "$root/gg/tests/ui.test.js" "http://127.0.0.1:$p1/" || fail=1
"$ui/serve.sh" start $p2 --seed-workload >/dev/null || exit 1
step "Workload tab";           node "$ui/workload.test.js" $p2 || fail=1
echo; [ $fail = 0 ] && echo "ui checks: all passed" || echo "ui checks: FAILED"
exit $fail
