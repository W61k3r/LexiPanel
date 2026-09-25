#!/usr/bin/env bash
# Auto-fit end to end: confirm, regress, interrupt (or the ones named), one after another.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); fail=0; port=${E2E_PORT:-18290}
mkdir -p /home/smbadmin/llama /home/smbadmin/models /home/smbadmin/llama_logs 2>/dev/null || true
for sc in ${@:-confirm regress interrupt}; do
    out=$(bash "$here/e2e/run.sh" "$sc" "$port" 2>&1); r=$?
    echo "$out" | grep -E 'RESULT|PASS|FAIL|Traceback|Error' | cut -c1-240
    [ $r = 0 ] || { fail=1; echo "$out" | tail -25; }
    port=$((port + 5))
done
exit $fail
