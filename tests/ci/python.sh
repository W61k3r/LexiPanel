#!/usr/bin/env bash
# Compile every Python file, bash -n every script, run the unit tests (safety checks included).
set -uo pipefail
cd "$(dirname "$0")/../.."
fail=0
out=$(find . -name '*.py' -not -path './.git/*' -not -path '*/node_modules/*' -print0 \
      | xargs -0 -n1 python3 -m py_compile 2>&1) || true
[ -z "$out" ] && echo "ok   every .py compiles on $(python3 --version)" || { echo "FAIL compile"; echo "$out"; fail=1; }
out=$(find . -name '*.sh' -not -path './.git/*' -not -path '*/node_modules/*' -print0 | xargs -0 -n1 bash -n 2>&1) || true
[ -z "$out" ] && echo "ok   every .sh parses" || { echo "FAIL bash -n"; echo "$out"; fail=1; }
python3 -m unittest discover tests || fail=1
find . -name __pycache__ -not -path './.git/*' -prune -exec rm -rf {} + 2>/dev/null
exit $fail
