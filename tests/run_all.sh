#!/usr/bin/env bash
# The release gate, run before uploading: the same checks GitHub runs (.github/workflows/ci.yml).
#   bash tests/run_all.sh            everything (about 6 minutes)
#   bash tests/run_all.sh --quick    compile + unit tests + script syntax only (under a minute)
set -uo pipefail
ci=$(cd "$(dirname "$0")" && pwd)/ci
declare -A res
bash "$ci/python.sh"; res[python]=$?
if [ "${1:-}" = --quick ]; then
    python3 "$ci/../ui/check_scripts.py" >/dev/null && res[scripts]=0 || res[scripts]=1
else
    bash "$ci/ui.sh"; res[ui]=$?
    bash "$ci/e2e.sh"; res[e2e]=$?
fi
echo; echo "================ summary"; bad=0
for k in "${!res[@]}"; do
    case ${res[$k]} in 0) s=PASS;; 2) s="SKIP (see above)";; *) s=FAIL; bad=1;; esac
    printf '%-8s %s\n' "$k" "$s"
done
[ $bad = 0 ] && echo "OK to upload" || echo "NOT OK: fix the failures above before uploading"
exit $bad
