#!/usr/bin/env bash
# Puts llama.cpp's built-in web UIs behind the panel's login: inserts the panel-written
# route snippet into /etc/caddy/Caddyfile between "# BEGIN/END LexiPanel web UIs", checks
# the result with `caddy validate`, then RESTARTS Caddy (never reload: `admin off` here).
# Any failure puts the previous Caddyfile back. Run after the panel says the routes are out
# of date (new instance, port change, web UI switched off):
#     sudo bash apply-caddy-webui.sh            install / update the routes
#     sudo bash apply-caddy-webui.sh --remove   take them out again
# The snippet is written by an unprivileged user, so every line of it is checked against
# the only four shapes the panel writes; anything else aborts before Caddy is touched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CF=/etc/caddy/Caddyfile
SNIP="$HERE/caddy-webui.snippet"
[ "$EUID" -eq 0 ] || { echo "run it with sudo: sudo bash $0"; exit 1; }
[ -f "$CF" ] || { echo "no $CF"; exit 1; }
MODE=install; [ "${1:-}" = "--remove" ] && MODE=remove

if [ "$MODE" = install ]; then
    [ -f "$SNIP" ] || { echo "no $SNIP: press 'Write routes' in the panel first"; exit 1; }
    bad=$(grep -n -v -E '^\s*(#.*|redir /ui/[A-Za-z0-9_-]+ /ui/[A-Za-z0-9_-]+/|handle_path /ui/[A-Za-z0-9_-]+/\* \{|reverse_proxy 127\.0\.0\.1:[0-9]{2,5}|\})\s*$' "$SNIP" || true)
    [ -z "$bad" ] || { echo "[ABORT] unexpected line(s) in the snippet:"; echo "$bad"; exit 1; }
    grep -q '^\s*# BEGIN LexiPanel web UIs' "$SNIP" && grep -q '^\s*# END LexiPanel web UIs' "$SNIP" \
        || { echo "[ABORT] snippet has no BEGIN/END markers"; exit 1; }
fi

TS=$(date -u +%Y%m%d_%H%M%S)
BAK="$CF.bak-$TS-lexipanel-webui"
NEW=$(mktemp /etc/caddy/.Caddyfile.new.XXXXXX)
trap 'rm -f "$NEW"' EXIT
cp -p "$CF" "$BAK"

python3 -I - "$CF" "$NEW" "$MODE" "$SNIP" <<'PY'
import re, sys
cf, new, mode, snip = sys.argv[1:5]
txt = open(cf).read()
begin, end = "# BEGIN LexiPanel web UIs", "# END LexiPanel web UIs"
# drop any previous block (with its indentation and trailing newline)
txt = re.sub(r"[ \t]*" + re.escape(begin) + r".*?" + re.escape(end) + r"[^\n]*\n(?:[ \t]*\n)?", "", txt, flags=re.S)
if mode == "install":
    block = open(snip).read()
    lines = txt.split("\n")
    # first site block: the first line ending in '{' that is not the global options block
    site = next((i for i, l in enumerate(lines) if l.rstrip().endswith("{") and l.strip() != "{"
                 and not l.startswith((" ", "\t"))), None)
    if site is None:
        sys.exit("no site block found in the Caddyfile")
    # insert before the catch-all 'handle {' (the panel) inside that site, else before its '}'
    depth, at = 0, None
    for i in range(site, len(lines)):
        l = lines[i].strip()
        if i > site and depth == 1 and re.fullmatch(r"handle\s*\{", l):
            at = i
            while at > site + 1 and lines[at - 1].strip().startswith("#"):
                at -= 1          # keep the '# Control API + UI' comment with its handle
            break
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > site:
            at = i
            break
    if at is None:
        sys.exit("could not find where the site block ends")
    lines[at:at] = block.rstrip("\n").split("\n") + [""]
    txt = "\n".join(lines)
open(new, "w").write(txt)
PY

chmod 644 "$NEW"
if ! caddy validate --config "$NEW" --adapter caddyfile >/tmp/lexipanel-caddy-validate.log 2>&1; then
    echo "[ABORT] caddy validate failed; nothing changed:"; tail -5 /tmp/lexipanel-caddy-validate.log
    rm -f "$BAK"; exit 1
fi
if cmp -s "$NEW" "$CF"; then echo "no change needed"; rm -f "$BAK"; exit 0; fi
install -m 644 -o root -g root "$NEW" "$CF"
if systemctl restart caddy && sleep 2 && systemctl is-active --quiet caddy; then
    echo "done ($MODE). Previous Caddyfile: $BAK"
    grep -o '/ui/[A-Za-z0-9_-]*/' "$CF" | sort -u | sed 's|^|  route: https://<panel>|'
else
    echo "[ROLLBACK] caddy did not come back; restoring $BAK"
    install -m 644 -o root -g root "$BAK" "$CF"
    systemctl restart caddy || true
    exit 1
fi
