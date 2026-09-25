#!/usr/bin/env bash
# Installs (or updates) the LexiPanel power options helper. Run once, and again after the
# panel says the helper is out of date:
#     sudo bash install-power.sh            (from the panel's power/ directory)
#     sudo bash install-power.sh --remove   takes it all out again (settings stay as they are)
# It installs three things and changes no setting by itself:
#   /usr/local/sbin/lexipanel-power           the helper (root:root 0755)
#   /etc/sudoers.d/lexipanel-power            5 verbs for the panel user (validated with visudo)
#   /etc/systemd/system/lexipanel-power.service   records boot defaults, applies the boot profile
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PANEL_USER="${PANEL_USER:-${SUDO_USER:-}}"
[ "$EUID" -eq 0 ] || { echo "run it with sudo: sudo bash $0"; exit 1; }

if [ "${1:-}" = "--remove" ]; then
    systemctl disable lexipanel-power.service 2>/dev/null || true
    rm -f /etc/systemd/system/lexipanel-power.service /etc/sudoers.d/lexipanel-power \
          /usr/local/sbin/lexipanel-power /etc/lexipanel/power.json
    rmdir --ignore-fail-on-non-empty /etc/lexipanel 2>/dev/null || true
    systemctl daemon-reload
    echo "removed. The watchdog/journald drop-ins it may have written stay:"
    ls /etc/systemd/system.conf.d/90-lexipanel-watchdog.conf /etc/modules-load.d/lexipanel-watchdog.conf \
       /etc/systemd/journald.conf.d/90-lexipanel-sync.conf 2>/dev/null || echo "  (none)"
    exit 0
fi

[ -n "$PANEL_USER" ] && id "$PANEL_USER" >/dev/null 2>&1 || {
    echo "which user runs the panel? PANEL_USER=<user> sudo -E bash $0"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
python3 -I -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$HERE/lexipanel_power.py"   # no root-owned __pycache__ in the panel dir

echo "[1/4] helper -> /usr/local/sbin/lexipanel-power"
install -m 755 -o root -g root "$HERE/lexipanel_power.py" /usr/local/sbin/lexipanel-power

echo "[2/4] sudoers rule for $PANEL_USER (5 verbs)"
tmp=$(mktemp)
sed "s/@USER@/$PANEL_USER/" "$HERE/lexipanel-power.sudoers" > "$tmp"
# a malformed sudoers file can lock you out of sudo entirely: validate first
if ! visudo -c -q -f "$tmp"; then echo "[ABORT] sudoers rule invalid, nothing installed"; rm -f "$tmp"; exit 1; fi
install -m 440 -o root -g root "$tmp" /etc/sudoers.d/lexipanel-power
rm -f "$tmp"

echo "[3/4] boot unit"
install -m 644 -o root -g root "$HERE/lexipanel-power.service" /etc/systemd/system/lexipanel-power.service
systemctl daemon-reload
systemctl enable lexipanel-power.service

echo "[4/4] check"
sudo -n -u "$PANEL_USER" sudo -n /usr/local/sbin/lexipanel-power status >/dev/null \
    && echo "  ok: $PANEL_USER can run the helper" || echo "  [WARN] the sudo check failed"
echo "Done. No setting was changed. Pick a boot profile in the panel (Power options);"
echo "the boot defaults are recorded at the next boot."
