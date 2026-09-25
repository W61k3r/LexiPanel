#!/bin/bash
# Installs the LexiPanel admin panel. Needs sudo. Idempotent.
set -e
PANEL=/home/admin/panel

command -v caddy >/dev/null || { echo "[ABORT] caddy missing. Run install-panel-deps.sh first."; exit 1; }
command -v ttyd  >/dev/null || { echo "[ABORT] ttyd missing.  Run install-panel-deps.sh first."; exit 1; }
command -v htpasswd >/dev/null || { echo "[ABORT] apache2-utils missing."; exit 1; }

# --- login ---
read -rp "Panel username [admin]: " USER; USER=${USER:-admin}
read -rsp "Panel password: " PASS; echo
[ -z "$PASS" ] && { echo "[ABORT] empty password"; exit 1; }
HASH=$(caddy hash-password --plaintext "$PASS")

# Caddy renamed basicauth -> basic_auth in 2.8. Pick the right one.
CV=$(caddy version 2>/dev/null | head -1 | grep -oE 'v?2\.[0-9]+' | head -1 | tr -d 'v')
CMIN=${CV#2.}
if [ "${CMIN:-0}" -ge 8 ] 2>/dev/null; then AUTHDIR=basic_auth; else AUTHDIR=basicauth; fi
echo "[INFO] caddy $CV -> using '$AUTHDIR' directive"

sed -e "s|admin \$2a\$14\$REPLACE_ME|$USER $HASH|" \
    -e "s|^\( *\)basic_auth {|\1$AUTHDIR {|" \
    -e "s|^\( *\)basicauth {|\1$AUTHDIR {|" "$PANEL/Caddyfile" > /tmp/Caddyfile.gen
sudo install -D -m 644 /tmp/Caddyfile.gen /etc/caddy/Caddyfile
rm -f /tmp/Caddyfile.gen
sudo mkdir -p /var/log/caddy && sudo chown caddy:caddy /var/log/caddy 2>/dev/null || true

# A hand-started panel would hold :8090 and make the unit restart-loop.
if [ -f "$PANEL/panel.pid" ]; then kill "$(cat "$PANEL/panel.pid")" 2>/dev/null || true; rm -f "$PANEL/panel.pid"; fi
sleep 1

# --- services ---
sudo install -m 644 "$PANEL/systemd/LexiPanel-panel.service" /etc/systemd/system/
sudo install -m 644 "$PANEL/systemd/LexiPanel-ttyd.service"  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now LexiPanel-panel LexiPanel-ttyd
sudo caddy validate --config /etc/caddy/Caddyfile 2>&1 | tail -3
sudo systemctl enable --now caddy
sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy

sleep 3
echo
echo "=== status ==="
for s in LexiPanel-panel LexiPanel-ttyd caddy; do printf "  %-14s %s\n" "$s" "$(systemctl is-active $s)"; done
echo
echo "Panel:  https://192.0.2.10   (user: $USER)"
echo
echo "The cert is from Caddy's internal CA, so browsers warn until you trust its root."
echo "Export it with:"
echo "  sudo cat /var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt"
echo "then import that into your browser or OS trust store."
