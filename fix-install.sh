#!/bin/bash
# Repairs the two install.sh failures. Preserves the password you already set.
#   1. Caddy 2.6.2 calls the directive 'basicauth'; the file says 'basic_auth'
#      (that name only exists from Caddy 2.8).
#   2. A hand-started panel was holding :8090, so the unit restart-looped.
set -e

echo "[1/4] fixing basic_auth -> basicauth in /etc/caddy/Caddyfile"
sudo cp -n /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak
sudo sed -i 's/^\( *\)basic_auth {/\1basicauth {/' /etc/caddy/Caddyfile

echo "[2/4] validating"
sudo caddy validate --config /etc/caddy/Caddyfile 2>&1 | tail -3

echo "[3/4] freeing :8090 if a manual panel still holds it"
if [ -f /home/admin/panel/panel.pid ]; then
    kill "$(cat /home/admin/panel/panel.pid)" 2>/dev/null || true
    rm -f /home/admin/panel/panel.pid
fi
sleep 1

echo "[4/4] restarting services (picks up the GPU tab + amdgpu_top support)"
sudo systemctl restart LexiPanel-panel
sudo systemctl restart caddy
sleep 4

echo
for s in LexiPanel-panel LexiPanel-ttyd caddy; do printf "  %-14s %s\n" "$s" "$(systemctl is-active $s)"; done
echo
if systemctl is-active --quiet caddy && systemctl is-active --quiet LexiPanel-panel; then
    echo "  Panel:  https://192.0.2.10"
    echo
    echo "  Browsers warn until you trust Caddy's internal CA. Export the root with:"
    echo "    sudo cat /var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt"
else
    echo "  Still failing. Show me:"
    echo "    sudo journalctl -u caddy -n 30 --no-pager"
    echo "    sudo journalctl -u LexiPanel-panel -n 30 --no-pager"
fi
