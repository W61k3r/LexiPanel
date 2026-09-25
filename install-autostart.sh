#!/bin/bash
# Makes the inference server start on boot, and hands the panel the (narrow)
# privilege it needs to start/stop/restart it.
set -e
PANEL=/home/admin/panel

echo "[1/5] stopping any hand-started server so systemd is the only owner"
pkill -f run_llama_vulkan 2>/dev/null || true
pkill -x llama-server 2>/dev/null || true
sleep 3
pkill -x -9 llama-server 2>/dev/null || true

echo "[2/5] installing unit"
sudo install -m 644 "$PANEL/systemd/LexiPanel-llama.service" /etc/systemd/system/

echo "[3/5] installing scoped sudoers rule"
sudo install -m 440 -o root -g root "$PANEL/systemd/LexiPanel-llama.sudoers" \
     /etc/sudoers.d/LexiPanel-llama
# A malformed sudoers file can lock you out of sudo entirely - validate, and
# remove it again if it does not parse.
if ! sudo visudo -c -f /etc/sudoers.d/LexiPanel-llama; then
    echo "[ABORT] sudoers rule invalid - removing it"
    sudo rm -f /etc/sudoers.d/LexiPanel-llama
    exit 1
fi

echo "[4/5] enabling + starting"
sudo systemctl daemon-reload
sudo systemctl enable LexiPanel-llama
sudo systemctl restart LexiPanel-panel
sudo systemctl start LexiPanel-llama

echo "[5/5] waiting for the model to load (~15-30s)"
for i in $(seq 1 60); do
    curl -sf --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1 && break
    sleep 2
done

echo
for s in LexiPanel-llama LexiPanel-panel LexiPanel-ttyd caddy; do
    printf "  %-14s %-10s boot:%s\n" "$s" "$(systemctl is-active $s)" "$(systemctl is-enabled $s 2>/dev/null)"
done
echo
echo "  health: $(curl -s --max-time 5 http://127.0.0.1:8081/health || echo unreachable)"
echo
echo "Panel Start/Stop/Restart now drive systemd, so a stop actually stays stopped."
echo "Verify boot survival with:  sudo reboot"
