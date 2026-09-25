#!/usr/bin/env bash
# Guided LexiPanel install: the README's Install steps in order, each explained and
# asked for, with a rough time for each and the elapsed time after it. It runs the
# same scripts you could run by hand (install-panel-deps.sh, install.sh,
# install-autostart.sh, power/install-power.sh, install-onnx.sh); nothing here is extra machinery.
#
#   bash install-interactive.sh           as the account that will own the panel (it sudos)
#   bash install-interactive.sh --check   only the checks: what it would do, changing nothing
#
# Re-running is safe. Updating an existing install copies the code over it and keeps
# every settings file, instance, profile and result (they are not in the release).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
ME=$(id -un)
MYHOME=$(getent passwd "$ME" | cut -d: -f6)
T0=$(date +%s)
N=0; TOTAL=10
elapsed() { local s=$(( $(date +%s) - T0 )); printf '%dm %02ds' $((s / 60)) $((s % 60)); }
step()    { N=$((N + 1)); echo; echo "=== [$N/$TOTAL] $1   (about $2)"; }
ok()      { echo "    done, $(elapsed) so far"; }
ask() {                                     # ask "question" [Y/n|y/N]
    local a d; d=$([ "${2:-Y/n}" = "Y/n" ] && echo y || echo n)
    [ $CHECK = 1 ] && { echo "    (--check) would ask: $1 [${2:-Y/n}]"; return 1; }
    read -rp "    $1 [${2:-Y/n}] " a; a=${a:-$d}; [[ "$a" =~ ^[Yy] ]]
}

echo "LexiPanel guided install $([ $CHECK = 1 ] && echo '(check only: nothing is changed)')"
echo "Typical total: 3-10 minutes, most of it package downloads."

# ------------------------------------------------------------------ 1. checks
step "Checks" "10 s"
[ "$(id -u)" -ne 0 ] || { echo "    Run this as the panel's own account, not root. It uses sudo where needed."; exit 1; }
command -v sudo >/dev/null || { echo "    sudo is required."; exit 1; }
UNIT="$HERE/systemd/LexiPanel-panel.service"
[ -f "$UNIT" ] && [ -f "$HERE/panel.py" ] || { echo "    Run this from the LexiPanel folder (panel.py and systemd/ next to it)."; exit 1; }
UNIT_USER=$(sed -n 's/^User=//p' "$UNIT")
DEST=$(sed -n 's/^WorkingDirectory=//p' "$UNIT")
CODE_HOME=$(sed -n 's/^HOME *=.*Path("\(\/[^"]*\)").*/\1/p' "$HERE/panel.py" | head -1)
echo "    you:               $ME (home $MYHOME)"
echo "    units written for: $UNIT_USER, panel in $DEST"
echo "    panel.py HOME:     ${CODE_HOME:-?}"
echo "    files here:        $HERE"
if [ "$ME" != "$UNIT_USER" ]; then
    echo
    echo "    The systemd units, sudoers rules and scripts are written for the account '$UNIT_USER'"
    echo "    (README: Adapting it to your box). Either run this as $UNIT_USER, or first change"
    echo "    every /home/$UNIT_USER and User=$UNIT_USER in the files this lists:"
    echo "        grep -rln -e '/home/$UNIT_USER' -e 'User=$UNIT_USER' '$HERE'"
    exit 1
fi
FIX_HOME=0
if [ -n "$CODE_HOME" ] && [ "$CODE_HOME" != "$MYHOME" ]; then
    echo "    [!] panel.py looks for models, builds and logs under $CODE_HOME, but your home is $MYHOME."
    ask "Point panel.py (the installed copy) at $MYHOME?" "Y/n" && FIX_HOME=1
fi
if command -v python3 >/dev/null && ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    echo "    [!] python3 is $(python3 -V 2>&1 | cut -d' ' -f2); LexiPanel needs 3.12 or newer."
fi
ok

# ------------------------------------------------------------------ 2. files
step "Put the files in place ($DEST, launch scripts in $MYHOME/llama)" "30 s"
if [ "$HERE" = "$DEST" ]; then
    echo "    already running from $DEST"
elif ask "Copy the panel into $DEST? (an existing install keeps its settings; its old code is saved to a .tgz first)" "Y/n"; then
    if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
        SAVE="$MYHOME/panel-code-before-$(date +%Y%m%d-%H%M%S).tgz"
        tar -czf "$SAVE" -C "$DEST" --exclude='*.gguf' --exclude='./instances' --exclude='./fit' . 2>/dev/null || true
        echo "    previous code saved to $SAVE"
    fi
    mkdir -p "$DEST"
    tar -C "$HERE" --exclude=.git --exclude=__pycache__ --exclude=node_modules -cf - . | tar -C "$DEST" -xf -
    echo "    copied"
else
    echo "    skipped: the rest of the steps use $DEST"
fi
if [ $FIX_HOME = 1 ] && [ -f "$DEST/panel.py" ]; then
    sed -i "/^HOME *=/s|Path(\"$CODE_HOME\")|Path(\"$MYHOME\")|" "$DEST/panel.py"
    echo "    panel.py HOME -> $MYHOME"
fi
if [ -d "$HERE/launch-scripts" ]; then
    [ $CHECK = 1 ] || mkdir -p "$MYHOME/llama"
    for f in "$HERE"/launch-scripts/*.sh; do
        t="$MYHOME/llama/$(basename "$f")"
        if [ ! -e "$t" ]; then
            [ $CHECK = 1 ] && echo "    would add $t" || { install -m 755 "$f" "$t"; echo "    added $t"; }
        elif ! cmp -s "$f" "$t"; then
            echo "    $t differs from this release (kept; yours may be tuned: diff them yourself)"
        fi
    done
fi
ok

# ------------------------------------------------------------------ 2b. access mode
step "Access: single-user (one login, as before) or multi-user (users, roles, API keys, audit)" "30 s"
CUR=$( [ -f "$DEST/auth/config.json" ] && sed -n 's/.*"mode": *"\([a-z]*\)".*/\1/p' "$DEST/auth/config.json" | head -1 || true)
echo "    now: ${CUR:-single}-user"
if [ "${CUR:-single}" = single ] && ask "Switch this panel to multi-user mode (you become its first admin, '$ME')?" "y/N"; then
    python3 "$DEST/auth.py" init --admin "$ME"
    MULTI=1
    echo "    multi-user: Caddy must use systemd/Caddyfile.multi-user.new (the panel logs users in itself)"
fi
ok

# ------------------------------------------------------------------ 3. packages
step "Packages (caddy, ttyd, htpasswd, python3, curl, lspci)" "1-3 min"
MISSING=""
for c in caddy ttyd htpasswd python3 curl lspci; do command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"; done
if [ -z "$MISSING" ]; then
    echo "    all present"
else
    echo "    missing:$MISSING"
    if ask "Install them now (install-panel-deps.sh)?" "Y/n"; then
        bash "$HERE/install-panel-deps.sh"
    elif [ $CHECK = 0 ]; then
        echo "    The front door needs them; run install-panel-deps.sh, then this again."; exit 1
    fi
fi
ok

# ------------------------------------------------------------------ 4. front door
step "Front door: Caddy with a login, the panel and terminal units (install.sh)" "1 min"
echo "    Asks for the panel's username and password, writes /etc/caddy/Caddyfile, starts"
echo "    LexiPanel-panel, LexiPanel-ttyd and caddy."
if ask "Run install.sh now?" "Y/n"; then
    bash "$DEST/install.sh"
fi
ok

# ------------------------------------------------------------------ 5. linger
step "Let instances survive logout and start at boot (loginctl enable-linger)" "5 s"
if [ -e "/var/lib/systemd/linger/$ME" ]; then
    echo "    already on for $ME"
elif ask "Turn on lingering for $ME?" "Y/n"; then
    sudo loginctl enable-linger "$ME" && echo "    on"
fi
ok

# ------------------------------------------------------------------ 6. main at boot
step "Optional: the legacy 'main' LLM on boot (install-autostart.sh)" "1 min"
echo "    Only if you use 'main' (launch-scripts/run_llama_*.sh with MODEL in params.env). It also"
echo "    starts it now. Instances made on the Status tab do not need this."
if ask "Install the main unit and its scoped sudoers rule?" "y/N"; then
    bash "$DEST/install-autostart.sh"
fi
ok

# ------------------------------------------------------------------ 7. power helper
step "Optional: the power helper (Power options and GPU Tuning changes)" "30 s"
echo "    One root helper that validates every value, one sudoers rule (5 verbs), a boot unit."
echo "    Without it both tabs are read-only. Changes no setting by itself."
if ask "Install the power helper?" "Y/n"; then
    sudo PANEL_USER="$ME" bash "$DEST/power/install-power.sh"
fi
ok

# ------------------------------------------------------------------ 8. GPU extras
step "Optional: AMD GPU extras" "10 s (OverDrive needs a reboot)"
if ! ls /sys/bus/pci/drivers/amdgpu/0000:* >/dev/null 2>&1; then
    echo "    no amdgpu card: nothing to do"
else
    if [ -f /etc/udev/rules.d/99-LexiPanel-gpu-powercap.rules ]; then
        echo "    GPU tab power-cap grant: already installed"
    elif ask "Let the GPU tab set power caps (one udev rule: power1_cap group-writable for $ME)?" "Y/n"; then
        echo "ACTION==\"add|change\", SUBSYSTEM==\"hwmon\", ATTR{name}==\"amdgpu\", RUN+=\"/bin/sh -c 'chgrp $ME /sys%p/power1_cap && chmod g+w /sys%p/power1_cap'\"" \
            | sudo tee /etc/udev/rules.d/99-LexiPanel-gpu-powercap.rules >/dev/null
        sudo udevadm control --reload && sudo udevadm trigger --action=change --subsystem-match=hwmon
        echo "    installed"
    fi
    MASK=$(cat /sys/module/amdgpu/parameters/ppfeaturemask 2>/dev/null || echo 0)
    if [ $(( MASK & 0x4000 )) -ne 0 ]; then
        echo "    OverDrive: on (GPU Tuning can change clocks, voltage offset and the fan curve)"
    else
        echo "    OverDrive: off, so GPU Tuning cannot change clocks, voltage or the fan curve."
        if ask "Turn it on at the next boot (systemd/99-amdgpu-overdrive.cfg -> /etc/default/grub.d/, update-grub)?" "y/N"; then
            sudo install -D -m 644 "$DEST/systemd/99-amdgpu-overdrive.cfg" /etc/default/grub.d/99-amdgpu-overdrive.cfg
            sudo update-grub && echo "    on after a reboot"
        fi
    fi
fi
ok

# ------------------------------------------------------------------ 9. ONNX Runtime
step "Optional: ONNX Runtime, for ONNX models on the CPU or an NPU (install-onnx.sh)" "1-2 min"
NPU=$(for d in /sys/class/accel/accel*; do if [ -e "$d/device/driver" ]; then basename "$(readlink -f "$d/device/driver")"; fi; done 2>/dev/null | tr '\n' ' ') || NPU=""
[ -n "$NPU" ] && echo "    NPU driver(s) seen: $NPU" || echo "    no NPU seen in /sys/class/accel (the CPU provider still works)"
if [ -x "$MYHOME/onnxrt/venv/bin/python" ] && "$MYHOME/onnxrt/venv/bin/python" -c "import onnxruntime_genai" 2>/dev/null; then
    echo "    already installed in $MYHOME/onnxrt/venv"
else
    case "$NPU" in *intel_vpu*) EXTRA="--openvino";; *qaic*|*fastrpc*) EXTRA="--qnn";; *) EXTRA="";; esac
    if ask "Install ONNX Runtime GenAI in $MYHOME/onnxrt/venv${EXTRA:+ (with ${EXTRA#--})}?" "$([ -n "$NPU" ] && echo Y/n || echo y/N)"; then
        bash "$DEST/install-onnx.sh" $EXTRA
        case "$NPU" in *amdxdna*) echo "    AMD Ryzen AI NPU: its VitisAI provider comes with AMD's Ryzen AI Software; set the"
                                  echo "    instance's Runtime Python to the Python that installs.";; esac
    fi
fi
ok

# ------------------------------------------------------------------ summary
echo
echo "=== Finished in $(elapsed)"
if [ $CHECK = 0 ]; then
    for s in LexiPanel-panel LexiPanel-ttyd caddy; do
        printf "  %-16s %s\n" "$s" "$(systemctl is-active "$s" 2>/dev/null || true)"
    done
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "Panel: https://${IP:-this-box}   (import Caddy's root certificate to silence the warning:"
echo "       sudo cat /var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt)"
echo "Next:  Builds tab -> Show upstream releases -> Install -> Use for vulkan (or rocm / cuda / cpu)"
echo "       Status tab -> New instance -> Parameters -> Start"
echo "       GPU Tuning -> run a benchmark at stock before changing anything"
echo "       Workload tab fills from real traffic; Auto-fit stays off until you turn it on"
echo "Check: bash $DEST/tests/run_all.sh --quick   (the full gate before any upload: without --quick)"
