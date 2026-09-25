#!/bin/bash
# Start the panel API, recording a pidfile so restarts never rely on pgrep -f
# (which matches the calling shell's own command line and kills it).
cd /home/admin/panel
PIDFILE=/home/admin/panel/panel.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    kill "$(cat $PIDFILE)"; sleep 2
fi
nohup python3 /home/admin/panel/panel.py > /tmp/panel.log 2>&1 &
echo $! > "$PIDFILE"
sleep 3
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "panel FAILED to stay up:"; tail -8 /tmp/panel.log
    OWNER=$(ss -tlnp 2>/dev/null | awk '/127.0.0.1:8090/{print}' | grep -o 'pid=[0-9]*' | head -1)
    [ -n "$OWNER" ] && echo "NOTE: :8090 is held by $OWNER - probably the systemd unit."
    echo "      If so this script is redundant: sudo systemctl restart LexiPanel-panel"
    exit 1
fi
curl -sf --max-time 5 http://127.0.0.1:8090/api/status >/dev/null \
  && echo "panel up (pid $PID)" || { echo "started but not answering:"; tail -5 /tmp/panel.log; }
