#!/bin/bash
# ============================================================================
# fix-firewall.sh - let agent reach the llama API on 8081.
#
# Safety model (why this cannot lock you out and cannot stall inference):
#   1. It NEVER adds a deny/reject rule, and NEVER runs `ufw enable/disable/reset`.
#      It only inserts ALLOW rules. An added allow cannot take access away.
#   2. It only deletes-and-reinserts a rule that is currently INEFFECTIVE
#      (sitting below a deny that already shadows it), so nothing working is
#      ever removed. Rules for 22/443 that already work are left untouched.
#   3. `ufw reload` re-applies the chains but does not flush conntrack, and
#      ufw's before-rules accept RELATED,ESTABLISHED first - so an in-flight
#      SSH session, panel session, or streaming completion is not dropped.
#      llama-server itself is never signalled, restarted, or reconfigured.
#   4. A dead-man watchdog restores the previous rules after --timeout seconds
#      unless you confirm you still have access. Disable with --no-watchdog.
#
# Usage:
#   sudo ./fix-firewall.sh                          # agent = 192.0.2.20
#   sudo ./fix-firewall.sh --agent 192.0.2.42
#   sudo ./fix-firewall.sh --yes --no-watchdog      # non-interactive
#   sudo ./fix-firewall.sh --revert                 # restore last backup
#   sudo ./fix-firewall.sh --check                  # report only, change nothing
# ============================================================================
set -euo pipefail

AGENT_IP="${AGENT_IP:-192.0.2.20}"
API_PORT=8081
TIMEOUT=180
ASSUME_YES=0
WATCHDOG=1
MODE=apply
BACKUP_ROOT=/var/backups/ufw-fix
SSH_PEERS=""
WEB_PEERS=""

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "  $*"; }
hdr() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- arg parsing
REEXEC_ARGS=()
while (( $# )); do
    case "$1" in
        --agent)      AGENT_IP="$2"; REEXEC_ARGS+=("$1" "$2"); shift 2 ;;
        --timeout)     TIMEOUT="$2";   REEXEC_ARGS+=("$1" "$2"); shift 2 ;;
        --yes|-y)      ASSUME_YES=1;   REEXEC_ARGS+=("$1"); shift ;;
        --no-watchdog) WATCHDOG=0;     REEXEC_ARGS+=("$1"); shift ;;
        --revert)      MODE=revert;    REEXEC_ARGS+=("$1"); shift ;;
        --check)       MODE=check;     REEXEC_ARGS+=("$1"); shift ;;
        --ssh-peers)   SSH_PEERS="$2"; shift 2 ;;
        --web-peers)   WEB_PEERS="$2"; shift 2 ;;
        -h|--help)     awk 'NR>1 && /^#/ {print} NR>1 && !/^#/ {exit}' "$0"; exit 0 ;;
        *)             die "unknown argument: $1" ;;
    esac
done

[[ "$AGENT_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "--agent needs a bare IPv4 address, got '$AGENT_IP'"

# ------------------------------------------------- capture live peers, then su
# Done before the sudo re-exec: sudo drops SSH_CONNECTION, and we want the real
# addresses of whoever is logged in right now so we never strand them.
peers_on() {
    ss -Htn state established "( sport = :$1 )" 2>/dev/null |
        awk '{print $4}' |
        sed -E 's/^\[?([0-9a-fA-F.:]+)\]?:[0-9]+$/\1/' |
        grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -v '^127\.' | sort -u || true
}

if [[ $EUID -ne 0 ]]; then
    SSH_PEERS=$(peers_on 22  | paste -sd, -)
    WEB_PEERS=$(peers_on 443 | paste -sd, -)
    exec sudo "$0" --ssh-peers "${SSH_PEERS:-}" --web-peers "${WEB_PEERS:-}" "${REEXEC_ARGS[@]+"${REEXEC_ARGS[@]}"}"
fi

command -v ufw >/dev/null || die "ufw not installed"
[[ "$(ufw status | head -1)" == *active* ]] || die "ufw is not active; refusing to guess at intent"

# ------------------------------------------------------- rule-order inspection
ip2int() { local IFS=. a b c d; read -r a b c d <<<"$1"; echo $(( (a<<24)|(b<<16)|(c<<8)|d )); }

in_cidr() {  # in_cidr <ip> <cidr-or-ip>
    local ip=$1 cidr=$2 net bits mask
    [[ $cidr == */* ]] || cidr="$cidr/32"
    net=${cidr%/*}; bits=${cidr#*/}
    (( bits == 0 )) && return 0
    mask=$(( (0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF ))
    (( ($(ip2int "$ip") & mask) == ($(ip2int "$net") & mask) ))
}

# Walks ufw-user-input in evaluation order and reports the verdict the first
# matching rule would give: ALLOW, DENY, or NONE (falls through to the policy).
verdict_for() {  # verdict_for <src-ip> <dport>
    local ip=$1 port=$2 line src dports tgt hit
    while read -r line; do
        [[ $line == -A* ]] || continue
        [[ $line == *" -p tcp"* || $line != *" -p "* ]] || continue
        src=0.0.0.0/0
        [[ $line =~ -s\ ([0-9./]+) ]] && src="${BASH_REMATCH[1]}"
        in_cidr "$ip" "$src" || continue
        dports=""
        [[ $line =~ --dport\ ([0-9]+) ]]  && dports="${BASH_REMATCH[1]}"
        [[ $line =~ --dports\ ([0-9,]+) ]] && dports="${BASH_REMATCH[1]}"
        if [[ -n $dports ]]; then
            hit=0
            IFS=, read -ra _p <<<"$dports"
            for p in "${_p[@]}"; do [[ $p == "$port" ]] && hit=1; done
            (( hit )) || continue
        fi
        tgt="${line##* -j }"
        case "$tgt" in
            ACCEPT|*user-limit-accept) echo ALLOW; return ;;
            DROP|REJECT|*user-limit)   echo DENY;  return ;;
        esac
    done < <(iptables -S ufw-user-input 2>/dev/null)
    echo NONE
}

report() {
    hdr "current verdicts (first matching rule in ufw-user-input)"
    note "agent $AGENT_IP -> ${API_PORT}/tcp : $(verdict_for "$AGENT_IP" "$API_PORT")"
    local p
    for p in ${SSH_PEERS//,/ }; do note "ssh peer $p -> 22/tcp        : $(verdict_for "$p" 22)"; done
    for p in ${WEB_PEERS//,/ }; do note "panel peer $p -> 443/tcp     : $(verdict_for "$p" 443)"; done
    [[ -z $SSH_PEERS$WEB_PEERS ]] && note "(no live SSH/panel peers detected - local console?)"
    hdr "ufw status numbered"
    ufw status numbered
}

if [[ $MODE == check ]]; then report; exit 0; fi

# ------------------------------------------------------------------- revert
if [[ $MODE == revert ]]; then
    last=$(ls -1d "$BACKUP_ROOT"/*/ 2>/dev/null | tail -1) || true
    [[ -n ${last:-} ]] || die "no backup under $BACKUP_ROOT"
    echo "Restoring from $last"
    bash "${last}rollback.sh"
    report
    exit 0
fi

# ------------------------------------------------------------------- preflight
hdr "preflight"
note "agent IP        : $AGENT_IP"
note "live ssh peers   : ${SSH_PEERS:-none}"
note "live panel peers : ${WEB_PEERS:-none}"
case ",$SSH_PEERS," in *",$AGENT_IP,"*)
    note "note             : $AGENT_IP is also your live SSH peer - if that is not where agent runs, rerun with --agent <ip>" ;;
esac

if curl -fsS -m 5 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    note "llama-server     : healthy on 127.0.0.1:$API_PORT"
else
    note "llama-server     : NOT answering on 127.0.0.1:$API_PORT (firewall is not your only problem)"
fi

busy=$(curl -fsS -m 5 "http://127.0.0.1:$API_PORT/metrics" 2>/dev/null |
       awk '/^llamacpp:requests_processing /{print $2}' | head -1)
if [[ -n ${busy:-} && ${busy%%.*} -gt 0 ]]; then
    note "in-flight requests: $busy  (reload keeps established connections; they will not be cut)"
else
    note "in-flight requests: ${busy:-unknown}"
fi

if [[ ! -t 0 ]] && (( ! ASSUME_YES )); then
    die "no tty for the confirmation prompt - rerun with --yes (and ideally --no-watchdog)"
fi

report

# --------------------------------------------------------------------- backup
BK="$BACKUP_ROOT/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$BK"
cp -a /etc/ufw/user.rules /etc/ufw/user6.rules "$BK/"
ufw status verbose        > "$BK/status-before.txt" 2>&1 || true
iptables -S ufw-user-input > "$BK/chain-before.txt"  2>&1 || true

cat > "$BK/rollback.sh" <<EOF
#!/bin/bash
# Restores the ufw rule files captured at $(date -Is), then reloads.
set -e
cp -a "$BK/user.rules"  /etc/ufw/user.rules
cp -a "$BK/user6.rules" /etc/ufw/user6.rules
ufw reload
EOF
chmod +x "$BK/rollback.sh"
hdr "backup"
note "$BK  (rollback: sudo bash $BK/rollback.sh)"

# ------------------------------------------------------------------- watchdog
if (( WATCHDOG )); then
    setsid nohup bash -c '
        bk="$1"; t="$2"
        for (( i=0; i<t; i++ )); do
            [[ -f "$bk/confirmed" ]] && exit 0
            sleep 1
        done
        [[ -f "$bk/confirmed" ]] || bash "$bk/rollback.sh" >>"$bk/watchdog.log" 2>&1
    ' _ "$BK" "$TIMEOUT" >/dev/null 2>&1 < /dev/null &
    note "watchdog armed: rolls back in ${TIMEOUT}s unless confirmed"
fi

# ---------------------------------------------------------------------- apply
# Only ever inserts allows. A rule is deleted first solely when it already
# exists but is shadowed by a deny, i.e. when it is doing nothing today.
ensure_allow() {  # ensure_allow <src-ip> <port> <comment>
    local ip=$1 port=$2 comment=$3
    local before; before=$(verdict_for "$ip" "$port")
    if [[ $before == ALLOW ]]; then
        note "$ip:$port already allowed - left untouched"
        return
    fi
    ufw --force delete allow from "$ip" to any port "$port" proto tcp >/dev/null 2>&1 || true
    ufw insert 1 allow from "$ip" to any port "$port" proto tcp comment "$comment" >/dev/null 2>&1 ||
        ufw allow from "$ip" to any port "$port" proto tcp comment "$comment" >/dev/null
    note "$ip:$port  $before -> $(verdict_for "$ip" "$port")"
}

hdr "applying"
# Inserted last = highest priority, so put the access paths above the API rule.
ensure_allow "$AGENT_IP" "$API_PORT" "agent llama api"
for p in ${WEB_PEERS//,/ }; do ensure_allow "$p" 443 "live panel session"; done
for p in ${SSH_PEERS//,/ }; do ensure_allow "$p" 22  "live ssh session"; done

# --------------------------------------------------------------------- verify
report

hdr "post-change service check"
if curl -fsS -m 10 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    note "llama-server still healthy on 127.0.0.1:$API_PORT"
else
    note "llama-server NOT answering - this is not caused by a firewall allow rule, check the unit"
fi
note "recent blocks for $AGENT_IP:"
grep -h "SRC=$AGENT_IP" /var/log/ufw.log 2>/dev/null | tail -3 | sed 's/^/    /' || note "    (none logged)"

# -------------------------------------------------------------------- confirm
if (( WATCHDOG )); then
    hdr "confirm"
    echo "Verify from agent now:"
    echo "    curl -m 10 http://$(hostname -I | awk '{print $1}'):$API_PORT/health"
    if (( ASSUME_YES )); then
        touch "$BK/confirmed"
        note "auto-confirmed (--yes); rules kept"
    else
        ans=""
        read -r -t "$TIMEOUT" -p "Type YES within ${TIMEOUT}s to keep these rules: " ans || true
        if [[ ${ans:-} == YES ]]; then
            touch "$BK/confirmed"
            note "confirmed; rules kept"
        else
            note "not confirmed - watchdog is rolling back to $BK"
            exit 1
        fi
    fi
fi

hdr "agent should point at"
echo "    http://$(hostname -I | awk '{print $1}'):$API_PORT/v1        (no API key)"
