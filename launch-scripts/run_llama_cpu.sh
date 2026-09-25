#!/bin/bash
# ============================================================================
# run_llama_cpu.sh
#
# CPU-only backend for LexiPanel. Reached by BACKEND=cpu, which the Vulkan script
# execs in place - same PID, so systemd's KillMode, Restart=on-failure and the
# start-limit throttle all still apply, and the failure counter has already
# been incremented before we get here.
#
# WHAT THIS IS FOR. It is not a serving path. A Xeon E-2224G is 4 cores with no
# SMT and roughly 40 GB/s of memory bandwidth; a 27B IQ4_XS needs ~15.7 GB read
# per token, so decode is bounded at low single-digit t/s no matter what is
# tuned here. What it buys is a server that still answers while the GPU is
# being reset, swapped, driver-debugged, or while a Vulkan/ROCm build is being
# bisected. Treat single-digit t/s as correct, not as a fault to chase.
#
# DELTAS vs the Vulkan script, and why:
#   1. NGL is forced to 0 and not read from params.env. A CPU build has no GPU
#      backend to offload to, and a non-zero value there would be silently
#      ignored - which reads as "it did what I asked" when it did not.
#   2. Its own RUNDIR (/dev/shm/llama_qwen38_cpu) and CPU_PORT override, so a
#      CPU instance can run ALONGSIDE the GPU one rather than fighting it for
#      port 8081 and the slot directory. Nothing here assumes it is alone.
#   3. Different defaults. On the GPU the KV cache lives on the card; here it
#      lives in the same 30 GB the weights are mmap'd from, so the Vulkan
#      script's ctx=245760 would be tens of GB of host KV. Defaults are sized
#      for a box that has to survive, not for depth.
#   4. No flash-attn forced on, no GPU readiness gate, no RADV ICD pinning, no
#      render group, no placement assertion - none of those have a meaning
#      without a card in the path.
#   5. The host-RAM watchdog is kept, and matters MORE here: on CPU the weights
#      are host memory by definition.
# ============================================================================
set -e

# --- BUILD SELECTION ---
# The panel writes panel/builds.env with a LIB_DIR_<BACKEND> per backend, so
# llama.cpp can be upgraded or rolled back from the UI instead of by editing
# this line. The value below is the fallback for a box with no builds.env.
# There is no shipped CPU build yet: install one from the panel's Builds tab
# (ubuntu-x64 asset), which is what writes LIB_DIR_CPU.
LIB_DIR="/home/admin/llama/b10883-cpu/llama-b10883"
_builds_env="/home/admin/panel/builds.env"
if [ -r "$_builds_env" ]; then
    # shellcheck disable=SC1090
    . "$_builds_env"
    _var="LIB_DIR_CPU"
    _sel="${!_var:-}"
    if [ -n "$_sel" ]; then
        if [ -x "$_sel/llama-server" ]; then
            LIB_DIR="$_sel"
        else
            echo "[WARN] builds.env selects $_sel but it has no llama-server;" \
                 "falling back to $LIB_DIR" >&2
        fi
    fi
fi
SERVER_PATH="$LIB_DIR/llama-server"

if [ ! -x "$SERVER_PATH" ]; then
    echo "[ABORT] no CPU llama-server at $SERVER_PATH"
    echo "        Install a cpu build: panel -> Builds -> Available upstream ->"
    echo "        the 'cpu' column, then 'Use for cpu'. That writes LIB_DIR_CPU"
    echo "        into $_builds_env."
    exit 1
fi

MODEL="/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
MMPROJ="/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
TEMPLATE_SRC="/home/admin/models/qwen3.8-safe-v2.jinja"

RUNDIR="/dev/shm/llama_qwen38_cpu"
SLOT_DIR="$RUNDIR/slots"
LOG_DIR="$RUNDIR/telemetry"
TEMPLATE="$RUNDIR/qwen3.8-safe-v2.jinja"

# ============================================================================
# TUNABLES - params.env (written by the panel) overrides every one of these.
# ============================================================================
BATCH=512                 # prefill on 4 cores gains nothing from a huge batch
UBATCH=128                # smaller micro-batch keeps the compute buffer small
CTX=32768                 # host-RAM KV; 245760 here would be tens of GB
NGL=0                     # forced below; listed only so params.env can be read
KV_TYPE=f16               # quantised KV on CPU costs more time than it saves
CACHE_RAM=2048
THREADS=4                 # E-2224G is 4c/4t
PORT=8081                 # CPU_PORT forces a side-by-side port
SPEC_TYPE=draft-mtp
SPEC_N_MAX=2
SPEC_DRAFT_MODEL=""
USE_MMPROJ=0              # BF16 projector on 4 cores is minutes per image
MMPROJ_OFFLOAD=0
CACHE_REUSE=256
RAM_FLOOR_MB=2048
N_PREDICT=16000
TEMP=1.0
TOP_P=0.95
TOP_K=20
MIN_P=0.0
REASONING_EFFORT=medium
REASONING_BUDGET=-1
PRESERVE_THINKING=true
MAX_TOOL_RESPONSE_CHARS=9000

PARAMS_ENV="/home/admin/panel/params.env"
if [ -r "$PARAMS_ENV" ]; then
    echo "[INFO] sourcing panel overrides from $PARAMS_ENV"
    # shellcheck disable=SC1090
    . "$PARAMS_ENV"
fi

# DELTA 1: not negotiable from params.env. A CPU build cannot offload.
if [ "${NGL:-0}" != "0" ]; then
    echo "[INFO] NGL=$NGL ignored - this is a CPU build with no GPU backend"
fi
NGL=0

# DELTA 2: a distinct port lets this run beside the GPU instance.
PORT=${CPU_PORT:-$PORT}

case "$(printf '%s' "$PRESERVE_THINKING" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) _pt=true ;;
    *)             _pt=false ;;
esac
TEMPLATE_KWARGS="{\"reasoning_effort\":\"$REASONING_EFFORT\",\"preserve_thinking\":$_pt,\"max_tool_response_chars\":$MAX_TOOL_RESPONSE_CHARS}"

export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

# ============================================================================
# RUNTIME TREE
# ============================================================================
mkdir -p "$SLOT_DIR" "$LOG_DIR"
cp -f "$TEMPLATE_SRC" "$TEMPLATE"
ACTUAL_SHA=$(sha256sum "$TEMPLATE" | cut -d' ' -f1)

echo "=============================================="
echo " run_llama_cpu  |  $(basename "$LIB_DIR")  |  CPU only"
echo " ctx=$CTX  np=1  kv=$KV_TYPE  spec=${SPEC_TYPE:-none} n=$SPEC_N_MAX  b=$BATCH ub=$UBATCH  threads=$THREADS"
echo " cache: ram=${CACHE_RAM}MiB reuse=${CACHE_REUSE}  ramfloor=${RAM_FLOOR_MB}MB  port=$PORT"
echo " template sha256 $ACTUAL_SHA"
echo " EXPECT single-digit t/s. This is a fallback path, not a serving path."
echo "=============================================="
echo

TELEMETRY_PID=""
if command -v python3 >/dev/null 2>&1; then
    python3 -m http.server "$(( PORT + 1 ))" --bind 127.0.0.1 --directory "$LOG_DIR" >/dev/null 2>&1 &
    TELEMETRY_PID=$!
fi

cleanup() {
    echo
    echo "[SHUTDOWN] preserving log, stopping telemetry..."
    if [ -f "$LOG_DIR/engine_debug.log" ]; then
        mkdir -p "$HOME/llama_logs"
        cp "$LOG_DIR/engine_debug.log" \
           "$HOME/llama_logs/engine_debug_cpu_$(date +%Y%m%d_%H%M%S).log"
    fi
    [ -n "$TELEMETRY_PID" ] && kill "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

MMPROJ_ARGS=()
if [ "$USE_MMPROJ" = "1" ]; then
    MMPROJ_ARGS=(--mmproj "$MMPROJ" --image-min-tokens 1024 --no-mmproj-offload)
fi

CACHE_REUSE_ARGS=()
[ "${CACHE_REUSE:-0}" -gt 0 ] && CACHE_REUSE_ARGS=(--cache-reuse "$CACHE_REUSE")

SPEC_ARGS=()
if [ -n "$SPEC_TYPE" ] && [ "$SPEC_TYPE" != "none" ]; then
    SPEC_ARGS=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-p-min 0)
    [ -n "$SPEC_DRAFT_MODEL" ] && SPEC_ARGS+=(--spec-draft-model "$SPEC_DRAFT_MODEL")
fi

"$SERVER_PATH" \
  -m "$MODEL" \
  "${MMPROJ_ARGS[@]}" \
  -c "$CTX" \
  -np 1 \
  --n-gpu-layers 0 \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
  --cache-ram "$CACHE_RAM" \
  "${CACHE_REUSE_ARGS[@]}" \
  --jinja \
  --chat-template-file "$TEMPLATE" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --reasoning-preserve \
  --reasoning on \
  --reasoning-format deepseek \
  --reasoning-budget "$REASONING_BUDGET" \
  "${SPEC_ARGS[@]}" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "$MIN_P" \
  --metrics \
  --no-warmup \
  --host 0.0.0.0 \
  --port "$PORT" \
  --slot-save-path "$SLOT_DIR" \
  --log-file "$LOG_DIR/engine_debug.log" \
  --batch-size "$BATCH" \
  --ubatch-size "$UBATCH" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --n-predict "$N_PREDICT" \
  --cont-batching 2> >(tee "$LOG_DIR/load_stderr.log" >&2) &

LLAMA_PID=$!

# ============================================================================
# FAILURE COUNTER RESET. The Vulkan script incremented it before dispatching
# here, so if this start proves itself the counter has to be cleared from here
# or the next start drops a tier for no reason.
# ============================================================================
FAIL_FILE="/home/admin/panel/.launch-fails"
LAUNCH_OK_SECONDS=120
(
  _t0=$(date +%s)
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
      if [ "$(( $(date +%s) - _t0 ))" -ge "$LAUNCH_OK_SECONDS" ]; then
          echo 0 > "$FAIL_FILE" 2>/dev/null || true
          echo "[FALLBACK] up ${LAUNCH_OK_SECONDS}s on cpu backend - failure counter reset"
          break
      fi
      sleep 5
  done
) &
TIER_PID=$!

# ============================================================================
# HOST-RAM WATCHDOG. Same contract as the Vulkan script. It matters more here:
# there is no GTT eviction to blame, but the weights ARE host memory, so an
# oversized CACHE_RAM plus the KV cache walks the box into the same swap-thrash
# livelock with no log.
# ============================================================================
(
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
      if [ "${avail:-999999}" -lt "$RAM_FLOOR_MB" ]; then
          echo "[WATCHDOG] MemAvailable ${avail}MB < ${RAM_FLOOR_MB}MB - killing llama-server before the host locks up" >&2
          kill -9 "$LLAMA_PID" 2>/dev/null
          break
      fi
      sleep 2
  done
) &
WATCHDOG_PID=$!

# Backend banner, same marker the panel's log classifier looks for.
( sleep 8
  printf '### LexiPanel-engine BACKEND=cpu BUILD=%s CTX=%s KV=%s\n' \
         "$(basename "$LIB_DIR")" "$CTX" "$KV_TYPE" >> "$LOG_DIR/engine_debug.log" 2>/dev/null || true
) &
BANNER_PID=$!

wait "$LLAMA_PID"
kill "$WATCHDOG_PID" "$TIER_PID" "$BANNER_PID" 2>/dev/null || true
