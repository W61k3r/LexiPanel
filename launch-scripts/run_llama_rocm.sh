#!/bin/bash
# ============================================================================
# run_llama_rocm.sh
# Qwen3.8-27B-Uncensored-HauhauCS-Aggressive IQ4_XS (native embedded MTP)
# ROCm / HIP (gfx1100) | 7900 XTX 24G | LexiPanel | HEADLESS
# ----------------------------------------------------------------------------
# ROCm sibling of run_llama_vulkan.sh, created 2026-09-04. That script
# stays the production path until this one is benchmarked and shown to win.
# Read its header first: the model facts, the MTP rule (DELTA 1b), the
# params.env precedence and the GTT/eviction post-mortem (DELTA 8) all apply
# here unchanged and are not repeated.
#
# BUILD: llama.cpp b10798, prebuilt ubuntu-rocm-10.0-x64.
#   /home/admin/llama/b10798-rocm/llama-b10798
#   libggml-hip.so is 985 MB of fat-binary GPU kernels; gfx1100 confirmed
#   present (strings | grep gfx). Do not "clean up" that file.
#
# RUNTIME DEPENDENCY - this build does NOT bundle the ROCm runtime:
#   libamdhip64.so.7  <- libamdhip64-7
#   librocblas.so.5   <- librocblas5
#   libhipblas.so.3   <- libhipblas3
#   Stock Ubuntu 26.04 archive carries all three at 7.1.0; no AMD repo needed.
#   Despite the "rocm-10.0" asset name, the sonames it links against are the
#   ones Ubuntu's ROCm 7.1 packages provide. Install (needs root):
#     sudo apt install -y libamdhip64-7 librocblas5 libhipblas3
#   ~260 MB down / ~760 MB installed, 9 packages, runtime only - no compiler,
#   no SDK. The preflight below fails loudly if they are missing.
#
# DELTAS vs the Vulkan script (each one deliberate):
#   R1. NO Vulkan ICD pinning. VK_DRIVER_FILES / VK_ICD_FILENAMES are
#       meaningless here. The Intel UHD P630 cannot enumerate under HSA at
#       all, so Vulkan trap 4 does not apply. HIP_VISIBLE_DEVICES=0 still
#       pins explicitly rather than relying on ordering.
#   R2. NO GGML_VK_ALLOW_SYSMEM_FALLBACK. The HIP analogue of that trap is
#       GGML_CUDA_ENABLE_UNIFIED_MEMORY: set it and HIP will silently back
#       device allocations with host memory over PCIe, which is the same
#       ~20x silent slowdown by another name. It is opt-in, so the fix is to
#       ensure it is NOT set. Explicitly unset below - do not "helpfully"
#       add it.
#   R3. PORT 8083, RUNDIR /dev/shm/llama_qwen38_rocm. Deliberately distinct
#       so this can run for comparison without touching the production
#       server on 8081. Never run both against the GPU at once: two 15.7 GB
#       loads is the config that hard-locked this box twice.
#   R4. NOT wired to systemd. Manual/benchmark use until it earns promotion.
#   R5. Tunables are sourced from the SAME panel params.env as the Vulkan
#       script, so a backend comparison is apples-to-apples by construction.
#       params-rocm.env, if present, is sourced after it for ROCm-only
#       overrides.
# ============================================================================
set -e

# --- BUILD SELECTION ---
# The panel writes panel/builds.env with a LIB_DIR_<BACKEND> per backend, so
# llama.cpp can be upgraded or rolled back from the UI instead of by editing
# this line. The value below is the fallback for a box with no builds.env, so
# this script still runs standalone exactly as it always did.
LIB_DIR="/home/admin/llama/b10798-rocm/llama-b10798"
_builds_env="/home/admin/panel/builds.env"
if [ -r "$_builds_env" ]; then
    # shellcheck disable=SC1090
    . "$_builds_env"
    _var="LIB_DIR_ROCM"
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

MODEL="/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
MMPROJ="/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
TEMPLATE_SRC="/home/admin/models/qwen3.8-safe-v2.jinja"
TEMPLATE_SHA256="4ed3960ba9caa33352f417bc6ac2f6e8358c76b4cbdbced9c59e9e16909f794b"

RUNDIR="/dev/shm/llama_qwen38_rocm"
SLOT_DIR="$RUNDIR/slots"
LOG_DIR="$RUNDIR/telemetry"
TEMPLATE="$RUNDIR/qwen3.8-safe-v2.jinja"

# ============================================================================
# TUNABLES - defaults mirror the Vulkan script so the comparison is fair.
# params.env (panel) overrides these; params-rocm.env overrides that.
# ============================================================================
BATCH=2048
UBATCH=512
CTX=244736
NGL=99
KV_TYPE=q5_1
CACHE_RAM=20000
THREADS=4
PORT=8081                 # params.env overrides; ROCM_PORT forces a side-by-side port
SPEC_TYPE=draft-mtp
SPEC_N_MAX=2
SPEC_DRAFT_MODEL=""
USE_MMPROJ=1
MMPROJ_OFFLOAD=0          # 0 => --no-mmproj-offload, see Vulkan DELTA 8
CACHE_REUSE=256
RAM_FLOOR_MB=512
N_PREDICT=16000
TEMP=1.0
TOP_P=0.95
TOP_K=20
MIN_P=0.0
REASONING_EFFORT=medium
REASONING_BUDGET=-1
PRESERVE_THINKING=true
MAX_TOOL_RESPONSE_CHARS=9000

for _pe in /home/admin/panel/params.env /home/admin/panel/params-rocm.env; do
    if [ -f "$_pe" ]; then
        echo "[INFO] sourcing overrides from $_pe"
        # shellcheck disable=SC1090
        . "$_pe"
    fi
done
# PORT comes from params.env like everything else. When BACKEND=rocm this
# script IS production - the systemd unit dispatches to it - so it has to bind
# the port clients already point at. Set ROCM_PORT explicitly only for the rare
# side-by-side comparison run, which also needs the Vulkan server stopped.
PORT=${ROCM_PORT:-$PORT}

case "$(printf '%s' "$PRESERVE_THINKING" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) _pt=true ;;
    *)             _pt=false ;;
esac
TEMPLATE_KWARGS="{\"reasoning_effort\":\"$REASONING_EFFORT\",\"preserve_thinking\":$_pt,\"max_tool_response_chars\":$MAX_TOOL_RESPONSE_CHARS}"

# Same guard as the Vulkan script: a full-size "draft" loads a second copy.
if [ -n "$SPEC_DRAFT_MODEL" ]; then
    _dsz=$(stat -c%s "$SPEC_DRAFT_MODEL" 2>/dev/null || echo 0)
    if [ "$_dsz" -gt 4000000000 ]; then
        echo "[ABORT] SPEC_DRAFT_MODEL is $(( _dsz/1024/1024 ))MB - a full-size target,"
        echo "        not a draft. Loads a SECOND full copy and OOMs the host."
        exit 1
    fi
fi

# ============================================================================
# Environment. See DELTA R2 - unified memory must stay OFF.
# ============================================================================
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# --- HIP/ROCr tuning driven from params.env ---------------------------------
# The ROCm build has an IDENTICAL CLI flag surface to the Vulkan build (checked
# by diffing --help), so every backend-specific knob is an environment
# variable. These names were extracted from libggml-hip.so and
# libhsa-runtime64.so on this box rather than taken from documentation.
#
# Empty means LEAVE UNSET, which is stock and is not the same as "0": some of
# these are tested with getenv() != NULL, so exporting "0" would ENABLE them.
# That is why this exports only non-empty values, and why
# GGML_CUDA_ENABLE_UNIFIED_MEMORY is unset above rather than set to 0.
for _v in GGML_CUDA_REGISTER_HOST GGML_CUDA_NO_PINNED GGML_CUDA_DISABLE_GRAPHS \
          GGML_CUDA_DISABLE_FUSION GGML_CUDA_GRAPH_OPT GGML_CUDA_DEVICES \
          GGML_CUDA_ENABLE_UNIFIED_MEMORY HSA_ENABLE_SDMA; do
    eval "_val=\${$_v:-}"
    if [ -n "$_val" ]; then
        export "$_v=$_val"
        echo "[HIP] $_v=$_val"
    fi
done
export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

# --- render group, same reasoning as the Vulkan script -----------------------
if ! id -nG | tr ' ' '\n' | grep -qx render; then
    if [ -z "${_SG_RENDER_REEXEC:-}" ] && getent group render | tr ',:' '\n' | grep -qx "$(id -un)"; then
        echo "[INFO] re-exec under 'sg render'"
        exec sg render -c "_SG_RENDER_REEXEC=1 $(printf '%q ' "$0" "$@")"
    fi
    echo "[ABORT] $(id -un) is not in the 'render' group"
    exit 1
fi

# ============================================================================
# Preflight
# ============================================================================
[ ! -x "$SERVER_PATH" ] && { echo "[ABORT] server missing: $SERVER_PATH"; exit 1; }
[ ! -f "$MODEL" ]       && { echo "[ABORT] model missing: $MODEL"; exit 1; }
[ ! -f "$TEMPLATE_SRC" ] && { echo "[ABORT] template missing: $TEMPLATE_SRC"; exit 1; }
if [ "$USE_MMPROJ" = "1" ] && [ ! -f "$MMPROJ" ]; then
    echo "[ABORT] mmproj missing: $MMPROJ"; exit 1
fi

# ROCm runtime. This is the failure everyone hits first, so name the fix.
_missing=$(ldd "$LIB_DIR/libggml-hip.so" 2>/dev/null | awk '/not found/{print $1}')
if [ -n "$_missing" ]; then
    echo "[ABORT] the ROCm runtime is not installed. Missing:"
    echo "$_missing" | sed 's/^/          /'
    echo "        fix (needs root, ~260MB download, stock Ubuntu archive):"
    echo "          sudo apt install -y libamdhip64-7 librocblas5 libhipblas3"
    exit 1
fi

# --- GPU readiness gate. Resolve by DRIVER, never by card number. ------------
AMD_DEV=""
for _c in /sys/class/drm/card*/device; do
    [ -e "$_c/uevent" ] || continue
    if grep -qx "DRIVER=amdgpu" "$_c/uevent" 2>/dev/null; then AMD_DEV="$_c"; break; fi
done
[ -z "$AMD_DEV" ] && { echo "[ABORT] no amdgpu DRM device present"; exit 1; }

_gpu_wait=0
while :; do
    _vram_total=$(cat "$AMD_DEV/mem_info_vram_total" 2>/dev/null || echo 0)
    _smu_ok=0
    if [ -r "$AMD_DEV/pp_dpm_sclk" ] && [ -n "$(ls "$AMD_DEV/hwmon" 2>/dev/null)" ]; then _smu_ok=1; fi
    if [ "${_vram_total:-0}" -ge 25000000000 ] && [ "$_smu_ok" = "1" ]; then break; fi
    [ "$_gpu_wait" -ge 60 ] && { echo "[ABORT] amdgpu not ready after 60s"; exit 1; }
    [ "$_gpu_wait" -eq 0 ] && echo "[WAIT]  amdgpu still initialising"
    sleep 1; _gpu_wait=$((_gpu_wait + 1))
done

# Refuse to start on top of another server. Two 15.7 GB loads = host OOM.
# Check the port we are ABOUT to bind, not a hardcoded one - under the
# BACKEND=rocm dispatch that port is 8081, the same one Vulkan uses.
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "[ABORT] something is already listening on ${PORT}."
    echo "        Two full model loads will OOM this host. Stop it first:"
    echo "          sudo systemctl stop LexiPanel-llama"
    exit 1
fi

mkdir -p "$SLOT_DIR" "$LOG_DIR"
cp "$TEMPLATE_SRC" "$TEMPLATE"
ACTUAL_SHA=$(sha256sum "$TEMPLATE" | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$TEMPLATE_SHA256" ]; then
    echo "[ABORT] template sha256 mismatch"
    echo "  expected: $TEMPLATE_SHA256"
    echo "  actual:   $ACTUAL_SHA"
    exit 1
fi

echo "[CHECK] backend : ROCm/HIP b10798 ($LIB_DIR)"
echo "[CHECK] amdgpu  : $AMD_DEV ready after ${_gpu_wait}s, $(( _vram_total / 1048576 )) MiB VRAM"
echo "[CHECK] server build:"
"$SERVER_PATH" --version 2>&1 | head -3
echo "[CHECK] HIP devices (expect ONE, gfx1100):"
"$SERVER_PATH" --list-devices 2>&1 | sed 's/^/         /'
echo

TELEMETRY_PID=""
if command -v python3 >/dev/null; then
    python3 -m http.server "$(( PORT + 1 ))" --bind 127.0.0.1 --directory "$LOG_DIR" >/dev/null 2>&1 &
    TELEMETRY_PID=$!
fi

cleanup() {
    echo
    echo "[SHUTDOWN] preserving log, stopping telemetry..."
    if [ -f "$LOG_DIR/engine_debug.log" ]; then
        mkdir -p "$HOME/llama_logs"
        cp "$LOG_DIR/engine_debug.log" \
           "$HOME/llama_logs/engine_debug_rocm_$(date +%Y%m%d_%H%M%S).log"
    fi
    [ -n "$TELEMETRY_PID" ] && kill "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "=============================================="
echo " run_llama_rocm  |  b10798  |  ROCm/HIP gfx1100"
echo " ctx=$CTX  np=1  kv=$KV_TYPE  spec=${SPEC_TYPE:-none} n=$SPEC_N_MAX  b=$BATCH ub=$UBATCH  threads=$THREADS"
echo " kwargs=reasoning_effort:$REASONING_EFFORT preserve_thinking:$_pt tool_resp:$MAX_TOOL_RESPONSE_CHARS"
echo " vision=$( [ "$USE_MMPROJ" = 1 ] && echo "ENABLED (offload=$( [ "$MMPROJ_OFFLOAD" = 1 ] && echo on-card || echo OFF-CARD ))" || echo disabled )"
echo " cache: ram=${CACHE_RAM}MiB reuse=${CACHE_REUSE}  ramfloor=${RAM_FLOOR_MB}MB  port=$PORT"
echo "=============================================="
echo

MMPROJ_ARGS=()
if [ "$USE_MMPROJ" = "1" ]; then
    MMPROJ_ARGS=(--mmproj "$MMPROJ" --image-min-tokens 1024)
    [ "$MMPROJ_OFFLOAD" = "1" ] || MMPROJ_ARGS+=(--no-mmproj-offload)
fi

CACHE_REUSE_ARGS=()
[ "${CACHE_REUSE:-0}" -gt 0 ] && CACHE_REUSE_ARGS=(--cache-reuse "$CACHE_REUSE")

SPEC_ARGS=()
if [ -n "$SPEC_TYPE" ] && [ "$SPEC_TYPE" != "none" ]; then
    SPEC_ARGS=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-p-min 0)
    [ -n "$SPEC_DRAFT_MODEL" ] && SPEC_ARGS+=(--spec-draft-model "$SPEC_DRAFT_MODEL" --spec-draft-ngl all)
fi

# stderr is TEE'd, not swallowed: the load-time allocation report (KV size,
# per-buffer sizes) goes to stderr and is NOT captured by --log-file. That gap
# is why the Vulkan side has no recorded memory breakdown to size against.
"$SERVER_PATH" \
  -m "$MODEL" \
  "${MMPROJ_ARGS[@]}" \
  -c "$CTX" \
  -np 1 \
  --n-gpu-layers "$NGL" \
  --flash-attn on \
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
# BACKEND BANNER. The engine log records nothing about which backend produced
# it, so a directory of archives is uninterpretable and anything fitted across
# them silently blends Vulkan and ROCm samples into one meaningless curve.
# llama-server truncates --log-file on open, so this has to be appended after
# it starts rather than pre-written. A .meta sidecar carries the same facts in
# a form that survives log rotation.
# ============================================================================
(
  for _i in $(seq 1 180); do
      kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
      [ -s "$LOG_DIR/engine_debug.log" ] && break
      sleep 1
  done
  kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
  _meta="### LexiPanel-engine BACKEND=rocm BUILD=b10798 CTX=${CTX} KV=${KV_TYPE}"
  _meta="$_meta B=${BATCH} UB=${UBATCH} SPEC=${SPEC_TYPE:-none} N_MAX=${SPEC_N_MAX}"
  _meta="$_meta MMPROJ=${USE_MMPROJ} MMPROJ_OFFLOAD=${MMPROJ_OFFLOAD} CACHE_RAM=${CACHE_RAM}"
  _meta="$_meta CACHE_REUSE=${CACHE_REUSE} STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$_meta" >> "$LOG_DIR/engine_debug.log"
  echo "$_meta"  > "$LOG_DIR/engine_debug.log.meta"
) &
BANNER_PID=$!

# --- placement assertion, same contract as the Vulkan script -----------------
PLACEMENT_MIN_VRAM_MB=18000
PLACEMENT_GTT_FAULT_MB=10000
PLACEMENT_DEADLINE_S=420
(
  _t0=$(date +%s); _verdict="timeout"
  while :; do
      kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
      _v=$(( $(cat "$AMD_DEV/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))
      _g=$(( $(cat "$AMD_DEV/mem_info_gtt_used"  2>/dev/null || echo 0) / 1048576 ))
      [ "$_v" -ge "$PLACEMENT_MIN_VRAM_MB" ] && { _verdict="ok"; break; }
      [ "$_g" -ge "$PLACEMENT_GTT_FAULT_MB" ] && { _verdict="fault"; break; }
      [ "$(( $(date +%s) - _t0 ))" -ge "$PLACEMENT_DEADLINE_S" ] && break
      sleep 2
  done
  kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
  if [ "$_verdict" = "ok" ]; then
      echo "[PLACEMENT] OK: VRAM ${_v} MiB / GTT ${_g} MiB"
  else
      echo "[PLACEMENT] FAULT (${_verdict}): VRAM ${_v} MiB / GTT ${_g} MiB" >&2
      kill -9 "$LLAMA_PID" 2>/dev/null || true
  fi
) &
PLACEMENT_PID=$!

# --- host-RAM watchdog. See the Vulkan script's HOST-RAM WATCHDOG block: with
#     it disarmed this box took a real global kernel OOM on 2026-09-04.
(
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
      if [ "${RAM_FLOOR_MB:-0}" -gt 0 ] && [ "${avail:-999999}" -lt "$RAM_FLOOR_MB" ]; then
          echo "[WATCHDOG] MemAvailable ${avail}MB < ${RAM_FLOOR_MB}MB - killing llama-server" >&2
          kill -9 "$LLAMA_PID" 2>/dev/null
          break
      fi
      sleep 2
  done
) &
WATCHDOG_PID=$!

wait "$LLAMA_PID"
kill "$WATCHDOG_PID" "$PLACEMENT_PID" "$BANNER_PID" 2>/dev/null || true
