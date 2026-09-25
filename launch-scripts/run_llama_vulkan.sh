#!/bin/bash
# ============================================================================
# run_llama_vulkan.sh
# Qwen3.8-27B-Uncensored-HauhauCS-Aggressive IQ4_XS (native embedded MTP)
# VULKAN (RADV / NAVI31) | 7900 XTX 24G | ctx=131072 | np=1
# HOST: LexiPanel - Ubuntu 26.04, Xeon E-2224G (4c/4t), 30G RAM, HEADLESS
# ----------------------------------------------------------------------------
# This is the LexiPanel port of the desktop
# llama_qwen3.8-27b-hauhau-aggressive_iq4xs_ctx131072_np1_vulkan_v4.sh
# Read that file's RESULT LOG for the reasoning behind the template, the KV
# type, the MTP setup and the reasoning-effort kwargs. Only the DELTAS from
# that script are documented here.
#
# DELTAS vs desktop v4 (each one deliberate):
#   1. BINARY: prebuilt llama.cpp b10766 ubuntu-vulkan-x64 release, not a
#      local b10675 build. "latest llama vulkan" per operator request.
#      Path: $LIB_DIR.
#   1b. MTP: EMBEDDED NextN, ON. Read this before touching the spec flags.
#
#      THE RULE: with --spec-type draft-mtp you pass NO --spec-draft-model.
#      The model card means it literally - "use any target GGUF BY ITSELF".
#      Correct form logs:
#        common_speculative_init_result: creating MTP draft context
#                                        against the target model
#      and costs +370 MiB (19.22 -> 19.59 GiB device, measured).
#
#      2026-09-02 POST-MORTEM - host OOM, hard lock, box power-cycled TWICE.
#      The first LexiPanel attempt set --spec-draft-model = $MODEL, reading
#      "by itself" as "pass the target as its own draft". It is not. That
#      logs instead:
#        common_speculative_init_result: loading draft model '...IQ4_XS.gguf'
#      a genuine SECOND full 15.7G load, and --spec-draft-ngl defaults to
#      'auto'. Two full 27B copies plus two 131072-token KV caches against
#      30G host / 24.5G VRAM. common_fit_params had already given up
#      ("n_gpu_layers already set by user to 99, abort") so nothing capped
#      the overcommit. The server reached "listening", then the box died.
#
#      DIAGNOSTIC: grep the load log for "loading draft model". If it is
#      there, a second copy is being loaded and the config is dangerous.
#      "creating MTP draft context" is the safe one.
#
#      --spec-draft-model is for a SMALL dedicated draft only. The desktop
#      v4 used its own ~1G mtp-Qwen3.8-27B-Q4_0.gguf, which is why it never
#      hit this. The HauhauCS FastMTP sidecar
#      (Qwen3.8-...-FastMTP-32K.gguf, 862M, downloaded and sha-verified) is
#      the other legitimate value, but it needs a source build with
#      HauhauCS-FastMTP-llama.cpp.patch at pinned commit
#      4df29be4f4c3673f428170fda944a5b19f743bb8. On this stock b10766 binary
#      it fails with "expected 5120, 248320, got 5120, 32768".
#      Per the model card FastMTP is worth a further +21.7% document TG /
#      +9.3% reasoning TG over embedded MTP at depth 3, on IQ4_XS.
#   2. THREADS 20 -> 4. E-2224G is 4c/4t. --threads / --threads-batch 4.
#   3. BATCH and UBATCH both 512 (operator instruction, 2026-09-02).
#      Desktop v4 ran -b 2048 with the -ub 512 default. -b 512 here also
#      shrinks the prefill compute buffer, which is the allocation that
#      triggered desktop ITEM 6 (device-lost on deep-context prefill).
#   4. RAMDISK: no sudo on this box. $RUNDIR is /dev/shm (tmpfs, 16G, already
#      mounted, world-writable per-user). Slots + telemetry + template live
#      there. Nothing in this script needs root.
#   5. --cache-ram 16384 -> 8192. 30G box, model mmap is ~16G.
#   6. VULKAN DEVICE: this box has BOTH an Intel UHD P630 iGPU (ANV) and the
#      7900 XTX (RADV). VK_DRIVER_FILES / VK_ICD_FILENAMES are pinned to the
#      radeon ICD so the Intel device is not even enumerated. Confirm with
#      the --list-devices output the script prints at startup.
#   7. TEMPLATE: same qwen3.8-safe-v2, shipped as an external file next to
#      the models (not embedded). sha256 pinned below. NOTE the pinned hash
#      is THIS COPY's hash, which is 3 bytes off the desktop's stated
#      27127-byte / 6004e953... copy (transcription drift, almost certainly
#      trailing whitespace). It parses under Jinja2; the mandatory check is
#      still the "failed to parse" grep on first run under minja.
#
#   8. 2026-09-04 STABILITY PASS, after a real kernel OOM killed the server.
#      Chain, measured not assumed: VRAM sat at 22,465 of 24,560 MiB (91.5%),
#      leaving ~2.0 GB. Transient spikes - the prefill compute buffer, and the
#      mmproj mtmd worst-case buffer the desktop v4 log measured at 884.62 MiB -
#      exceeded free VRAM, so amdgpu evicted weights into GTT. GTT is host RAM
#      pinned by the driver and shows up in neither ps RSS nor the cgroup, so
#      15.25 GB of the model was invisibly resident in RAM. That plus the anon
#      prompt cache took the box to a global OOM (7.8 GB of 8.19 GB swap used).
#      The desktop never saw this - its ITEM 1 records "Fully on-card. No GTT
#      streaming" - because it ran with headroom this box did not have.
#      Changes, all operator-directed:
#        --no-mmproj-offload   projector off-card, ~1.7 GB of VRAM back and the
#                              biggest transient spike removed. MMPROJ_OFFLOAD=0.
#        --batch-size 2048     b10766's own default; 512 was throttling prefill.
#                              -ub stays 512: ub drives the compute buffer and
#                              the desktop ITEM 6 device-lost risk, -b does not.
#        --cache-reuse 256     KV-shift reuse of a drifted prefix instead of a
#                              full re-prefill. Aimed at long agentic loops.
#        RAM_FLOOR_MB=512      watchdog re-armed, floor lowered from 2048.
#        KV stays q5_1, ctx stays 244736, cache-ram stays 20000.
#      MMPROJ_OFFLOAD and CACHE_REUSE are deliberately NOT params.env keys:
#      panel.py save_params() rewrites that file from its own DEFAULTS list and
#      would drop any key it does not know about.
#
# FIRST-RUN CHECKLIST (same spirit as desktop v1):
#   [ ] --list-devices shows exactly one device, "Radeon RX 7900 XTX (RADV
#       NAVI31)" or similar. If it shows an Intel device, the ICD pin failed.
#   [ ] load log: no "failed to parse" on the chat template (minja != jinja2)
#   [ ] load log: KV cache size and the DeltaNet recurrent-state size - read
#       the real numbers, the desktop hybrid-layer math is an estimate
#   [ ] a full ~131k-token prefill completes without
#       "ggml_vulkan: device lost" / "ErrorDeviceLost". If it does NOT:
#         dmesg | grep -iE "amdgpu|ring|reset|VM_L2|page fault|timeout" | tail -40
#         - VM_L2_PROTECTION_FAULT / page fault addr -> VRAM OOM at prefill
#           peak. Keep -ub 256, or drop -c to 98304.
#         - "ring ... timeout" -> raise amdgpu.lockup_timeout on the kernel
#           cmdline, do NOT shrink -ub.
#   [ ] MTP draft acceptance in the logs (desktop saw 0.819 @ n=2). If it is
#       well below ~0.6 the stock sidecar is not tracking this finetune.
# ============================================================================
set -e

# --- BUILD SELECTION ---
# The panel writes panel/builds.env with a LIB_DIR_<BACKEND> per backend, so
# llama.cpp can be upgraded or rolled back from the UI instead of by editing
# this line. The value below is the fallback for a box with no builds.env, so
# this script still runs standalone exactly as it always did.
LIB_DIR="/home/admin/llama/b10766/llama-b10766"
_builds_env="/home/admin/panel/builds.env"
if [ -r "$_builds_env" ]; then
    # shellcheck disable=SC1090
    . "$_builds_env"
    _var="LIB_DIR_VULKAN"
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
# DRAFT: intentionally unset. See DELTA 1b - pointing this at a full-size
# target OOM'd the host and hard-locked the box on 2026-09-02.
DRAFT=""
MMPROJ="/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

RUNDIR="/dev/shm/llama_qwen38"
SLOT_DIR="$RUNDIR/slots"
LOG_DIR="$RUNDIR/telemetry"
TEMPLATE_SRC="/home/admin/models/qwen3.8-safe-v2.jinja"
TEMPLATE="$RUNDIR/qwen3.8-safe-v2.jinja"
TEMPLATE_SHA256="4ed3960ba9caa33352f417bc6ac2f6e8358c76b4cbdbced9c59e9e16909f794b"

# --- Template kwargs are now BUILT from the tunables below, after params.env
#     is sourced, so the panel can drive them. See TEMPLATE_KWARGS further down.
#     Revert to template defaults by commenting the --chat-template-kwargs
#     line in the launch block, NOT by setting it to '{}'. ---

# ============================================================================
# TUNABLES. Defaults live here; the web panel overrides them by writing
# /home/admin/panel/params.env, which is sourced just below. That keeps ONE
# source of truth for launch logic (this script) while letting the panel drive
# the values. Delete params.env to fall back to these defaults.
# ============================================================================
BATCH=512                 # operator instruction 2026-09-02
UBATCH=512                # ditto
CTX=131072                # operator requirement
NGL=99
KV_TYPE=q8_0              # both K and V
CACHE_RAM=4096
THREADS=4                 # E-2224G is 4c/4t
PORT=8081
SPEC_TYPE=draft-mtp       # embedded NextN. NEVER set SPEC_DRAFT_MODEL to a
SPEC_N_MAX=2              # full-size target - see DELTA 1b.
SPEC_DRAFT_MODEL=""       # empty = embedded MTP. Small sidecar files only.
USE_MMPROJ=1
# Projector OFF-CARD (2026-09-04). --no-mmproj-offload keeps the ~848 MiB
# projector plus its ~885 MiB mtmd worst-case buffer out of VRAM. That buffer
# is a transient spike, and spikes are what force amdgpu to evict weights to
# GTT. Vision still works; image encode is slower. 1 = on-card (stock).
# NOTE: script-only. panel.py save_params() rewrites params.env from its own
# DEFAULTS list, so a key added there by hand is dropped on the next save.
MMPROJ_OFFLOAD=0
# KV-shift reuse of cached prefix instead of a full re-prefill when the prompt
# drifts. Needs prompt caching (--cache-ram) on. 0 = off. Script-only, as above.
CACHE_REUSE=256
# Host-RAM watchdog floor, MB. Kills llama-server if MemAvailable drops below
# it. 0 disables. See the HOST-RAM WATCHDOG block further down for why this is
# not optional on this box.
RAM_FLOOR_MB=512
N_PREDICT=16000

# --- Sampling. These are only DEFAULTS, applied to requests that do not carry
#     their own. An OpenAI-compatible client that sends temperature/top_p wins
#     over these. 1.0/0.95/20/0.0 is Qwen's recommended thinking-mode pairing.
TEMP=1.0
TOP_P=0.95
TOP_K=20
MIN_P=0.0

# --- Reasoning. WARNING: this template aliases 'high' to 'xhigh', its MAXIMUM,
#     and at that level injects a "think carefully, validate assumptions,
#     consider alternatives" instruction into every prompt. 'medium' injects
#     nothing (neutral); 'low' actively instructs brevity. Set 2026-09-03 to
#     medium - 'high' was making it think forever.
#     REASONING_BUDGET is a hard token cap on thinking; -1 = unrestricted.
#     PRESERVE_THINKING=true keeps reasoning in context across turns, which
#     grows the prompt every turn and is a main driver of the deep re-prefills.
REASONING_EFFORT=medium
REASONING_BUDGET=-1
PRESERVE_THINKING=true
MAX_TOOL_RESPONSE_CHARS=8000

PARAMS_ENV="/home/admin/panel/params.env"
if [ -f "$PARAMS_ENV" ]; then
    echo "[INFO] sourcing panel overrides from $PARAMS_ENV"
    # shellcheck disable=SC1090
    . "$PARAMS_ENV"
fi

# ============================================================================
# TIERED FALLBACK
# ----------------------------------------------------------------------------
# A config that will not load produces a restart loop, and systemd's
# StartLimitBurst=3 then parks the unit as 'failed' with no server at all -
# which is exactly what happened on 2026-09-04 when BACKEND was switched to a
# backend whose runtime was missing. Rather than fail three times identically,
# each retry drops to a more conservative tier. The three attempts systemd
# allows therefore become: your config, then safe, then minimal.
#
#   fails=0  ->  params.env, whatever you configured
#   fails=1  ->  params-tier-safe.env     ctx 131072 q5_1 cram 8192 ub 256
#   fails>=2 ->  params-tier-minimal.env  ctx 65536 q4_0 cram 4096 no vision
#
# The counter is incremented here and cleared by a watcher below once the
# server has stayed up past LAUNCH_OK_SECONDS, so an occasional crash after a
# long healthy run does not drag you down a tier.
# ============================================================================
FAIL_FILE="/home/admin/panel/.launch-fails"
LAUNCH_OK_SECONDS=120
_fails=$(cat "$FAIL_FILE" 2>/dev/null || echo 0)
case "$_fails" in ''|*[!0-9]*) _fails=0 ;; esac

LAUNCH_TIER="normal"
if [ "$_fails" -ge 2 ]; then
    LAUNCH_TIER="minimal"
elif [ "$_fails" -ge 1 ]; then
    LAUNCH_TIER="safe"
fi
if [ "$LAUNCH_TIER" != "normal" ]; then
    _tf="/home/admin/panel/params-tier-${LAUNCH_TIER}.env"
    if [ -f "$_tf" ]; then
        echo "[FALLBACK] ${_fails} consecutive failed starts - dropping to '${LAUNCH_TIER}' tier"
        echo "[FALLBACK] sourcing $_tf (overrides params.env)"
        # shellcheck disable=SC1090
        . "$_tf"
    else
        echo "[FALLBACK] wanted tier '${LAUNCH_TIER}' but $_tf is missing - keeping params.env"
    fi
fi
echo "$(( _fails + 1 ))" > "$FAIL_FILE" 2>/dev/null || true

# A device selection only the Vulkan path can honour must not be silently
# dropped by the ROCm/CPU hand-off below (see DEVICE SELECTION further down).
if [ -f /home/admin/panel/main-devices.json ] && [ "${BACKEND:-vulkan}" = "rocm" ]; then
    if ! /usr/bin/python3 /home/admin/panel/main_devices.py "$LIB_DIR" rocm >/dev/null; then
        echo "[ABORT] panel/main-devices.json cannot run on BACKEND=rocm (reasons above)"
        exit 1
    fi
fi

# ============================================================================
# BACKEND DISPATCH (2026-09-04). The systemd unit's ExecStart points here, so
# this script is the entry point for BOTH backends and hands off rather than
# needing a unit edit (which needs root on this box). BACKEND=rocm execs the
# ROCm script in place - same PID, so systemd's KillMode=control-group,
# Restart=on-failure and the start-limit throttle all still apply.
# ============================================================================
if [ "${BACKEND:-vulkan}" = "rocm" ]; then
    _rocm="/home/admin/llama/run_llama_rocm.sh"
    if [ ! -x "$_rocm" ]; then
        echo "[ABORT] BACKEND=rocm but $_rocm is missing or not executable"
        exit 1
    fi
    echo "[INFO] BACKEND=rocm - handing off to $_rocm"
    exec "$_rocm" "$@"
fi

if [ "${BACKEND:-vulkan}" = "cpu" ]; then
    _cpu="/home/admin/llama/run_llama_cpu.sh"
    if [ ! -x "$_cpu" ]; then
        echo "[ABORT] BACKEND=cpu but $_cpu is missing or not executable"
        exit 1
    fi
    echo "[INFO] BACKEND=cpu - handing off to $_cpu"
    exec "$_cpu" "$@"
fi

# An unrecognised BACKEND used to fall through and silently run Vulkan, so a
# typo served the wrong engine with no indication anywhere. Fail instead.
case "${BACKEND:-vulkan}" in
    vulkan) ;;
    *) echo "[ABORT] unknown BACKEND='${BACKEND}' - expected vulkan, rocm or cpu"
       exit 1 ;;
esac

# Built AFTER the panel overrides so the panel can drive reasoning effort.
# PRESERVE_THINKING must be a bare JSON boolean, so normalise whatever was set.
case "$(printf '%s' "$PRESERVE_THINKING" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) _pt=true ;;
    *)             _pt=false ;;
esac
TEMPLATE_KWARGS="{\"reasoning_effort\":\"$REASONING_EFFORT\",\"preserve_thinking\":$_pt,\"max_tool_response_chars\":$MAX_TOOL_RESPONSE_CHARS}"

# Guard the one setting that has hard-locked this box twice.
if [ -n "$SPEC_DRAFT_MODEL" ]; then
    _dsz=$(stat -c%s "$SPEC_DRAFT_MODEL" 2>/dev/null || echo 0)
    if [ "$_dsz" -gt 4000000000 ]; then
        echo "[ABORT] SPEC_DRAFT_MODEL is $(( _dsz/1024/1024 ))MB - that is a full-size"
        echo "        target, not a draft. This loads a SECOND full copy and OOMs the"
        echo "        host. See DELTA 1b. Use the embedded head (empty) or a sidecar."
        exit 1
    fi
fi

# ============================================================================
# Vulkan device pinning: RADV only, hide the Intel iGPU.
# ============================================================================
RADV_ICD=""
for p in /usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
         /usr/share/vulkan/icd.d/radeon_icd.json \
         /usr/share/vulkan/icd.d/radeon_icd.i686.json; do
    [ -f "$p" ] && RADV_ICD="$p" && break
done
if [ -z "$RADV_ICD" ]; then
    echo "[ABORT] RADV ICD not found in /usr/share/vulkan/icd.d/"
    echo "        install it:  sudo apt install -y mesa-vulkan-drivers libvulkan1 libgomp1 vulkan-tools"
    exit 1
fi

# --- GPU device-node access. RADV opens /dev/dri/renderD* directly; the
#     user must be in the 'render' group. usermod was done, but a login
#     session started before that still has stale credentials. If this
#     process lacks the group yet the passwd db grants it, re-exec once
#     under 'sg render' (passwordless for a real member). A full re-login
#     also fixes it permanently.
if ! id -nG | tr ' ' '\n' | grep -qx render; then
    if [ -z "${_SG_RENDER_REEXEC:-}" ] && getent group render | tr ',:' '\n' | grep -qx "$(id -un)"; then
        echo "[INFO] $(id -un) has 'render' in the group db but not in this session - re-exec under 'sg render'"
        exec sg render -c "_SG_RENDER_REEXEC=1 $(printf '%q ' "$0" "$@")"
    fi
    echo "[ABORT] $(id -un) is not in the 'render' group - RADV cannot open /dev/dri/renderD*"
    echo "        fix:   sudo usermod -aG render,video $(id -un)   then log out/in"
    exit 1
fi
export VK_DRIVER_FILES="$RADV_ICD"     # Vulkan loader >= 1.3.207
export VK_ICD_FILENAMES="$RADV_ICD"    # older loader fallback
export GGML_VK_VISIBLE_DEVICES=0

# --- DEVICE SELECTION (added 2026-09-17) ------------------------------------
# The panel can put main on any GPU, or several: it writes
# panel/main-devices.json. Without that file nothing below runs and main stays
# pinned to RADV / Vulkan0 exactly as above. With it, panel/main_devices.py
# resolves the selection through the same code every other instance's launch
# plan uses. If the selection cannot be pinned, ABORT - never quietly serve
# from a different card than the one the panel shows.
MAIN_DEVICES_FILE="/home/admin/panel/main-devices.json"
MAIN_AMD_ONLY=1
MAIN_DEVICE_NAMES="7900 XTX (default)"
DEVICE_ARGS=()
if [ -f "$MAIN_DEVICES_FILE" ]; then
    if ! _devenv=$(/usr/bin/python3 /home/admin/panel/main_devices.py "$LIB_DIR" vulkan); then
        echo "[ABORT] $MAIN_DEVICES_FILE selects devices this launch cannot pin (reasons above)."
        echo "        Fix the selection on the panel's Status tab, or delete the file for XTX-only."
        exit 1
    fi
    eval "$_devenv"
    export VK_DRIVER_FILES="$MAIN_VK_ICDS" VK_ICD_FILENAMES="$MAIN_VK_ICDS"
    if [ -n "$MAIN_VK_VISIBLE" ]; then
        export GGML_VK_VISIBLE_DEVICES="$MAIN_VK_VISIBLE"
    else
        unset GGML_VK_VISIBLE_DEVICES
    fi
    [ -n "$MAIN_VK_DEVICE" ] && DEVICE_ARGS=(--device "$MAIN_VK_DEVICE")
    echo "[DEVICES] $MAIN_DEVICE_NAMES"
fi

# Per-device placement. Empty means not passed, which is stock; these only
# matter once main spans more than one card or offloads to the CPU.
PLACEMENT_ARGS=()
[ -n "${SPLIT_MODE:-}" ]       && PLACEMENT_ARGS+=(--split-mode "$SPLIT_MODE")
[ -n "${TENSOR_SPLIT:-}" ]     && PLACEMENT_ARGS+=(--tensor-split "$TENSOR_SPLIT")
[ -n "${MAIN_GPU:-}" ]         && PLACEMENT_ARGS+=(--main-gpu "$MAIN_GPU")
[ -n "${OVERRIDE_TENSORS:-}" ] && PLACEMENT_ARGS+=(--override-tensor "$OVERRIDE_TENSORS")
export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

# --- THE PLACEMENT FIX. Do not remove. -------------------------------------
# ggml-vulkan defaults to allowing a silent fall back to system memory. On
# this box it took that path for the WHOLE model: measured 2026-09-02 at
# VRAM 26 MiB / GTT 15693 MiB, i.e. every weight served over PCIe, giving
# 2.79 t/s prefill and 5.86 t/s decode. No error, no warning - it just runs
# ~20x too slow. That is the same signature as the desktop script's v1
# "placement fault" (6163M VRAM vs 13629M GTT), which was never attributed.
# THIS is the attribution.
#
# Setting it to 0 forbids the fallback. Measured immediately after:
#   Total device: 18.35 GiB, total host: 149.24 MiB   (no mmproj)
# which fits 24560 MiB VRAM with room for the projector's ~1.7G.
#
# Side benefit: an over-budget config now FAILS LOUDLY at load instead of
# silently degrading, so raising -c or re-adding a draft model can no longer
# quietly turn into a GTT crawl.
export GGML_VK_ALLOW_SYSMEM_FALLBACK=0

# --- RADV tuning driven from params.env -------------------------------------
# Names taken from libggml-vulkan.so in build b10798. Same convention as the
# ROCm script: empty means LEAVE UNSET, which is stock and is NOT the same as
# "0" - several of these are tested with getenv() != NULL, so exporting "0"
# would ENABLE them. Only non-empty values are exported.
# ALLOW_SYSMEM_FALLBACK above is deliberately NOT in this list: it is a guard,
# not a tunable, and must stay 0. See the PLACEMENT FIX block.
for _v in GGML_VK_DISABLE_COOPMAT GGML_VK_DISABLE_COOPMAT2 GGML_VK_DISABLE_F16 \
          GGML_VK_DISABLE_BFLOAT16 GGML_VK_DISABLE_FUSION GGML_VK_DISABLE_GRAPH_OPTIMIZE \
          GGML_VK_DISABLE_MMVQ GGML_VK_FORCE_MMVQ GGML_VK_DISABLE_INTEGER_DOT_PRODUCT \
          GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM GGML_VK_PREFER_HOST_MEMORY \
          GGML_VK_ENABLE_MEMORY_PRIORITY GGML_VK_FORCE_MAX_ALLOCATION_SIZE \
          GGML_VK_SUBALLOCATION_BLOCK_SIZE GGML_VK_MAX_NODES_PER_SUBMIT \
          GGML_VK_DISABLE_ASYNC GGML_VK_MEMORY_LOGGER GGML_VK_PERF_LOGGER; do
    eval "_val=\${$_v:-}"
    if [ -n "$_val" ]; then
        export "$_v=$_val"
        echo "[VK] $_v=$_val"
    fi
done

# Uncomment to see every allocation and whether it landed on device or host:
# export GGML_VK_MEMORY_LOGGER=1

# ============================================================================
# Preflight
# ============================================================================
[ ! -x "$SERVER_PATH" ] && { echo "[ABORT] server missing/not executable: $SERVER_PATH"; exit 1; }
[ ! -f "$MODEL" ]       && { echo "[ABORT] model missing: $MODEL"; exit 1; }
if [ ! -f "$MMPROJ" ]; then
    echo "[ABORT] mmproj missing: $MMPROJ"
    ls -lh "$(dirname "$MMPROJ")" 2>/dev/null | grep -i mmproj || echo "  (no mmproj-* in that folder)"
    exit 1
fi
[ ! -f "$TEMPLATE_SRC" ] && { echo "[ABORT] template missing: $TEMPLATE_SRC"; exit 1; }

mkdir -p "$SLOT_DIR" "$LOG_DIR"
cp "$TEMPLATE_SRC" "$TEMPLATE"
ACTUAL_SHA=$(sha256sum "$TEMPLATE" | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$TEMPLATE_SHA256" ]; then
    echo "[ABORT] template sha256 mismatch"
    echo "  expected: $TEMPLATE_SHA256"
    echo "  actual:   $ACTUAL_SHA"
    exit 1
fi

FREE_SHM=$(df -m "$RUNDIR" | awk 'NR==2 {print $4}')
[ "${FREE_SHM:-0}" -lt 512 ] && { echo "[WARN] /dev/shm has only ${FREE_SHM}M free"; }

echo "[CHECK] model  : $(ls -lh "$MODEL" | awk '{print $5, $9}')"
echo "[CHECK] draft  : embedded NextN head, NO separate draft file (DELTA 1b)"
echo "[CHECK] mmproj : $(ls -lh "$MMPROJ" | awk '{print $5, $9}')"
echo "[CHECK] template: $(wc -c < "$TEMPLATE") bytes, sha256 OK ($(grep -o 'template_version = "[^"]*"' "$TEMPLATE"))"
echo "[CHECK] kwargs : $TEMPLATE_KWARGS"
echo "[CHECK] RADV ICD: $RADV_ICD"
echo "[CHECK] server build:"
"$SERVER_PATH" --version 2>&1 | head -3
# ============================================================================
# GPU READINESS GATE  (added 2026-09-04, after the post-power-loss cold boot)
# ----------------------------------------------------------------------------
# On the 2026-09-04 cold boot the model loaded entirely into GTT: VRAM 26 MiB /
# GTT 15,632 MiB - the classic placement fault - even though
# GGML_VK_ALLOW_SYSMEM_FALLBACK=0 was already set and in the process environ.
# A manual restart 3 minutes later, byte-identical config, put 23,038 MiB in
# VRAM. So the difference was never the config; it was WHEN the process started.
#
# Kernel timeline for that boot:
#     t=8.186s  amdgpu 0000:03:00.0 begins initialising
#     t=8.840s  LexiPanel-llama.service starts          <-- 0.65s into the probe
#     t=9.906s  --list-devices already reports "24560 MiB, 24533 MiB free"
#
# The unit's ExecStartPre gate was written to prevent exactly this, but it waits
# for /dev/dri/renderD128 - which on this box is the INTEL UHD P630 at
# 00:02.0. The 7900 XTX is renderD129 (03:00.0). i915 registers its node early,
# so that gate has been passing instantly on every boot since it was written.
#
# RADV will enumerate the card and report all 24560 MiB free while the kernel
# is still bringing the VRAM manager up, and will then place every buffer in
# GTT. GGML_VK_ALLOW_SYSMEM_FALLBACK governs ggml's own fallback decision for an
# over-budget config. It does not govern where the KERNEL puts a buffer object
# when VRAM is not ready yet. That is why the guard did not fire.
#
# Resolve the card by DRIVER, never by card number: numbering moves across
# boots, and card0 does not exist on this box at all any more (it was the
# simple-framebuffer handed off to amdgpu; /dev/dri/by-path still has a dangling
# card0 symlink for it).
# ============================================================================
AMD_DEV=""
for _c in /sys/class/drm/card*/device; do
    [ -e "$_c/uevent" ] || continue
    if grep -qx "DRIVER=amdgpu" "$_c/uevent" 2>/dev/null; then AMD_DEV="$_c"; break; fi
done
if [ -z "$AMD_DEV" ]; then
    echo "[ABORT] no amdgpu DRM device present - the 7900 XTX did not probe"
    exit 1
fi

_gpu_wait=0
while :; do
    _vram_total=$(cat "$AMD_DEV/mem_info_vram_total" 2>/dev/null || echo 0)
    _smu_ok=0
    if [ -r "$AMD_DEV/pp_dpm_sclk" ] && [ -n "$(ls "$AMD_DEV/hwmon" 2>/dev/null)" ]; then _smu_ok=1; fi
    if [ "${_vram_total:-0}" -ge 25000000000 ] && [ "$_smu_ok" = "1" ]; then break; fi
    if [ "$_gpu_wait" -ge 60 ]; then
        echo "[ABORT] amdgpu not ready after 60s (vram_total=${_vram_total} smu_ok=${_smu_ok})"
        exit 1
    fi
    if [ "$_gpu_wait" -eq 0 ]; then
        echo "[WAIT]  amdgpu still initialising - holding the launch until VRAM + SMU are up"
    fi
    sleep 1
    _gpu_wait=$((_gpu_wait + 1))
done

# Resizable BAR has to survive the BIOS. This boot came up with the RTC reading
# 2025-09-15, i.e. the CMOS lost its contents - so BIOS defaults are in play and
# 'Above 4G Decoding' can come back off. If it ever does, visible VRAM collapses
# to 256 MiB and NOTHING can keep the model on the card. Fail here, loudly,
# rather than 20x slower and silently.
_vis_total=$(cat "$AMD_DEV/mem_info_vis_vram_total" 2>/dev/null || echo 0)
if [ "$_vis_total" -lt "$(( _vram_total / 2 ))" ]; then
    echo "[ABORT] Resizable BAR looks DISABLED: visible VRAM $(( _vis_total / 1048576 )) MiB of $(( _vram_total / 1048576 )) MiB total."
    echo "        Re-enable 'Above 4G Decoding' and 'Re-Size BAR Support' in BIOS, then reboot."
    exit 1
fi

echo "[CHECK] amdgpu : $AMD_DEV ready after ${_gpu_wait}s"
echo "[CHECK] VRAM   : $(( _vram_total / 1048576 )) MiB total, $(( _vis_total / 1048576 )) MiB host-visible (ReBAR OK)"

echo "[CHECK] Vulkan devices (expect ONE, the 7900 XTX / RADV NAVI31):"
"$SERVER_PATH" --list-devices 2>&1 | sed 's/^/         /'
echo "[CHECK] prior GPU resets since boot (want none):"
dmesg 2>/dev/null | grep -iE "amdgpu.*(reset|timeout|VM_L2|page fault)" | tail -5 || echo "         (none, or dmesg needs privilege)"
echo

# ============================================================================
# Optional telemetry file server (llama --log-file target, browsable :8082)
# ============================================================================
TELEMETRY_PID=""
if command -v python3 >/dev/null; then
    python3 -m http.server 8082 --bind 127.0.0.1 --directory "$LOG_DIR" >/dev/null 2>&1 &
    TELEMETRY_PID=$!
fi

cleanup() {
    echo
    echo "[SHUTDOWN] preserving log, stopping telemetry..."
    if [ -f "$LOG_DIR/engine_debug.log" ]; then
        mkdir -p "$HOME/llama_logs"
        cp "$LOG_DIR/engine_debug.log" \
           "$HOME/llama_logs/engine_debug_vulkan_$(date +%Y%m%d_%H%M%S).log"
        [ -f "$LOG_DIR/engine_debug.log.meta" ] && cp "$LOG_DIR/engine_debug.log.meta" \
           "$HOME/llama_logs/engine_debug_vulkan_$(date +%Y%m%d_%H%M%S).log.meta" 2>/dev/null
        echo "[SHUTDOWN] log -> $HOME/llama_logs/"
    fi
    [ -n "$TELEMETRY_PID" ] && kill "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "=============================================="
echo " run_llama_vulkan  |  b10766  |  RADV NAVI31"
echo " ctx=$CTX  np=1  kv=$KV_TYPE  spec=${SPEC_TYPE:-none} n=$SPEC_N_MAX  b=$BATCH ub=$UBATCH  threads=$THREADS"
echo " template=qwen3.8-safe-v2 (sha pinned)"
echo " kwargs=reasoning_effort:$REASONING_EFFORT preserve_thinking:$_pt tool_resp:$MAX_TOOL_RESPONSE_CHARS budget:$REASONING_BUDGET"
echo " sampling: temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P"
echo " vision=ENABLED (mmproj bf16, offload=$( [ "$MMPROJ_OFFLOAD" = "1" ] && echo on-card || echo OFF-CARD ))"
echo " cache: ram=${CACHE_RAM}MiB reuse=${CACHE_REUSE}  ramfloor=${RAM_FLOOR_MB:-?}MB"
echo " tier : ${LAUNCH_TIER}  (consecutive prior failures: ${_fails})"
echo " devices: ${MAIN_DEVICE_NAMES}"
echo " *** first run: work the FIRST-RUN CHECKLIST at the top of this file ***"
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
    # Only pass --spec-draft-model for a real sidecar; empty means embedded head.
    [ -n "$SPEC_DRAFT_MODEL" ] && SPEC_ARGS+=(--spec-draft-model "$SPEC_DRAFT_MODEL" --spec-draft-ngl all)
    [ -n "${SPEC_DRAFT_DEVICE:-}" ] && SPEC_ARGS+=(--spec-draft-device "$SPEC_DRAFT_DEVICE")
fi
[ "$USE_MMPROJ" = "1" ] && [ "$MMPROJ_OFFLOAD" = "1" ] && [ -n "${MMPROJ_DEVICE:-}" ] \
    && MMPROJ_ARGS+=(--mmproj-device "$MMPROJ_DEVICE")

"$SERVER_PATH" \
  -m "$MODEL" \
  "${DEVICE_ARGS[@]}" \
  "${PLACEMENT_ARGS[@]}" \
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
  --cont-batching &

LLAMA_PID=$!

# Clear the failure counter once this start has proven itself, so a crash after
# hours of healthy service does not push the NEXT start down a tier.
(
  _t0=$(date +%s)
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
      if [ "$(( $(date +%s) - _t0 ))" -ge "$LAUNCH_OK_SECONDS" ]; then
          echo 0 > "$FAIL_FILE" 2>/dev/null || true
          echo "[FALLBACK] up ${LAUNCH_OK_SECONDS}s on tier '${LAUNCH_TIER}' - failure counter reset"
          break
      fi
      sleep 5
  done
) &
TIER_PID=$!

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
  _meta="### LexiPanel-engine BACKEND=vulkan BUILD=b10766 CTX=${CTX} KV=${KV_TYPE}"
  _meta="$_meta B=${BATCH} UB=${UBATCH} SPEC=${SPEC_TYPE:-none} N_MAX=${SPEC_N_MAX}"
  _meta="$_meta MMPROJ=${USE_MMPROJ} MMPROJ_OFFLOAD=${MMPROJ_OFFLOAD} CACHE_RAM=${CACHE_RAM}"
  _meta="$_meta CACHE_REUSE=${CACHE_REUSE} STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$_meta" >> "$LOG_DIR/engine_debug.log"
  echo "$_meta"  > "$LOG_DIR/engine_debug.log.meta"
) &
BANNER_PID=$!

# ============================================================================
# PLACEMENT ASSERTION  (added 2026-09-04)
# Last line of defence for the GTT fault described at the GPU READINESS GATE
# above. The readiness gate should prevent it; this proves it. Wait for the
# server to start listening (llama-server binds only after "model loaded"),
# let allocation settle, then look at where the weights actually are.
#
# On a fault, killing llama-server is the correct move: `wait` below returns
# non-zero under `set -e`, so the unit sees a failure and its
# Restart=on-failure / RestartSec=20 / StartLimitBurst=3 retries a few times -
# which is precisely what fixed it by hand on 2026-09-04 - then stays down
# instead of serving at 1/20th speed with nothing in the log.
#
# Sample once, shortly after load. Do NOT poll: idle eviction legitimately
# moves buffers to GTT later on, and a late sample would false-positive.
# Healthy on this box is ~23,000 MiB VRAM / a few hundred MiB GTT.
# ============================================================================
PLACEMENT_MIN_VRAM_MB=18000
PLACEMENT_GTT_FAULT_MB=10000
PLACEMENT_DEADLINE_S=420
(
  # The thresholds describe the whole model on the XTX. Split across cards, or
  # on another card, the XTX legitimately holds less or nothing; the guard that
  # still applies there is GGML_VK_ALLOW_SYSMEM_FALLBACK=0, which makes a spill
  # to host memory fail the load instead.
  if [ "$MAIN_AMD_ONLY" != "1" ]; then
      echo "[PLACEMENT] skipped: devices are '${MAIN_DEVICE_NAMES}', not the XTX alone"
      exit 0
  fi
  _t0=$(date +%s)
  _verdict="timeout"
  while :; do
      kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
      _v=$(( $(cat "$AMD_DEV/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))
      _g=$(( $(cat "$AMD_DEV/mem_info_gtt_used"  2>/dev/null || echo 0) / 1048576 ))
      # Weights on the card: done, healthy.
      if [ "$_v" -ge "$PLACEMENT_MIN_VRAM_MB" ]; then _verdict="ok"; break; fi
      # Multiple GB in GTT with VRAM still empty: that IS the fault, no need to wait.
      if [ "$_g" -ge "$PLACEMENT_GTT_FAULT_MB" ]; then _verdict="fault"; break; fi
      if [ "$(( $(date +%s) - _t0 ))" -ge "$PLACEMENT_DEADLINE_S" ]; then break; fi
      sleep 2
  done
  kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
  if [ "$_verdict" = "ok" ]; then
      echo "[PLACEMENT] OK: VRAM ${_v} MiB / GTT ${_g} MiB"
  else
      echo "[PLACEMENT] FAULT (${_verdict}): VRAM ${_v} MiB / GTT ${_g} MiB - weights are on the host bus, not the card." >&2
      echo "[PLACEMENT] That is the ~20x slowdown (2.79 t/s prefill / 5.86 t/s decode). Killing so systemd retries." >&2
      kill -9 "$LLAMA_PID" 2>/dev/null || true
  fi
) &
PLACEMENT_PID=$!

# ============================================================================
# HOST-RAM WATCHDOG
# Added after the 2026-09-02 host OOM hard-locked this box (DELTA 1b). A
# Linux OOM on a swap-thrashing 30G host does not reliably kill the offender
# before the machine becomes unusable, so kill llama-server ourselves while
# there is still enough headroom to do it.
#
# 2026-09-04 CORRECTION. This guard was briefly set to 0 on the theory that its
# 05:49:37 and 06:51:20 kills that day were false positives. They were not.
# With it disarmed the box took a real global kernel OOM at 16:06:47:
#   Out of memory: Killed process 108751 (llama-server)
#     total-vm:45021632kB anon-rss:8931112kB
#   systemd: 17.1G memory peak, 7.8G memory SWAP peak (of 8.19G)
# The earlier 'not thrashing' reading was taken at idle, hours after the fact,
# and was simply the wrong moment to sample.
#
# The memory it could not see is GTT. When VRAM runs near full, amdgpu evicts
# model weights into GTT, which is host RAM pinned by the driver - it appears
# in neither ps RSS nor the cgroup. Measured at the OOM via DRM fdinfo:
#   drm-total-vram 23,564,136 KiB allocated (22,465 MiB = 91.5% of the card)
#   drm-memory-vram     16,896 KiB actually resident
#   drm-memory-gtt  15,993,932 KiB = 15.25 GB of the model sitting in RAM
# 15.25 GB GTT + an anon prompt cache allowed to reach 19.5 GB does not fit in
# 30.67 GB. Hence --no-mmproj-offload above: it buys VRAM headroom so the
# eviction that starts the chain does not happen.
#
# Floor set to 512 MB by the operator: low enough not to trip on the deep
# prompt caching this box does by design, high enough to act before the
# kernel has to. Value lives in the TUNABLES block at the top.
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

wait "$LLAMA_PID"
kill "$WATCHDOG_PID" "$PLACEMENT_PID" "$BANNER_PID" "$TIER_PID" 2>/dev/null || true
