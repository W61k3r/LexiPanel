#!/bin/bash
# ============================================================================
# bench_backends.sh - Vulkan vs ROCm backend comparison on LexiPanel
# ----------------------------------------------------------------------------
# Compares three builds on the SAME model with IDENTICAL llama-bench params:
#   1. b10766 vulkan   - current production build
#   2. b10798 vulkan   - latest build, same backend  (isolates version drift)
#   3. b10798 rocm     - latest build, HIP/gfx1100   (isolates the backend)
# Running 2 alongside 3 is the whole point: without it a ROCm win is
# indistinguishable from 32 builds of upstream improvement.
#
# llama-bench does NOT exercise speculative decoding, so these numbers are raw
# backend throughput, not the MTP-accelerated t/s the server delivers. That is
# what you want for a backend decision; compare server t/s separately.
#
# DEVICE PINNING IS MANDATORY. This box has an Intel UHD P630 that Vulkan
# happily enumerates as device 1 (confirmed: "Found 2 Vulkan devices"). An
# unpinned run can land on it and the comparison is meaningless.
#
# GPU-EXCLUSIVE. Each build loads the full 15.7 GB model. Two concurrent loads
# is the config that hard-locked this box twice, so this refuses to start while
# anything holds port 8081.
# ============================================================================
set -u

MODEL="/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
OUT_DIR="${1:-/home/admin/llama/bench-$(date +%Y%m%d_%H%M%S)}"

# Match the production launch config so the numbers transfer.
NGL=99; FA=1; KV=q5_1; BATCH=2048; UBATCH=512; THREADS=4; REPS=3

# Test matrix. Depth is the axis that matters here - the workload is 100k+.
PROMPTS="512,4096,16384"      # prefill throughput
GEN="128"                     # decode throughput
DEPTHS="0,8192,32768"         # decode measured after N tokens of context

RADV_ICD="/usr/share/vulkan/icd.d/radeon_icd.json"

mkdir -p "$OUT_DIR"
echo "[INFO] output -> $OUT_DIR"

# --- guards -----------------------------------------------------------------
if ss -tln 2>/dev/null | grep -q ':8081 '; then
    echo "[ABORT] port 8081 is in use - the production server is up."
    echo "        This benchmark needs the GPU to itself. Two 15.7 GB loads OOM the host."
    echo "          sudo systemctl stop LexiPanel-llama       # then re-run"
    echo "          sudo systemctl start LexiPanel-llama      # afterwards"
    exit 1
fi
if ss -tln 2>/dev/null | grep -q ':8083 '; then
    echo "[ABORT] port 8083 is in use - the ROCm server is up. Stop it first."; exit 1
fi
[ ! -f "$MODEL" ] && { echo "[ABORT] model missing: $MODEL"; exit 1; }

avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "$avail" -lt 8000 ] && echo "[WARN] MemAvailable ${avail}MB is low; results may be noisy"

run_bench () {
    local tag="$1" dir="$2" backend="$3"
    local bin="$dir/llama-bench"
    if [ ! -x "$bin" ]; then
        echo "[SKIP] $tag: no llama-bench at $bin"; return
    fi

    # Per-backend env. Pin the device in BOTH cases.
    local -a env=()
    if [ "$backend" = "vulkan" ]; then
        [ ! -f "$RADV_ICD" ] && { echo "[SKIP] $tag: RADV ICD missing"; return; }
        env=(VK_DRIVER_FILES="$RADV_ICD" VK_ICD_FILENAMES="$RADV_ICD"
             GGML_VK_VISIBLE_DEVICES=0 GGML_VK_ALLOW_SYSMEM_FALLBACK=0)
    else
        # Bail early with the real fix rather than a linker error mid-run.
        local missing
        missing=$(ldd "$dir/libggml-hip.so" 2>/dev/null | awk '/not found/{print $1}')
        if [ -n "$missing" ]; then
            echo "[SKIP] $tag: ROCm runtime not installed. Missing: $(echo $missing | tr '\n' ' ')"
            echo "         sudo apt install -y libamdhip64-7 librocblas5 libhipblas3"
            return
        fi
        env=(HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0)
    fi

    echo
    echo "=============================================================="
    echo " $tag  ($backend)"
    echo " $("$bin" --version 2>&1 | head -1)"
    echo "=============================================================="

    # Free page cache pressure between builds so load time is comparable.
    sync

    LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}" env -- "${env[@]}" \
      "$bin" -m "$MODEL" \
        -ngl "$NGL" -fa "$FA" -ctk "$KV" -ctv "$KV" \
        -b "$BATCH" -ub "$UBATCH" -t "$THREADS" -r "$REPS" \
        -p "$PROMPTS" -n "$GEN" -d "$DEPTHS" \
        --progress -o md 2> "$OUT_DIR/${tag}.stderr" \
        | tee "$OUT_DIR/${tag}.md"

    local rc=${PIPESTATUS[0]}
    echo "[INFO] $tag exit=$rc  (stderr -> $OUT_DIR/${tag}.stderr)"

    # Record where memory actually landed - the whole point on this box.
    for d in /sys/class/drm/card*/device; do
        grep -qx "DRIVER=amdgpu" "$d/uevent" 2>/dev/null || continue
        echo "$tag post-run: VRAM $(( $(cat $d/mem_info_vram_used)/1048576 )) MiB / GTT $(( $(cat $d/mem_info_gtt_used)/1048576 )) MiB" \
          | tee -a "$OUT_DIR/placement.txt"
    done
}

{
  echo "host      : $(hostname)  $(uname -sr)"
  echo "date      : $(date -u '+%F %T UTC')"
  echo "model     : $MODEL"
  echo "params    : ngl=$NGL fa=$FA kv=$KV b=$BATCH ub=$UBATCH threads=$THREADS reps=$REPS"
  echo "matrix    : -p $PROMPTS  -n $GEN  -d $DEPTHS"
  echo "note      : llama-bench does not exercise MTP/speculative decoding."
} | tee "$OUT_DIR/run-info.txt"

run_bench "b10766-vulkan" "/home/admin/llama/b10766/llama-b10766"        vulkan
run_bench "b10798-vulkan" "/home/admin/llama/b10798-vulkan/llama-b10798" vulkan
run_bench "b10798-rocm"   "/home/admin/llama/b10798-rocm/llama-b10798"   rocm

echo
echo "=============================================================="
echo " RESULTS -> $OUT_DIR"
ls -1 "$OUT_DIR"
echo "=============================================================="
