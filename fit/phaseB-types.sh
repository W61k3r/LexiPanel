#!/usr/bin/env bash
# =============================================================================
# LexiPanel Fit - phase B: how fast is each integer format on the 7900 XTX?
#
#   bash ~/fitquant/phaseB-types.sh quantize   # CPU only; main keeps serving
#   bash ~/fitquant/phaseB-types.sh bench      # STOPS main, benchmarks, RESTARTS main
#   bash ~/fitquant/phaseB-types.sh deep       # STOPS main: the 3 fastest at 128k depth
#   bash ~/fitquant/phaseB-types.sh report     # table from the results so far
#   bash ~/fitquant/phaseB-types.sh clean      # delete the test quants (asks)
#
# Each test model is the real 27B made from the BF16 source in ONE format
# ("--pure"), so its speed is the speed of that format on this card. Quality is
# not measured here (that is phase D); these files are speed probes only.
# Settings mirror main: flash attention, KV iq4_nl, batch 2048 / ubatch 512,
# all layers on the XTX, the same Vulkan pinning, sysmem fallback OFF (so a
# format the card cannot hold fails loudly instead of crawling from host RAM).
#
# main is stopped with the passwordless sudoers rule and ALWAYS restarted on
# exit - error, Ctrl-C or success (trap). The operator cleared GPU use for the
# Fit work on 2026-09-24.
# =============================================================================
set -Eeuo pipefail

BF16="${BF16:-$HOME/models/src/Qwen3.8-27B-TurboFCFusion-735-882/Qwen3.8-27B-TurboFCFusion-735-882-BF16.gguf}"
OUT="${OUT:-$HOME/models/fit-test}"
RES="$HOME/fitquant/results/phaseB"
CPU_BIN="${CPU_BIN:-$HOME/llama/b11149-cpu/llama-b11149}"
VK_BIN="${VK_BIN:-$HOME/llama/b11149-vulkan/llama-b11149}"
TYPES="${TYPES:-Q4_0 IQ4_NL IQ4_XS Q4_K Q5_K Q6_K}"
QTHREADS="${QTHREADS:-6}"
mkdir -p "$OUT" "$RES"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mSTOP:\033[0m %s\n' "$*"; exit 1; }
name_of() { echo "$OUT/Qwen38-27B-pure-$1.gguf"; }

main_up()   { curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"ok"'; }
stop_main() {
  say "Stopping main (LexiPanel-llama)"
  sudo -n /usr/bin/systemctl stop LexiPanel-llama || die "could not stop main (sudoers rule missing?)"
  for _ in $(seq 60); do pgrep -f 'llama-server -m' >/dev/null || break; sleep 1; done
  pgrep -f 'llama-server -m' >/dev/null && die "main is still running after 60 s - not benchmarking over it"
  sleep 3                                       # let the driver release VRAM
  echo "   main stopped at $(date +%H:%M:%S)"
}
start_main() {
  say "Starting main (LexiPanel-llama)"
  sudo -n /usr/bin/systemctl start LexiPanel-llama || { echo "!! start failed - run: sudo systemctl start LexiPanel-llama"; return; }
  for _ in $(seq 300); do main_up && { echo "   main healthy at $(date +%H:%M:%S)"; return; }; sleep 2; done
  echo "!! main did not report healthy within 10 min - check: journalctl -u LexiPanel-llama -n 50"
}

vk_env() {  # exactly what main runs with
  export GGML_VK_VISIBLE_DEVICES=0 GGML_VK_ALLOW_SYSMEM_FALLBACK=0
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
}

bench_one() {  # $1 type, $2 depths, $3 reps, $4 tag
  local f; f=$(name_of "$1")
  [[ -f "$f" ]] || { echo "   (no $f - run quantize first)"; return; }
  echo "   $1 at depth $2 ..."
  if ! "$VK_BIN/llama-bench" -m "$f" -ngl 99 -fa on -ctk iq4_nl -ctv iq4_nl -b 2048 -ub 512 -t 6 \
        -p 512 -n 128 -d "$2" -r "$3" -o jsonl >"$RES/$1.$4.jsonl" 2>"$RES/$1.$4.log"; then
    echo "   !! $1 failed - see $RES/$1.$4.log (tail below)"; tail -5 "$RES/$1.$4.log" | sed 's/^/      /'
  fi
}

report() {
  python3 - "$RES" <<'EOF'
import json, glob, os, sys
res = sys.argv[1]; rows = {}
for f in sorted(glob.glob(os.path.join(res, "*.jsonl"))):
    t = os.path.basename(f).split(".")[0]
    for line in open(f):
        try: d = json.loads(line)
        except ValueError: continue
        kind = "pp" if d.get("n_prompt") else "tg"
        rows.setdefault(t, {})[(kind, d.get("n_depth", 0))] = (d.get("avg_ts"), d.get("stddev_ts"), d.get("model_size"))
if not rows: print("no results yet"); sys.exit()
depths = sorted({k[1] for r in rows.values() for k in r})
hdr = f"{'format':8} {'size GiB':>8}" + "".join(f" {'pp512@'+str(d//1024)+'k':>11} {'tg128@'+str(d//1024)+'k':>11}" for d in depths)
print(hdr); print("-" * len(hdr))
best_tg = max((v[0] for r in rows.values() for k, v in r.items() if k[0] == "tg" and k[1] == 0 and v[0]), default=None)
for t, r in sorted(rows.items(), key=lambda kv: -(kv[1].get(("tg", 0), (0,))[0] or 0)):
    size = next((v[2] for v in r.values() if v[2]), 0) / 2**30
    line = f"{t:8} {size:8.2f}"
    for d in depths:
        pp, tg = r.get(("pp", d)), r.get(("tg", d))
        line += f" {pp[0]:11.1f}" if pp and pp[0] else f" {'-':>11}"
        line += f" {tg[0]:11.2f}" if tg and tg[0] else f" {'-':>11}"
    tg0 = r.get(("tg", 0), (None,))[0]
    if best_tg and tg0 and tg0 < 0.5 * best_tg: line += "   <- under half the best: CPU fallback?"
    print(line)
EOF
}

case "${1:-}" in
quantize)
  [[ -f "$BF16" ]] || die "no BF16 source at $BF16 (run phaseA-source.sh)"
  for t in $TYPES; do
    f=$(name_of "$t")
    if [[ -f "$f" ]]; then echo "   have $t"; continue; fi
    say "Quantize $t (CPU, low priority)"
    nice -n 19 ionice -c3 "$CPU_BIN/llama-quantize" --pure "$BF16" "$f.part" "$t" "$QTHREADS" 2>&1 \
      | grep -E 'model size|quant size|error|failed' || true
    [[ -s "$f.part" ]] || die "quantize $t produced nothing"
    mv "$f.part" "$f"
  done
  ls -la "$OUT"; df -h "$OUT" | tail -1 ;;
bench)
  vk_env; trap start_main EXIT
  stop_main
  say "Benchmark each format at depth 0 and 32k (2 runs each)"
  for t in $TYPES; do bench_one "$t" 0,32768 2 short; done
  report ;;
deep)
  vk_env
  top=$(report | awk 'NR>2{print $1}' | head -3)
  [[ -n "$top" ]] || die "no short results yet - run bench first"
  trap start_main EXIT
  stop_main
  say "The three fastest at 128k depth (1 run each; each fills 131k tokens first)"
  for t in $top; do bench_one "$t" 131072 1 deep; done
  report ;;
report) report ;;
clean)
  ls -la "$OUT"; read -r -p "Delete these test quants? [y/N] " a
  [[ "$a" =~ ^[Yy]$ ]] && rm -f "$OUT"/Qwen38-27B-pure-*.gguf && echo deleted ;;
*) sed -n '2,24p' "$0" ;;
esac
