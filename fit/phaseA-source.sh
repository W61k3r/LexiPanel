#!/usr/bin/env bash
# =============================================================================
# LexiPanel Fit - phase A: get a model's FULL-PRECISION source and turn it into
# a BF16 GGUF that later phases can requantize for this box's cards.
#
#   bash ~/fitquant/phaseA-source.sh --check    # preflight only, changes nothing
#   bash ~/fitquant/phaseA-source.sh            # do it (asks once before starting)
#   bash ~/fitquant/phaseA-source.sh --yes      # do it, no question
#
# Safe to stop (Ctrl-C) and re-run: every step resumes or skips if done.
#
# What it touches:
#   ~/fitquant/          tools: uv, a private Python 3.12, llama.cpp's converter
#   ~/models/src/<NAME>/ the downloaded safetensors, the BF16 GGUF, a manifest
# What it never touches: the panel, any service, any GPU, any setting, sudo.
# To undo completely:   rm -rf ~/fitquant ~/models/src/<NAME>
#
# Why a full-precision source: requantizing an already-quantized GGUF stacks
# rounding error on rounding error (llama-quantize itself warns it "can
# severely reduce quality"). Every later quant is made from this BF16 file.
# =============================================================================
set -Eeuo pipefail

# ---- what to fetch (override any of these from the environment) -------------
REPO="${REPO:-DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU}"
REV="${REV:-fd6a26869dc5775b3c81ce65468af634dd300b02}"   # pinned: the exact upload checked 2026-09-23
NAME="${NAME:-Qwen3.8-27B-TurboFCFusion-735-882}"
LLAMA_TAG="${LLAMA_TAG:-b11011}"                          # same llama.cpp as the running builds
CPU_BIN="${CPU_BIN:-$HOME/llama/b11149-cpu/llama-b11149}" # CPU build: the sanity check never opens a GPU

BASE="$HOME/fitquant"
SRC="$HOME/models/src/$NAME"
HFDIR="$SRC/hf"
GGUF="$SRC/$NAME-BF16.gguf"
LLAMA="$BASE/llama.cpp-$LLAMA_TAG"
VENV="$BASE/venv"
PY="$VENV/bin/python"
UV="$BASE/bin/uv"
LOG="$BASE/logs/phaseA-$(date +%Y%m%d_%H%M%S).log"

export UV_CACHE_DIR="$BASE/cache/uv" UV_PYTHON_INSTALL_DIR="$BASE/python"
export HF_HOME="$BASE/cache/hf"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""    # conversion is CPU work; keep GPUs out of it
# reuse the token the panel's Download tab saved, if any (this repo is public, so optional)
[[ -z "${HF_TOKEN:-}" && -s "$HOME/.cache/huggingface/token" ]] && export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"

MODE=run; YES=0
for a in "$@"; do
  case "$a" in
    --check) MODE=check ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $a (try --help)"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31mSTOP:\033[0m %s\n' "$*"; exit 1; }
done_mark() { touch "$SRC/.done-$1"; }
is_done()   { [[ -f "$SRC/.done-$1" ]]; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1e9}'; }
trap '[[ $BASHPID == $$ ]] && { echo; echo "Stopped at line $LINENO. Re-run the same command to resume."; }' ERR

# =============================================================================
say "Preflight"
[[ $EUID -ne 0 ]] || die "run as your normal user, not root - nothing here needs sudo"
for t in curl git tar sha256sum python3 nice ionice df awk; do
  command -v "$t" >/dev/null || die "missing tool: $t"
done
ok "tools present"

# Ask Hugging Face what the source is, at the pinned revision.
META="$(curl -fsS --max-time 60 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        "https://huggingface.co/api/models/$REPO/revision/$REV?blobs=true")" \
  || die "could not reach Hugging Face for $REPO @ ${REV:0:10}"
read -r SRC_BYTES ARCH < <(python3 -c '
import json,sys; d=json.loads(sys.stdin.read())
tot=sum((s.get("size") or 0) for s in d.get("siblings",[]))
print(tot, ",".join((d.get("config") or {}).get("architectures") or ["?"]))' <<<"$META")
ok "source: $REPO @ ${REV:0:10}  ($(gb "$SRC_BYTES"), architecture $ARCH)"

# Space: the download, plus a BF16 GGUF about the same size, plus 10 GB slack.
have_src=$( (du -sb "$HFDIR" 2>/dev/null || true) | awk '{print $1}'); have_src=${have_src:-0}
have_gguf=0; is_done convert && have_gguf=$SRC_BYTES
NEED=$(( SRC_BYTES - have_src + SRC_BYTES - have_gguf + 10000000000 ))
if (( NEED < 0 )); then NEED=0; fi
at="$SRC"; while [[ ! -d "$at" ]]; do at=$(dirname "$at"); done   # nearest folder that exists
FREE=$(df -B1 --output=avail "$at" | tail -1 | tr -d ' ')
(( FREE > NEED )) || die "needs $(gb $NEED) free on $(df --output=target "$at" | tail -1), has $(gb $FREE)"
ok "disk: needs $(gb $NEED) more, $(gb $FREE) free"

AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
(( AVAIL_KB > 6*1024*1024 )) || die "only $((AVAIL_KB/1024)) MB RAM available; the converter wants ~6 GB headroom"
ok "RAM: $((AVAIL_KB/1024/1024)) GB available (conversion streams tensors; ~2-6 GB used)"

[[ -x "$CPU_BIN/llama-quantize" ]] || warn "no CPU llama-quantize at $CPU_BIN - the final size check will be skipped"

cat <<EOF

   Plan
     1. uv + private Python 3.12          -> $BASE      (~1.5 GB incl. CPU-only PyTorch)
     2. llama.cpp $LLAMA_TAG converter    -> $LLAMA
     3. download safetensors (resumable)  -> $HFDIR     ($(gb "$SRC_BYTES"))
     4. verify every file's SHA-256 against Hugging Face
     5. convert to BF16 GGUF, low priority -> $GGUF
     6. inspect: tensor count, MTP head present, types
     7. size check: llama-quantize --dry-run to Q4_K_M (CPU, reads the header only)
   Download time depends on your line: at 10 MB/s about 1.5 h, at 2 MB/s about 8 h.
   Conversion: roughly 15-40 min of CPU and disk. Log: $LOG
EOF
[[ $MODE == check ]] && { echo; echo "Preflight passed. Nothing was changed."; exit 0; }
if (( ! YES )); then
  read -r -p $'\n   Start? [y/N] ' ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "Not started."; exit 0; }
fi
mkdir -p "$SRC" "$BASE/logs"
printf '%s' "$META" > "$SRC/.hf-meta.json"                 # what step 4 verifies against
exec > >(tee -a "$LOG") 2>&1

# =============================================================================
say "1. uv and Python"
if [[ ! -x "$UV" ]]; then
  mkdir -p "$BASE/bin" "$BASE/dl"
  U="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
  curl -fL --retry 3 -o "$BASE/dl/uv.tar.gz" "$U"
  curl -fL --retry 3 -o "$BASE/dl/uv.tar.gz.sha256" "$U.sha256"
  (cd "$BASE/dl" && echo "$(awk '{print $1}' uv.tar.gz.sha256)  uv.tar.gz" | sha256sum -c -) \
    || die "uv download failed its checksum"
  tar -xzf "$BASE/dl/uv.tar.gz" -C "$BASE/dl"
  install -m 755 "$BASE"/dl/uv-x86_64-unknown-linux-gnu/uv "$UV"
fi
ok "$("$UV" --version)"

# =============================================================================
say "2. llama.cpp $LLAMA_TAG converter"
if [[ ! -f "$LLAMA/convert_hf_to_gguf.py" ]]; then
  rm -rf "$LLAMA"
  git clone --quiet --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
ok "converter at $(git -C "$LLAMA" rev-parse --short HEAD)"

# llama.cpp's requirements pin transformers 4.57.6, but models saved with transformers 5
# (Qwen3.8 uploads among them) name their tokenizer class "TokenizersBackend", which 4.x
# cannot load - the conversion dies at "Set model tokenizer". transformers 5 converts them;
# checked 2026-09-23: the vocabulary it writes matches DavidAU's published GGUF exactly
# (tokens, merges, special ids, chat template). So the packages are listed here instead of
# using the requirements file.
if [[ ! -x "$PY" ]] || ! "$PY" -c 'import sys, torch, numpy, sentencepiece, huggingface_hub, transformers
sys.exit(int(transformers.__version__.split(".")[0]) < 5)' 2>/dev/null; then
  [[ -x "$PY" ]] || "$UV" venv --quiet --python 3.12 "$VENV"
  "$UV" pip install --quiet --python "$PY" --extra-index-url https://download.pytorch.org/whl/cpu \
      "numpy~=2.2.6" "sentencepiece>=0.1.98,<0.3.0" "protobuf>=4.21.0,<5.0.0" "torch==2.11.0" \
      "transformers>=5,<6" "huggingface_hub[hf_xet]>=0.34"
fi
ok "Python env: $("$PY" -c 'import sys,torch,transformers;print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")')"

# =============================================================================
say "3. Download $REPO"
if ! is_done download; then
  "$PY" - "$REPO" "$REV" "$HFDIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=rev, local_dir=out, max_workers=4)
EOF
  done_mark download
fi
ok "files in $HFDIR ($(du -sh "$HFDIR" | cut -f1))"

# =============================================================================
say "4. Verify checksums"
if ! is_done verify; then
  python3 - "$HFDIR" "$SRC/manifest.json" "$REPO" "$REV" "$SRC/.hf-meta.json" <<'EOF' || die "a file does not match Hugging Face - delete it and re-run to fetch it again"
import hashlib, json, os, sys, time
root, man, repo, rev, meta = sys.argv[1:6]
d = json.load(open(meta))
out = dict(repo=repo, revision=rev, checked=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files={})
bad = 0
for s in d["siblings"]:
    name = s["rfilename"]; p = os.path.join(root, name)
    if not os.path.isfile(p):
        print("   MISSING", name); bad += 1; continue
    want = (s.get("lfs") or {}).get("sha256")
    size = os.path.getsize(p)
    if want:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        got = h.hexdigest()
        status = "ok" if got == want else "MISMATCH"
        bad += got != want
    else:
        got, status = None, "small file (git, no LFS hash)"
    print(f"   {status:8} {size/1e9:7.2f} GB  {name}")
    out["files"][name] = dict(size=size, sha256=got)
json.dump(out, open(man, "w"), indent=1)
sys.exit(1 if bad else 0)
EOF
  done_mark verify
fi
ok "all files match - manifest: $SRC/manifest.json"

# =============================================================================
say "5. Convert to BF16 GGUF"
if ! is_done convert; then
  rm -f "$GGUF"
  (cd "$LLAMA" && nice -n 19 ionice -c3 "$PY" convert_hf_to_gguf.py "$HFDIR" \
      --outtype bf16 --outfile "$GGUF")
  done_mark convert
fi
ok "$GGUF ($(gb "$(stat -c %s "$GGUF")"))"

# =============================================================================
say "6. Inspect"
PYTHONPATH="$LLAMA/gguf-py" "$PY" - "$GGUF" <<'EOF'
import sys, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
names = [t.name for t in r.tensors]
types = collections.Counter(t.tensor_type.name for t in r.tensors)
mtp = [n for n in names if "nextn" in n]
f = r.fields.get("general.architecture")
arch = bytes(f.parts[f.data[0]]).decode() if f else "?"
print(f"   architecture {arch}, {len(names)} tensors, types {dict(types)}")
print(f"   MTP head: {'present, ' + str(len(mtp)) + ' tensors' if mtp else 'NOT FOUND - speculative decoding would be lost'}")
EOF

# =============================================================================
say "7. Size check (dry run, no file written)"
if [[ -x "$CPU_BIN/llama-quantize" ]]; then
  nice -n 19 "$CPU_BIN/llama-quantize" --dry-run "$GGUF" Q4_K_M 2>&1 \
    | grep -Ei 'model size|quant size|size =' | tail -3 || true
  echo "   (DavidAU's published MTP Q4_K_M is 18.50 GB - a close number means the BF16 is complete)"
fi

# =============================================================================
say "Done"
( cd "$SRC" && sha256sum "$(basename "$GGUF")" | tee "$(basename "$GGUF").sha256" )
cat <<EOF

   BF16 source ready:  $GGUF
   Manifest:           $SRC/manifest.json
   Log:                $LOG

   The safetensors in $HFDIR ($(du -sh "$HFDIR" | cut -f1)) are no longer needed for
   llama.cpp quants. Keep them if you want the component work (phase E); otherwise:
       rm -rf "$HFDIR"
EOF
