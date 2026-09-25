#!/usr/bin/env bash
# Package every hard-to-reproduce config on LexiPanel into one tarball.
# Deliberately EXCLUDES *.gguf (17 GB, re-downloadable from HuggingFace) and the
# llama.cpp binaries (re-downloadable prebuilt). Everything else here is either
# hand-tuned or was learned the hard way, so it all goes in.
# Usage: make-backup.sh [outdir]   -> prints the tarball path on stdout
set -euo pipefail

OUTDIR="${1:-/home/admin/backups}"
TS=$(date +%Y%m%d_%H%M%S)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
R="$STAGE/LexiPanel-config-$TS"
mkdir -p "$R"/{panel,llama,models,llama_logs,netplan-staging,etc/systemd,etc/default}

# --- user-space config -----------------------------------------------------
cp -a /home/admin/panel/. "$R/panel/" 2>/dev/null || true
rm -rf "$R/panel/__pycache__"
# launch script + its prior revisions; skip the 33MB release tarball and b10766/
for f in /home/admin/llama/*.sh /home/admin/llama/*.sh.* /home/admin/llama/*.md; do
    [ -f "$f" ] && cp -a "$f" "$R/llama/"
done
# model-adjacent config only, never the weights
find /home/admin/models -maxdepth 1 -type f ! -name '*.gguf' \
     -exec cp -a {} "$R/models/" \; 2>/dev/null || true
cp -a /home/admin/llama_logs/. "$R/llama_logs/" 2>/dev/null || true
cp -a /home/admin/netplan-staging/. "$R/netplan-staging/" 2>/dev/null || true

# --- system config ---------------------------------------------------------
for u in LexiPanel-llama LexiPanel-panel LexiPanel-ttyd LexiPanel-shell; do
    [ -r "/etc/systemd/system/$u.service" ] && \
        cp -a "/etc/systemd/system/$u.service" "$R/etc/systemd/"
done
[ -r /etc/caddy/Caddyfile ]  && cp -a /etc/caddy/Caddyfile "$R/etc/Caddyfile"
[ -r /etc/default/ttyd ]     && cp -a /etc/default/ttyd    "$R/etc/default/ttyd"

# --- live state, for diffing after a future change -------------------------
{
  echo "# LexiPanel live state, captured $(date -Is)"
  echo
  echo "## service state"
  for u in LexiPanel-llama LexiPanel-panel LexiPanel-ttyd LexiPanel-shell caddy ttyd; do
      printf '%-14s %-10s %s\n' "$u" "$(systemctl is-active "$u" 2>&1)" \
                                     "$(systemctl is-enabled "$u" 2>&1)"
  done
  echo; echo "## listening sockets"; ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | sort -u
  PID=$(pgrep -f 'llama-server -m' | head -1 || true)
  if [ -n "$PID" ]; then
      echo; echo "## llama-server argv"; tr '\0' ' ' < "/proc/$PID/cmdline"; echo
      echo; echo "## llama-server vulkan env"
      tr '\0' '\n' < "/proc/$PID/environ" | grep -E '^(GGML|VK_)' | sort
  fi
  echo; echo "## kernel"; uname -r
  echo; echo "## gpu"
  # Resolve the amdgpu node by DRIVER, never by card number. This block used to
  # read card0, which does not exist on this box - card numbering moved when
  # amdgpu took over the simple-framebuffer. Under `set -e` the failed cat made
  # the arithmetic abort the whole LIVE-STATE block, so every backup taken
  # before 2026-09-09 is silently TRUNCATED here and has no GPU section, no
  # build manifest and no "not captured" note. Fixed, and made non-fatal: a
  # missing counter must degrade this file, never lose the rest of it.
  AMD_DEV=""
  for _c in /sys/class/drm/card*/device; do
      if grep -q '^DRIVER=amdgpu$' "$_c/uevent" 2>/dev/null; then AMD_DEV="$_c"; break; fi
  done
  if [ -n "$AMD_DEV" ]; then
      echo "card_path $AMD_DEV"
      for _k in mem_info_vram_used mem_info_vram_total mem_info_gtt_used mem_info_gtt_total; do
          _v=$(cat "$AMD_DEV/$_k" 2>/dev/null || echo "")
          if [ -n "$_v" ]; then echo "${_k#mem_info_} $(( _v / 1048576 )) MiB"
          else echo "${_k#mem_info_} unavailable"; fi
      done
      echo "link $(cat "$AMD_DEV/current_link_speed" 2>/dev/null || echo ?) x$(cat "$AMD_DEV/current_link_width" 2>/dev/null || echo ?)"
  else
      echo "no amdgpu device found"
  fi

  # The tarball deliberately excludes the llama.cpp binaries, so record WHICH
  # builds were installed and which one each backend was pointed at. Without
  # this a restore knows the config but not the engine it was tuned against.
  echo; echo "## llama.cpp builds installed (binaries excluded from this tarball)"
  for _d in /home/admin/llama/b*/; do
      [ -d "$_d" ] || continue
      _bin=$(find "$_d" -maxdepth 2 -name llama-server -type f 2>/dev/null | head -1)
      [ -n "$_bin" ] || continue
      _flav=cpu
      _libdir=$(dirname "$_bin")
      [ -e "$_libdir/libggml-vulkan.so" ] && _flav=vulkan
      [ -e "$_libdir/libggml-hip.so" ] && _flav=rocm
      printf '%-24s %-7s %s\n' "$(basename "$_d")" "$_flav" "$_libdir"
  done
  echo; echo "## active build per backend (panel/builds.env)"
  if [ -r /home/admin/panel/builds.env ]; then
      grep -E '^LIB_DIR_' /home/admin/panel/builds.env || echo "(no selections)"
  else
      echo "(no builds.env - every backend on its launch-script fallback)"
  fi

  echo; echo "## NOT captured (needs root): /etc/ufw/*.rules, /var/lib/caddy PKI root"
} > "$R/LIVE-STATE.txt"

for d in AI-INSTRUCTIONS.md CLAUDE.md; do
    cp -a "/home/admin/$d" "$R/" 2>/dev/null || true
done

TARBALL="$OUTDIR/LexiPanel-config-$TS.tar.gz"
mkdir -p "$OUTDIR"
tar -czf "$TARBALL" -C "$STAGE" "LexiPanel-config-$TS"
chmod 600 "$TARBALL"   # contains the Caddy basicauth bcrypt hash
ln -sfn "$TARBALL" "$OUTDIR/LexiPanel-config-latest.tar.gz"
echo "$TARBALL"
