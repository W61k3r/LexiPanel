#!/usr/bin/env bash
# ============================================================================
# audit-flags.sh - diff a llama.cpp build's knob surface against what this box
# actually exposes. Referenced from the FLAG SURFACE AUDIT block in panel.py;
# run it after every upgrade, before trusting the parameter table.
#
#   ./audit-flags.sh                     audit the active vulkan build
#   ./audit-flags.sh <build-dir>         audit a specific build
#   ./audit-flags.sh <old-dir> <new-dir> diff two builds against each other
#
# WHY THIS EXISTS RATHER THAN A ONE-LINE GREP: flag definitions live in the
# LEFT column of --help, descriptions in the right. A loose grep over whole
# lines both scoops flag names out of prose AND silently drops aliases,
# depending on how the regex consumes the "a, --b" separator. Doing it by hand
# on 2026-09-16 produced a false "--top-n-sigma was added in b11007" when the
# flag was present in both builds. So: split the columns FIRST, then extract.
# ============================================================================
set -uo pipefail
export LC_ALL=C
PANEL=/home/admin/panel
LLAMA=/home/admin/llama

flags_of() {   # $1 = build dir -> sorted unique --flags
    LD_LIBRARY_PATH="$1" "$1/llama-server" --help 2>&1 \
      | sed 's/  \+.*$//' \
      | grep -oE '\-\-[a-z0-9][a-z0-9-]*' | sort -u
}
envs_of() {    # $1 = build dir, $2 = prefix -> sorted unique env var names
    local so
    for so in libggml-vulkan.so libggml-hip.so libggml-base.so; do
        [ -f "$1/$so" ] && strings "$1/$so"
    done 2>/dev/null | grep -oE "^$2[A-Z0-9_]+$" | sort -u
}
active_vulkan() { sed -n 's/^LIB_DIR_VULKAN="\(.*\)"$/\1/p' "$PANEL/builds.env"; }

NEW="${1:-$(active_vulkan)}"
[ -x "$NEW/llama-server" ] || { echo "no llama-server in: $NEW" >&2; exit 1; }

if [ $# -ge 2 ]; then
    OLD="$1"; NEW="$2"
    echo "### build-to-build: $(basename "$OLD") -> $(basename "$NEW")"
    echo "--- flags ADDED:";   comm -13 <(flags_of "$OLD") <(flags_of "$NEW") | sed 's/^/    /'
    echo "--- flags REMOVED:"; comm -23 <(flags_of "$OLD") <(flags_of "$NEW") | sed 's/^/    /'
    for p in GGML_VK_ GGML_CUDA_; do
        echo "--- $p ADDED:";   comm -13 <(envs_of "$OLD" $p) <(envs_of "$NEW" $p) | sed 's/^/    /'
        echo "--- $p REMOVED:"; comm -23 <(envs_of "$OLD" $p) <(envs_of "$NEW" $p) | sed 's/^/    /'
    done
    exit 0
fi

echo "### $("$NEW/llama-server" --version 2>&1 | head -1)   ($NEW)"
flags_of "$NEW" > /tmp/.af_build.$$

# What this box references anywhere: the panel's tips and tables, plus every
# launch script. A flag named in either is "known", even if it is only
# documented - an unexposed flag we have deliberately written about is not the
# same as one nobody has looked at.
{ cat "$PANEL/panel.py"; cat "$LLAMA"/run_llama_*.sh; } 2>/dev/null \
  | grep -oE '\-\-[a-z0-9][a-z0-9-]*' | sort -u > /tmp/.af_ours.$$

echo "--- in the build, referenced NOWHERE here ($(comm -23 /tmp/.af_build.$$ /tmp/.af_ours.$$ | wc -l)):"
comm -23 /tmp/.af_build.$$ /tmp/.af_ours.$$ | sed 's/^/    /'

echo "--- we reference, but the build does NOT have (investigate: renamed or removed):"
comm -13 /tmp/.af_build.$$ /tmp/.af_ours.$$ | sed 's/^/    /'

echo "--- flagged DEPRECATED or REMOVED by this build (must not be exposed):"
LD_LIBRARY_PATH="$NEW" "$NEW/llama-server" --help 2>&1 \
  | grep -B0 -iE 'DEPRECATED|has been removed' | sed 's/^/    /' | head -20

for p in GGML_VK_ GGML_CUDA_; do
    n=$(envs_of "$NEW" $p | wc -l)
    have=$(envs_of "$NEW" $p | grep -cFf <(grep -oE "${p}[A-Z0-9_]+" "$PANEL/panel.py" | sort -u) 2>/dev/null || echo 0)
    echo "--- $p in build: $n   exposed/mentioned in panel.py: $have"
    envs_of "$NEW" $p | grep -vFf <(grep -oE "${p}[A-Z0-9_]+" "$PANEL/panel.py" | sort -u) 2>/dev/null | sed 's/^/    missing: /'
done
rm -f /tmp/.af_build.$$ /tmp/.af_ours.$$
