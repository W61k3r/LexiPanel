#!/usr/bin/env bash
# The packages LexiPanel needs, and the optional ones for particular features.
# The panel itself is stdlib-only Python 3.12+: no pip, no venv, no compiler.
#
#   bash install-panel-deps.sh            required: caddy ttyd apache2-utils python3 curl pciutils
#   bash install-panel-deps.sh --rocm     + ROCm runtime libraries (ROCm llama.cpp builds)
#   bash install-panel-deps.sh --tts      + eSpeak-ng (Kokoro / Piper / Kitten TTS in audio.cpp)
#   bash install-panel-deps.sh --jinja    + python3-jinja2 (parse check when saving chat templates)
#   bash install-panel-deps.sh --ups      + NUT client (UPS status on the Power options tab)
#   bash install-panel-deps.sh --all      all of the above
# Idempotent: installed packages are left alone. Needs sudo (or root).
set -euo pipefail
SUDO=; [ "$(id -u)" -eq 0 ] || SUDO=sudo
ROCM=0 TTS=0 JINJA=0 UPS=0
for a in "$@"; do
    case "$a" in
        --rocm) ROCM=1 ;; --tts) TTS=1 ;; --jinja) JINJA=1 ;; --ups) UPS=1 ;;
        --all) ROCM=1 TTS=1 JINJA=1 UPS=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option $a (try --help)"; exit 2 ;;
    esac
done

ID=unknown ID_LIKE=
[ -f /etc/os-release ] && . /etc/os-release
FAMILY=$ID
case " $ID $ID_LIKE " in
    *" debian "*|*" ubuntu "*) FAMILY=debian ;;
    *" fedora "*|*" rhel "*) FAMILY=fedora ;;
    *" arch "*) FAMILY=arch ;;
esac
echo "=== LexiPanel dependencies ($ID, $FAMILY family)"

case "$FAMILY" in
    debian)
        PKGS="caddy ttyd apache2-utils python3 curl pciutils"
        [ $ROCM = 1 ] && PKGS="$PKGS libamdhip64-7 librocblas5 libhipblas3"
        [ $TTS = 1 ] && PKGS="$PKGS libespeak-ng1 espeak-ng-data"
        [ $JINJA = 1 ] && PKGS="$PKGS python3-jinja2"
        [ $UPS = 1 ] && PKGS="$PKGS nut-client"
        $SUDO apt-get update
        # shellcheck disable=SC2086
        $SUDO apt-get install -y --no-install-recommends $PKGS ;;
    fedora)
        PKGS="caddy ttyd httpd-tools python3 curl pciutils"
        [ $TTS = 1 ] && PKGS="$PKGS espeak-ng"
        [ $JINJA = 1 ] && PKGS="$PKGS python3-jinja2"
        [ $UPS = 1 ] && PKGS="$PKGS nut-client"
        [ $ROCM = 1 ] && echo "[NOTE] ROCm on Fedora: sudo dnf install rocm-hip-runtime rocblas hipblas"
        # shellcheck disable=SC2086
        $SUDO dnf install -y $PKGS ;;
    arch)
        PKGS="caddy ttyd apache python curl pciutils"
        [ $ROCM = 1 ] && PKGS="$PKGS rocm-hip-runtime rocblas hipblas"
        [ $TTS = 1 ] && PKGS="$PKGS espeak-ng"
        [ $JINJA = 1 ] && PKGS="$PKGS python-jinja"
        [ $UPS = 1 ] && PKGS="$PKGS nut"
        # shellcheck disable=SC2086
        $SUDO pacman -S --needed --noconfirm $PKGS ;;
    *)
        echo "[WARN] $ID is not a distribution this script knows. Install by hand:"
        echo "       caddy, ttyd, htpasswd (apache utils), python3 >= 3.12, curl, lspci (pciutils)" ;;
esac

echo
echo "=== check"
fail=0
for c in caddy ttyd htpasswd python3 curl lspci; do
    if command -v "$c" >/dev/null 2>&1; then printf "  [OK]      %s\n" "$c"
    else printf "  [MISSING] %s\n" "$c"; fail=1; fi
done
if command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
        printf "  [OK]      python %s\n" "$(python3 -c 'import platform; print(platform.python_version())')"
    else
        printf "  [TOO OLD] python %s: LexiPanel needs 3.12 or newer\n" "$(python3 -c 'import platform; print(platform.python_version())')"
        fail=1
    fi
fi
command -v amdgpu_top >/dev/null 2>&1 || echo "  [optional] amdgpu_top (extra GPU tab detail): .deb from github.com/Umio-Yasuno/amdgpu_top/releases"
[ -x "$HOME/onnxrt/venv/bin/python" ] || echo "  [optional] ONNX Runtime (ONNX models, NPUs): bash install-onnx.sh [--openvino|--qnn|--cuda]"
command -v node >/dev/null 2>&1 || echo "  [optional] node (the browser checks in tests/run_all.sh): sudo apt install nodejs npm"
[ $fail = 0 ] && echo "All required packages are present." || { echo "Something required is missing (above)."; exit 1; }
