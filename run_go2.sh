#!/bin/bash
# Launch the all-in-one Go2 window (FPV + drive + tricks). Works on macOS or inside WSL (Ubuntu-24.04).
# usage: run_go2.sh [--demo] [--ip <dog-ip>] [other go2.py flags]
#   --demo   fake robot + synthetic video: preview the window without the dog
DEMO=0; IP="192.168.12.1"; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --demo)     DEMO=1; PASS+=("--demo") ;;
    --selftest) DEMO=1; PASS+=("--selftest") ;;
    --selftest-follow) DEMO=1; PASS+=("--selftest-follow") ;;
    --selftest-voice)  DEMO=1; PASS+=("--selftest-voice") ;;
    --selftest-objects) DEMO=1; PASS+=("--selftest-objects") ;;
    --selftest-voicemove) DEMO=1; PASS+=("--selftest-voicemove") ;;
    --selftest-listen) DEMO=1; PASS+=("--selftest-listen") ;;
    --selftest-heel)   DEMO=1; PASS+=("--selftest-heel") ;;
    --selftest-calib)  DEMO=1; PASS+=("--selftest-calib") ;;
    --fetch-model)     DEMO=1; PASS+=("--fetch-model") ;;   # downloads the person detector; needs internet, not the dog
    --ip)       shift; IP="$1" ;;
    *)          PASS+=("$1") ;;
  esac
  shift
done
export ROBOT_IP="$IP"

cd ~/dimensional-applications || { echo "DimOS install not found"; exit 1; }
source .venv/bin/activate
if [ -f ~/.dimos.env ]; then set -a; . ~/.dimos.env; set +a; fi

if [ "$DEMO" = 0 ]; then
  echo "Checking the dog at $ROBOT_IP ..."
  if ! ping -c 2 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
    echo "Can't reach the dog at $ROBOT_IP. Join Go2_61331_29d4be72 (password 88888888), close the phone app, and try again."
    exit 1
  fi
  # The dog accepts one controller at a time. Ask before stopping a DimOS run that's holding it.
  if dimos status 2>&1 | grep -q "Run ID"; then
    dimos status 2>&1 | head -6
    read -rp "A DimOS run is active and would block the dog. Stop it? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then dimos stop || true; else echo "Leaving it running; exiting."; exit 1; fi
  fi
fi

# Get the directory where this script lives, resolve symlinks
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || readlink "$0" 2>/dev/null || echo "$0")")" && pwd)"
exec python "$SCRIPT_DIR/go2.py" "${PASS[@]}"
