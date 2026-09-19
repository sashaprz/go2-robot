#!/bin/bash
# Launch a DimOS blueprint against the Go2. Runs inside WSL (Ubuntu-24.04).
# usage: run_dimos.sh [blueprint] [robot_ip]
#   blueprint default: unitree-go2-webrtc-keyboard-teleop   (drive with the keyboard, no LLM, works offline)
#   robot_ip  default: 192.168.12.1        (the dog's own hotspot)
BLUEPRINT="${1:-unitree-go2-webrtc-keyboard-teleop}"
export ROBOT_IP="${2:-192.168.12.1}"

if [ -z "$DIMOS_SKIP_PREFLIGHT" ]; then
  echo "Checking the dog at $ROBOT_IP ..."
  if ! ping -c 2 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
    echo
    echo "Can't reach the dog at $ROBOT_IP."
    echo "  - Is the PC on the dog's Wi-Fi (Go2_61331_29d4be72, password 88888888)?"
    echo "  - Or, if the dog is on another network, pass its IP:  dimos-go2.bat $BLUEPRINT <dog-ip>"
    exit 1
  fi
fi

cd ~/dimensional-applications || { echo "DimOS install not found"; exit 1; }
source .venv/bin/activate

# Optional API keys (e.g. OPENAI_API_KEY=...) for the agentic blueprints live here, not in this script.
if [ -f ~/.dimos.env ]; then set -a; . ~/.dimos.env; set +a; fi

echo "Starting DimOS blueprint '$BLUEPRINT' for robot $ROBOT_IP  (Ctrl+C to stop)"
exec dimos run "$BLUEPRINT"
