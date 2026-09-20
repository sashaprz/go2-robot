#!/bin/bash
# Launch gesture robot control
# usage: ./run_gesture.sh [--confirm-real] [--forward-speed 0.2]

PASS=()
while [ $# -gt 0 ]; do
  PASS+=("$1")
  shift
done

# Add the Nordic toolchain Python packages to PYTHONPATH so MediaPipe can be found
export PYTHONPATH="/opt/nordic/ncs/toolchains/ef4fc6722e/lib/python3.12/site-packages:$PYTHONPATH"

# Load AES key
if [ -f ~/.dimos.env ]; then set -a; . ~/.dimos.env; set +a; fi

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run gesture control using the DimOS Python
exec /Users/anastasiyavolgina/dimensional-applications/.venv/bin/python "$SCRIPT_DIR/gesture_robot.py" "${PASS[@]}"
