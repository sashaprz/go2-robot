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

# Find Python executable - try venv first, then system python3
if [ -f "$SCRIPT_DIR/dimensional-applications/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/dimensional-applications/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../dimensional-applications/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/../dimensional-applications/.venv/bin/python"
else
  PYTHON="python3"
fi

# Run gesture control using the DimOS Python
exec "$PYTHON" "$SCRIPT_DIR/gesture_robot.py" "${PASS[@]}"
