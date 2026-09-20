@echo off
REM Read-only firmware check: connects to the dog, listens ~8 s on its status topics, prints any version-like fields. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python fw_probe.py %*"
echo.
pause
