@echo off
REM Records what the dog's lidar sees for 60 s, so a person tracker can be built on real data. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click it. The window tells you what to do and when. Saves lidar_recording.npz next to these files.
set ARGS=%*
if "%ARGS%"=="" set ARGS=60 save
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python lidar_probe.py %ARGS%"
echo.
pause
