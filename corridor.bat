@echo off
REM Walk down a corridor: stays in the middle, stops at a dead end or when something is in the way. Uses the dog's lidar.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click to really walk (Ctrl-C in this window stops it; 40 s max). Try corridor-dry.bat FIRST: it never moves the dog.
REM Settings live in OPTS below: --speed is the top forward speed in m/s. Edit the line, save, relaunch.
set OPTS=--speed 0.2
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python corridor.py --live %OPTS% %*"
echo.
pause
