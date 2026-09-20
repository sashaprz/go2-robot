@echo off
REM Guide mode: leads the dog from where it stands to a point given in metres (AHEAD,LEFT), round obstacles its lidar sees.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click to really walk (Ctrl-C in this window stops it). Try guide-dry.bat FIRST: it never moves the dog.
REM Settings live in OPTS below. --goto 4,1 = 4 m ahead and 1 m to the left; several points (--goto 4,0 4,3) are visited in order.
REM --speed is the top forward speed in m/s, --seconds the time limit. Edit the line, save, relaunch.
set OPTS=--goto 3,0 --speed 0.3
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python guide.py --live %OPTS% %*"
echo.
pause
