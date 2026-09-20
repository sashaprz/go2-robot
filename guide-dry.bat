@echo off
REM Guide DRY RUN: plans the route from the dog's lidar and prints the map and what it WOULD do. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Stand the dog where the walk would start, facing the way it would go, and compare the picture with the room. Ctrl-C ends it.
set OPTS=--goto 3,0
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python guide.py --live --dry %OPTS% %*"
echo.
pause
