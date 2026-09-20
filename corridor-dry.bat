@echo off
REM Corridor walk, DRY RUN: prints left / right / front distances and the command it WOULD send. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click it with the dog standing in a corridor; compare the distances with a tape measure. Ctrl-C ends it (40 s max).
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python corridor.py --live --dry"
echo.
pause
