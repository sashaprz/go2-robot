@echo off
REM Run while on a network WITH internet (HackMIT), NOT the dog's Wi-Fi.
REM Double-click this from Explorer or run it in a normal terminal (it needs keyboard input).
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/fetch_aes_key.sh
echo.
pause
