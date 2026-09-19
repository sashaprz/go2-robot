@echo off
REM Go2 all-in-one: live camera + driving + tricks in one window (runs inside WSL). Join the dog's Wi-Fi first.
REM   go2.bat          -> connect to the dog
REM   go2.bat --demo   -> preview the window with a fake robot (no dog needed)
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh %*
echo.
pause
