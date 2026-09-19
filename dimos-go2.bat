@echo off
REM Launch DimOS for the Go2 (runs inside WSL Ubuntu-24.04).
REM   dimos-go2.bat                                  -> keyboard teleop on the dog's hotspot (192.168.12.1)
REM   dimos-go2.bat unitree-go2-basic                -> visualization only, NO control
REM   dimos-go2.bat unitree-go2-agentic              -> natural-language control (needs an LLM key + internet)
REM   dimos-go2.bat unitree-go2-basic 192.168.1.50   -> dog on another network, at that IP
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_dimos.sh %*
echo.
pause
