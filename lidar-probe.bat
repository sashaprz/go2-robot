@echo off
REM Records what the dog's lidar sees for 60 s, so a person tracker can be built on real data. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click it (from Explorer is fine). The window tells you what to do and when, and at the end where the files are.
REM Files are always saved in THIS folder (the one this .bat is in): lidar_recording.npz and lidar_probe_report.txt
cd /d "%~dp0"
del "%~dp0lidar_recording.npz" 2>nul
del "%~dp0lidar_probe_report.txt" 2>nul
set ARGS=%*
if "%ARGS%"=="" set ARGS=60 save
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python lidar_probe.py %ARGS%"
echo.
echo ============================================================
if exist "%~dp0lidar_recording.npz" echo RECORDING SAVED: %~dp0lidar_recording.npz
if not exist "%~dp0lidar_recording.npz" echo No recording was saved - the report below says why.
if exist "%~dp0lidar_probe_report.txt" echo REPORT: %~dp0lidar_probe_report.txt
echo You don't need to send anything: just tell Claude it finished.
echo ============================================================
explorer "%~dp0"
pause
