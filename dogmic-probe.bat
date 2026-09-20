@echo off
REM Records ~44 s of the dog's OWN microphone and shows what the speech model makes of it. NEVER moves the dog.
REM Join the dog's Wi-Fi and close the phone app / go2.bat first (the dog accepts one controller at a time).
REM Double-click it. The window tells you what to do and when. Files are saved in THIS folder: dogmic_recording.wav and dogmic_probe_report.txt
cd /d "%~dp0"
del "%~dp0dogmic_recording.wav" 2>nul
del "%~dp0dogmic_probe_report.txt" 2>nul
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env && set +a && cd /mnt/c/Users/Sasha/go2-robot && python dogmic_probe.py %*"
echo.
echo ============================================================
if exist "%~dp0dogmic_recording.wav" echo RECORDING SAVED: %~dp0dogmic_recording.wav
if not exist "%~dp0dogmic_recording.wav" echo No recording was saved - the report says why.
if exist "%~dp0dogmic_probe_report.txt" echo REPORT: %~dp0dogmic_probe_report.txt
echo You don't need to send anything: just tell Claude it finished.
echo ============================================================
explorer "%~dp0"
pause
