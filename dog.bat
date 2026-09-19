@echo off
REM Go2 controller launcher. Join the Go2_61331_29d4be72 Wi-Fi first (password 88888888).
cd /d "%~dp0"
".venv\Scripts\python.exe" -W ignore dog.py
echo.
pause
