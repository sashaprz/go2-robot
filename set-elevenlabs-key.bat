@echo off
REM Save your ElevenLabs API key for voice commands (hidden prompt; stored in WSL ~/.dimos.env, never in the repo).
REM Double-click from Explorer or run in a normal terminal (it needs keyboard input).
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/set_elevenlabs_key.sh
echo.
pause
