@echo off
REM Go2 all-in-one: live camera + driving + tricks + voice + heel in one window (runs inside WSL). Join the dog's Wi-Fi first.
REM   go2.bat          -> connect to the dog
REM   go2.bat --demo   -> preview the window with a fake robot (no dog needed)
REM Settings live in OPTS below, so you can just double-click this file. Edit the line, save, relaunch.
REM   --motion-mode mcf : the dog's motion mode (the back-leg stand, id 2050, belongs to it). Delete it to use the default.
REM   "follow me" (T, or say it): --follow-speed 0.8 = fastest it walks, m/s (was 0.35).
REM                                --follow-height 0.78 = how close it stops: higher = closer (0.60 was the old ~2.5 m, 0.78 is ~1.3 m,
REM                                above ~0.9 the camera loses your feet).
REM   --heel-speed 1.0  : fastest the dog walks while heeling, m/s (a strolling pace is ~1.0; it falls behind you above this).
REM   heel = the "follow me" steering, holding you a little off-centre so the dog walks at your side, CLOSE (a leash length):
REM     --heel-range 1.1   : distance to hold, metres from the dog's centre to you. The lidar teaches the camera how far you are
REM                          and warns if you are closer; below ~0.9 m the camera loses your feet, so don't go much lower.
REM     --heel-offset 0.18 : how far off-centre it holds you (0 = dead ahead like follow, 0.25 = more to the side)
REM     --heel-range 0     : ignore the lidar; the camera alone holds ~1.5 m (--heel-height 0.82 to 0.9 = closer, less exact)
REM   --no-heel-lidar    : add this to switch the lidar off entirely (the camera alone then holds the distance).
REM   --heel-style geometric : the older heel (camera geometry; needs calibrating with J). Not the default.
REM   --mic dog : listen through the DOG's own microphone instead of the computer's (run dogmic-probe.bat first to see if it works).
REM   To save what the always-on mic hears (to study misses in loud places), add: --voice-log /mnt/c/Users/Sasha/go2-voice-log
REM --- PHONE (iPhone): its microphone becomes the voice mic (the wake word works through it) and its motion sensors tell follow/heel whether you
REM     have stopped or are still walking when you drop out of the camera's view (it waits for you instead of giving up). It only works while
REM     the phone's page is open and started; before that, and if the phone drops out, the laptop mic and the camera do the job.
REM     Join the phone and this PC to the dog's Wi-Fi, then read phone-setup.txt (the certificate is a once-only step, phone-firewall.bat once too).
REM     Every launch: scan the QR code in phone_qr.html (it opens by itself below) or type the link in phone_url.txt, tap Start.
REM     While the phone mic streams the quicker speech model (base.en) is used; to always use the accurate slower one add  --fast-model none  to OPTS below.
REM     To switch it off, make the next line: set PHONE=
set PHONE=1
REM --- Microphone: your AirPods. Windows captures the named mic (winmic.py, started below) and the app listens to it, whatever the Windows
REM     default input is. Put them in your ears and connect them to THIS PC (not the iPhone). If they are silent, or the helper isn't
REM     working, the app says so and uses the laptop mic instead. To use the laptop mic always, make the next line: set WINMIC=
REM     (On this Windows-on-ARM laptop the AirPods mic stays silent, so it is off; PHONE above replaces it. PHONE wins if both are set.)
REM     (First time on another PC: .venv\Scripts\python.exe -m pip install sounddevice, then list the mics: .venv\Scripts\python.exe winmic.py --list)
set WINMIC=
if defined PHONE powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'phone_server[.]py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if defined PHONE if exist "%~dp0phone_qr.html" del "%~dp0phone_qr.html" >nul 2>&1
if defined PHONE start "" /B "%~dp0.venv\Scripts\python.exe" "%~dp0phone_server.py" >nul 2>&1
if defined WINMIC powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'winmic[.]py --serve' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if defined WINMIC start "" /B "%~dp0.venv\Scripts\python.exe" "%~dp0winmic.py" --serve --device "%WINMIC%" >nul 2>&1
set OPTS=--motion-mode mcf --heel-speed 1.0
if defined WINMIC set OPTS=%OPTS% --mic win --mic-device "%WINMIC%"
if defined PHONE set OPTS=%OPTS% --phone
if defined PHONE call :showqr
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh %OPTS% %*
echo.
pause
goto :eof

:showqr
REM the phone server writes phone_qr.html a moment after it starts; open it once it is there (the first run also makes the certificates: ~10 s)
for /l %%i in (1,1,30) do (
  if exist "%~dp0phone_qr.html" (
    start "" "%~dp0phone_qr.html"
    goto :eof
  )
  ping -n 2 127.0.0.1 >nul
)
goto :eof
