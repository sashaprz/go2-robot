@echo off
REM Go2 all-in-one: live camera + driving + tricks + voice + heel in one window (runs inside WSL). Join the dog's Wi-Fi first.
REM   go2.bat          -> connect to the dog
REM   go2.bat --demo   -> preview the window with a fake robot (no dog needed)
REM Settings live in OPTS below, so you can just double-click this file. Edit the line, save, relaunch.
REM   --motion-mode mcf : the dog's motion mode (the back-leg stand, id 2050, belongs to it). Delete it to use the default.
REM   --heel-speed 0.6  : fastest the dog walks while heeling, m/s (default 0.8; a normal walking pace is ~1.2).
REM   heel uses the camera AND the lidar by default. To use the camera only, add: --no-heel-lidar
REM   heel distance: --heel-lead 1.1 (m ahead of the dog's centre) and --heel-gap 0.5 (m sideways) are the defaults. Smaller = closer,
REM     e.g. --heel-lead 0.9 --heel-gap 0.45, but the camera then sees less of you (lens is only 35 cm off the floor); see README.
REM   To save what the always-on mic hears (to study misses in loud places), add: --voice-log /mnt/c/Users/Sasha/go2-voice-log
set OPTS=--motion-mode mcf --heel-speed 0.6
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh %OPTS% %*
echo.
pause
