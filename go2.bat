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
set OPTS=--motion-mode mcf --heel-speed 1.0
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh %OPTS% %*
echo.
pause
