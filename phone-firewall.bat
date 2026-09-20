@echo off
REM Run ONCE, as administrator (right-click > Run as administrator): lets your iPhone reach the phone page this PC serves (ports 8443 and 8080),
REM from the local network only (the dog's Wi-Fi), on any Windows network profile. Nothing outside the local subnet can connect.
net session >nul 2>&1
if errorlevel 1 (
  echo This needs administrator rights: right-click phone-firewall.bat and choose "Run as administrator".
  pause
  exit /b 1
)
netsh advfirewall firewall delete rule name="Go2 phone link" >nul 2>&1
netsh advfirewall firewall add rule name="Go2 phone link" dir=in action=allow protocol=TCP localport=8443,8080 remoteip=localsubnet profile=any
echo.
echo Done. If the phone still can't open the page, check that the phone and this PC are on the SAME Wi-Fi (the dog's).
pause
