@echo off
REM ============================================================
REM  Registers a Windows scheduled task that runs daily.bat
REM  every evening at 21:00 (9 PM). Double-click this once.
REM  To pick another time, change 21:00 below and re-run.
REM ============================================================
setlocal
cd /d "%~dp0"

set "TASKNAME=VietlottDaily"
set "RUNTIME=21:00"
set "SCRIPT=%~dp0daily.bat"

schtasks /create /tn "%TASKNAME%" /tr "\"%SCRIPT%\"" /sc daily /st %RUNTIME% /f
if %errorlevel%==0 (
    echo.
    echo ============================================================
    echo  Scheduled task "%TASKNAME%" created.
    echo  It runs every day at %RUNTIME% and writes reports\latest.md
    echo.
    echo  Manage it in Task Scheduler, or:
    echo    Run now:  schtasks /run /tn %TASKNAME%
    echo    Remove:   schtasks /delete /tn %TASKNAME% /f
    echo ============================================================
) else (
    echo.
    echo Could not create the task. Right-click this file and choose
    echo "Run as administrator", then try again.
)
pause
