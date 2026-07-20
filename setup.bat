@echo off
REM ============================================================
REM  One-time setup using uv: creates the environment and installs
REM  dependencies from pyproject.toml. Double-click this once.
REM ============================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: uv is not installed or not on PATH.
    echo Install it, then re-run this file:
    echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo   or see https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo Syncing environment with uv...
uv sync
if errorlevel 1 (
    echo.
    echo ERROR: uv sync failed. See messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete.
echo  Next: double-click install_schedule.bat to run every evening,
echo  or run daily.bat now to test it.
echo ============================================================
pause
