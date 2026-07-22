@echo off
REM ============================================================
REM  Train + evaluate the models on both games, with LIVE output.
REM  Models: positional (ordered) and joint number x position.
REM  Models train on the fly, so this runs each walk-forward backtest.
REM ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv not found. See the README Setup section.
    pause
    exit /b 1
)

echo Installing / refreshing ML dependencies: uv sync --extra ml ...
uv sync --extra ml
echo.
echo Training + evaluating all models. Live progress below.
echo ============================================================

for %%G in (power_655 power_645) do (
    echo.
    echo #################### %%G ####################
    echo.
    echo [1/2] positional / ordered model: ridge + gb
    uv run python run.py ml-backtest-pos %%G --model both
    echo.
    echo [2/2] joint number x position model
    uv run python run.py ml-backtest-joint %%G
)

echo.
echo ============================================================
echo  Done. To also save a copy, run from PowerShell:
echo    .\train_all.bat ^| Tee-Object logs\train_all.log
echo ============================================================
pause
