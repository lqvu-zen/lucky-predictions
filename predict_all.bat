@echo off
REM ============================================================
REM  Next-draw prediction from every model, both games, LIVE:
REM    - positional / ordered: ridge and gb
REM    - joint number x position
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

echo Refreshing dependencies...
uv sync --extra ml >nul 2>&1
echo Generating predictions from every model. Live output below.
echo ============================================================

for %%G in (power_655 power_645) do (
    echo.
    echo #################### %%G ####################
    echo.
    echo [1/4] for-fun heuristic strategies
    uv run python run.py predict %%G --strategy all
    echo.
    echo [2/4] positional / ordered: ridge
    uv run python run.py ml-predict-pos %%G --model ridge
    echo.
    echo [3/4] positional / ordered: gradient boosting
    uv run python run.py ml-predict-pos %%G --model gb
    echo.
    echo [4/4] joint number x position
    uv run python run.py ml-predict-joint %%G
)

echo.
echo ============================================================
echo  Done. To also save a copy, run from PowerShell:
echo    .\predict_all.bat ^| Tee-Object reports\latest_predictions.txt
echo ============================================================
pause
