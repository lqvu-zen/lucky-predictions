@echo off
REM ============================================================
REM  Next-draw prediction from EVERY model, both games, LIVE:
REM    - heuristic strategies for fun
REM    - ML per-number: logreg and gb
REM    - ML positional / ordered
REM    - ML joint number x position
REM  ML per-number runs with --no-log so it does not touch the ledger.
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
    echo [1/5] heuristic strategies
    uv run python run.py predict %%G --strategy all
    echo.
    echo [2/5] ML per-number: logistic regression
    uv run python run.py ml-predict %%G --model logreg --no-log
    echo.
    echo [3/5] ML per-number: gradient boosting
    uv run python run.py ml-predict %%G --model gb --no-log
    echo.
    echo [4/5] ML positional / ordered
    uv run python run.py ml-predict-pos %%G
    echo.
    echo [5/5] ML joint number x position
    uv run python run.py ml-predict-joint %%G
)

echo.
echo ============================================================
echo  Done. To also save a copy, run from PowerShell:
echo    .\predict_all.bat ^| Tee-Object reports\latest_predictions.txt
echo ============================================================
pause
