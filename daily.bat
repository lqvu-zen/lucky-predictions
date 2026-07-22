@echo off
REM ============================================================
REM  Daily job (via uv): crawl latest draws, analyze, write report.
REM  This is what the scheduled task runs each evening.
REM  Output and errors are appended to logs\daily.log
REM  `uv run` auto-creates/syncs the environment if needed.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

echo. >> "logs\daily.log"
echo [%date% %time%] starting daily run >> "logs\daily.log"
REM --extra ml pulls in numpy/scikit-learn so the predict->score loop runs
uv run --extra ml python run.py daily >> "logs\daily.log" 2>&1
echo [%date% %time%] finished >> "logs\daily.log"
