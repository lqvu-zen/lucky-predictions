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

REM Show progress live on screen AND append it to the log (Tee-Object).
REM PYTHONUNBUFFERED makes Python flush immediately through the pipe, so the
REM step banners appear as they happen instead of all at once at the end.
REM --extra ml pulls in numpy/scikit-learn so the predict->score loop runs
set PYTHONUNBUFFERED=1
powershell -NoProfile -Command "uv run --extra ml python run.py daily 2>&1 | Tee-Object -Append -FilePath 'logs\daily.log'"

REM Publish: commit new draws + predictions and push so the live site updates.
REM (Your Vietnam IP crawls fine; GitHub's runners are blocked, so we push
REM  from here instead of relying on the cloud crawl.)
git add data predictions >> "logs\daily.log" 2>&1
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "daily update %date%" >> "logs\daily.log" 2>&1
    REM integrate any remote changes first so the push isn't rejected
    git pull --rebase >> "logs\daily.log" 2>&1
    git push >> "logs\daily.log" 2>&1 && (echo [%date% %time%] pushed new data+predictions >> "logs\daily.log") || (echo [%date% %time%] PUSH FAILED - run: git pull --rebase then git push >> "logs\daily.log")
) else (
    echo [%date% %time%] no new draws to publish >> "logs\daily.log"
)
echo [%date% %time%] finished >> "logs\daily.log"
