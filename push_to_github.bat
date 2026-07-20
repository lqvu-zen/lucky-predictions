@echo off
REM ============================================================
REM  Initialize git and push this project to GitHub as a PRIVATE
REM  repository. Double-click to run (on your own machine).
REM
REM  Requires git. For automatic repo creation it also uses the
REM  GitHub CLI "gh" (https://cli.github.com) — run `gh auth login`
REM  once beforehand. If gh is not installed, manual steps are shown.
REM ============================================================
setlocal
cd /d "%~dp0"

set "REPONAME=vietlott-predictions"

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git is not installed or not on PATH.
    echo Get it from https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Clean any partial/locked repo state left from an earlier attempt.
if exist ".git" (
    echo Removing existing .git folder to start clean...
    rmdir /s /q ".git"
)

echo Initializing repository...
git init
git branch -M main

REM Use your global identity if set; otherwise fall back to these.
git config user.name  >nul 2>&1 || git config user.name  "Vu"
git config user.email >nul 2>&1 || git config user.email "ktvn100@gmail.com"

git add -A
git commit -m "Initial commit: Vietlott crawler, analysis, prediction, daily automation"

where gh >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo Creating PRIVATE GitHub repo "%REPONAME%" and pushing...
    gh repo create "%REPONAME%" --private --source=. --remote=origin --push
    if errorlevel 1 (
        echo.
        echo gh failed. Have you run `gh auth login` yet? See manual steps below.
        goto manual
    )
    echo.
    echo Done. Your private repo is live.
    goto end
)

:manual
echo.
echo ============================================================
echo  gh CLI not available. Finish in two steps:
echo    1) Create an EMPTY private repo at https://github.com/new
echo       - Name: %REPONAME%
echo       - Visibility: Private
echo       - Do NOT add a README, .gitignore, or license
echo    2) Then run (replace YOUR_USERNAME):
echo         git remote add origin https://github.com/YOUR_USERNAME/%REPONAME%.git
echo         git push -u origin main
echo ============================================================

:end
pause
