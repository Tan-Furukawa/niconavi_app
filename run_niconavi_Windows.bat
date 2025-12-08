@echo off
setlocal

cd /d "%~dp0"

set "APP_URL=http://localhost:8551/app"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH. Please install it first, for example:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    exit /b 1
)

echo Syncing dependencies with uv (Python 3.12+)...
uv sync
if errorlevel 1 exit /b %errorlevel%

echo Attempting to open %APP_URL% in Google Chrome...
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 2; $url='%APP_URL%'; $msg='Google Chrome was not found; please open %APP_URL% in your browser (clickable).'; $candidates=@('chrome.exe', \"$Env:ProgramFiles\Google\Chrome\Application\chrome.exe\", \"$Env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe\", \"$Env:LocalAppData\Google\Chrome\Application\chrome.exe\"); $chrome=$candidates | Where-Object { Test-Path $_ } | Select-Object -First 1; if ($chrome) { try { Start-Process $chrome $url } catch { Write-Host $msg; Start-Process $url } } else { Write-Host $msg; Start-Process $url }"

echo Running niconavi.py via uv...
uv run python niconavi.py
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
