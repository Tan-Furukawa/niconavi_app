@echo off
setlocal

cd /d "%~dp0"

set "MINIMUM_PYTHON_VERSION=3.12"
set "APP_URL=http://localhost:8551/app"

for /f "tokens=1,2 delims=." %%a in ("%MINIMUM_PYTHON_VERSION%") do (
    set "REQUIRED_MAJOR=%%a"
    set "REQUIRED_MINOR=%%b"
)

set "PYTHON_CMD="
set "PYTHON_EXE="
for %%P in ("py -3" python3 python) do (
    call :check_python %%~P
    if defined PYTHON_CMD goto :have_python
)
echo Python %MINIMUM_PYTHON_VERSION% or newer was not found. Please install Python %MINIMUM_PYTHON_VERSION% or later and rerun this script.
exit /b 1
:have_python
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_CMD%"

set "PIP="
%PYTHON_CMD% -m pip --version >nul 2>nul
if not errorlevel 1 (
    set "PIP=%PYTHON_CMD% -m pip"
    goto :have_pip
)
for %%P in (pip pip3) do (
    where %%P >nul 2>nul
    if not errorlevel 1 (
        set "PIP=%%P"
        goto :have_pip
    )
)
echo pip was not found. Please install pip (the Python package manager) and rerun this script.
exit /b 1
:have_pip

where pipenv >nul 2>nul
if errorlevel 1 (
    echo pipenv not found; installing with %PIP%...
    %PIP% install --user pipenv
    for /f "delims=" %%i in ('%PYTHON_CMD% -c "import site;print(site.USER_BASE)"') do set "USER_BASE=%%i"
    if defined USER_BASE (
        set "PATH=%USER_BASE%\Scripts;%USER_BASE%\bin;%PATH%"
    )
)

echo Installing project dependencies via pipenv (Python %MINIMUM_PYTHON_VERSION%+)...
pipenv --python "%PYTHON_EXE%" install
if errorlevel 1 exit /b %errorlevel%

pipenv --venv >nul 2>nul
if errorlevel 1 (
echo pipenv failed to create a virtual environment. Please check the output above.
    exit /b 1
)

echo Starting niconavi.py inside pipenv...
echo Attempting to open %APP_URL% in your browser...
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 2; $url='%APP_URL%'; $msg='Google Chrome was not found; please open %APP_URL% in your browser (clickable).'; $candidates=@('chrome.exe', \"$Env:ProgramFiles\Google\Chrome\Application\chrome.exe\", \"$Env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe\", \"$Env:LocalAppData\Google\Chrome\Application\chrome.exe\"); $chrome=$candidates | Where-Object { Test-Path $_ } | Select-Object -First 1; if ($chrome) { try { Start-Process $chrome $url } catch { Write-Host $msg; Start-Process $url } } else { Write-Host $msg; Start-Process $url }"

pipenv run python niconavi.py
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%

:check_python
set "CAND=%*"
set "RAW_VER="
for /f "usebackq tokens=2 delims= " %%v in (`%CAND% -V 2^>nul`) do set "RAW_VER=%%v"
if not defined RAW_VER goto :eof
set "MAJOR="
set "MINOR="
for /f "tokens=1,2 delims=." %%a in ("%RAW_VER%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if not defined MAJOR goto :eof
if not defined MINOR goto :eof
set /a MAJOR_INT=%MAJOR% >nul 2>nul
if errorlevel 1 goto :eof
set /a MINOR_INT=%MINOR% >nul 2>nul
if errorlevel 1 goto :eof

if %MAJOR_INT% GTR %REQUIRED_MAJOR% goto :set_python
if %MAJOR_INT% EQU %REQUIRED_MAJOR% if %MINOR_INT% GEQ %REQUIRED_MINOR% goto :set_python
goto :eof

:set_python
set "PYTHON_CMD=%CAND%"
if not defined PYTHON_EXE (
    for /f "usebackq delims=" %%p in (`%CAND% -c "import sys;print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%p"
)
goto :eof
