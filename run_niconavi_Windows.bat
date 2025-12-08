@echo off
setlocal

cd /d "%~dp0"

set "REQUIRED_PYTHON_VERSION=3.12"

set "PYTHON_CMD="
for %%P in ("py -3.12" python3.12 python3 python) do (
    call :check_python %%~P
    if defined PYTHON_CMD goto :have_python
)
echo Python %REQUIRED_PYTHON_VERSION% was not found. Please install Python %REQUIRED_PYTHON_VERSION% and rerun this script.
exit /b 1
:have_python

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

echo Installing project dependencies via pipenv (Python %REQUIRED_PYTHON_VERSION%)...
pipenv --python %REQUIRED_PYTHON_VERSION% install
if errorlevel 1 exit /b %errorlevel%

pipenv --venv >nul 2>nul
if errorlevel 1 (
    echo pipenv failed to create a virtual environment. Please check the output above.
    exit /b 1
)

echo Starting niconavi.py inside pipenv...
pipenv run python niconavi.py

endlocal

:check_python
set "CAND=%*"
set "RAW_VER="
for /f "usebackq tokens=2 delims= " %%v in (`%CAND% -V 2^>nul`) do set "RAW_VER=%%v"
if not defined RAW_VER goto :eof
for /f "tokens=1,2 delims=." %%a in ("%RAW_VER%") do (
    if "%%a.%%b"=="%REQUIRED_PYTHON_VERSION%" (
        set "PYTHON_CMD=%CAND%"
    )
)
goto :eof
