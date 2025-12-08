@echo off
setlocal

cd /d "%~dp0"

set "PYTHON="
for %%P in (python python3 py) do (
    where %%P >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=%%P"
        goto :have_python
    )
)
echo Python was not found. Please install Python 3.x and rerun this script.
exit /b 1
:have_python

set "PIP="
for %%P in (pip pip3) do (
    where %%P >nul 2>nul
    if not errorlevel 1 (
        set "PIP=%%P"
        goto :have_pip
    )
)
%PYTHON% -m pip --version >nul 2>nul
if not errorlevel 1 (
    set "PIP=%PYTHON% -m pip"
    goto :have_pip
)
echo pip was not found. Please install pip (the Python package manager) and rerun this script.
exit /b 1
:have_pip

where pipenv >nul 2>nul
if errorlevel 1 (
    echo pipenv not found; installing with %PIP%...
    %PIP% install --user pipenv
    for /f "delims=" %%i in ('%PYTHON% -c "import site;print(site.USER_BASE)"') do set "USER_BASE=%%i"
    if defined USER_BASE (
        set "PATH=%USER_BASE%\Scripts;%USER_BASE%\bin;%PATH%"
    )
)

echo Installing project dependencies via pipenv...
pipenv install
if errorlevel 1 exit /b %errorlevel%

pipenv --venv >nul 2>nul
if errorlevel 1 (
    echo pipenv failed to create a virtual environment. Please check the output above.
    exit /b 1
)

echo Starting niconavi.py inside pipenv...
pipenv run python niconavi.py

endlocal
