#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Python was not found. Please install Python 3.x and rerun this script."
  exit 1
fi

if command -v pip3 >/dev/null 2>&1; then
  PIP_CMD=("pip3")
elif command -v pip >/dev/null 2>&1; then
  PIP_CMD=("pip")
elif $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
  PIP_CMD=("$PYTHON_CMD" -m pip)
else
  echo "pip was not found. Please install pip (the Python package manager) and rerun this script."
  exit 1
fi

if ! command -v pipenv >/dev/null 2>&1; then
  echo "pipenv not found; installing with ${PIP_CMD[*]}..."
  "${PIP_CMD[@]}" install --user pipenv
  USER_BASE="$($PYTHON_CMD -m site --user-base 2>/dev/null || true)"
  if [ -n "$USER_BASE" ]; then
    export PATH="$USER_BASE/bin:$PATH"
  fi
fi

echo "Installing project dependencies via pipenv..."
pipenv install

if ! pipenv --venv >/dev/null 2>&1; then
  echo "pipenv failed to create a virtual environment. Please check the output above."
  exit 1
fi

echo "Starting niconavi.py inside pipenv..."
pipenv run python niconavi.py
