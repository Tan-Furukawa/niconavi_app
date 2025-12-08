#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REQUIRED_PYTHON_VERSION="3.12"

find_python_required() {
  local cmd version
  for cmd in python3.12 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version="$("$cmd" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
      if [ "$version" = "$REQUIRED_PYTHON_VERSION" ]; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_CMD="$(find_python_required || true)"
if [ -z "$PYTHON_CMD" ]; then
  cat <<EOF
Python ${REQUIRED_PYTHON_VERSION} is required but was not found.
Please install Python ${REQUIRED_PYTHON_VERSION} and ensure it is on your PATH, then retry.
EOF
  exit 1
fi

if ! command -v pipenv >/dev/null 2>&1; then
  if "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    echo "pipenv not found; installing with ${PYTHON_CMD} -m pip ..."
    "$PYTHON_CMD" -m pip install --user pipenv
    USER_BASE="$("$PYTHON_CMD" -m site --user-base 2>/dev/null || true)"
    if [ -n "$USER_BASE" ]; then
      export PATH="$USER_BASE/bin:$PATH"
    fi
  else
    echo "pipenv not found and pip is not available. Please install them and retry."
    exit 1
  fi
fi

echo "Installing project dependencies via pipenv (Python ${REQUIRED_PYTHON_VERSION})..."
pipenv --python "${REQUIRED_PYTHON_VERSION}" install

if ! pipenv --venv >/dev/null 2>&1; then
  echo "pipenv failed to create a virtual environment. Please check the output above."
  exit 1
fi

echo "Starting niconavi.py inside pipenv..."
pipenv run python niconavi.py
