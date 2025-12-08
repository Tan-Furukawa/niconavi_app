#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REQUIRED_PYTHON_VERSION="3.12"
APP_URL="http://localhost:8551/app"

open_browser() {
  local url="$1"

  if open -Ra "Google Chrome" >/dev/null 2>&1; then
    open -a "Google Chrome" "$url" >/dev/null 2>&1 && return 0
  fi
  open "$url" >/dev/null 2>&1
  return $?
}

launch_browser_async() {
  (
    sleep 2
    open_browser "$APP_URL" || echo "Could not automatically open $APP_URL; please open it manually."
  ) &
}

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
Install it via Homebrew, for example:

  brew install python@3.12

and ensure it is on your PATH, then retry.
EOF
  exit 1
fi

# mac は pipenv を Homebrew 管理にしておく前提
if ! command -v pipenv >/dev/null 2>&1; then
  cat <<EOF
pipenv was not found. Please install it via Homebrew and retry:

  brew install pipenv

EOF
  exit 1
fi

echo "Installing project dependencies via pipenv (Python ${REQUIRED_PYTHON_VERSION})..."
pipenv --python "${REQUIRED_PYTHON_VERSION}" install

if ! pipenv --venv >/dev/null 2>&1; then
  echo "pipenv failed to create a virtual environment. Please check the output above."
  exit 1
fi

echo "Starting niconavi.py inside pipenv..."
echo "Attempting to open ${APP_URL} in your browser..."
launch_browser_async
pipenv run python niconavi.py
