#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MINIMUM_PYTHON_VERSION="3.12"
APP_URL="http://localhost:8551/app"

open_browser() {
  local url="$1"

  for cmd in google-chrome google-chrome-stable chromium-browser chromium chrome; do
    if command -v "$cmd" >/dev/null 2>&1; then
      "$cmd" "$url" >/dev/null 2>&1 &
      return 0
    fi
  done

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
    return 1
  fi

  return 2
}

launch_browser_async() {
  (
    sleep 2
    open_browser "$APP_URL"
    status=$?
    if [ "$status" -ne 0 ]; then
      echo "Google Chrome was not available; please open ${APP_URL} in your browser (clickable)."
    fi
  ) &
}

version_at_least() {
  local candidate="$1" required="$2"
  local cand_major="${candidate%%.*}"
  local cand_minor="${candidate#*.}"
  cand_minor="${cand_minor%%.*}"
  local req_major="${required%%.*}"
  local req_minor="${required#*.}"
  req_minor="${req_minor%%.*}"

  if [ "$cand_major" -gt "$req_major" ]; then
    return 0
  fi

  if [ "$cand_major" -eq "$req_major" ] && [ "$cand_minor" -ge "$req_minor" ]; then
    return 0
  fi

  return 1
}

find_python_required() {
  local cmd version
  for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version="$("$cmd" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
      if [ -n "$version" ] && version_at_least "$version" "$MINIMUM_PYTHON_VERSION"; then
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
Python ${MINIMUM_PYTHON_VERSION}+ is required but was not found.
Please install Python ${MINIMUM_PYTHON_VERSION} or newer and ensure it is on your PATH, then retry.
EOF
  exit 1
fi

PYTHON_EXE="$("$PYTHON_CMD" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
if [ -z "$PYTHON_EXE" ]; then
  echo "Could not determine Python executable path for ${PYTHON_CMD}."
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

echo "Installing project dependencies via pipenv (Python ${MINIMUM_PYTHON_VERSION}+)..."
pipenv --python "${PYTHON_EXE}" install

if ! pipenv --venv >/dev/null 2>&1; then
  echo "pipenv failed to create a virtual environment. Please check the output above."
  exit 1
fi

echo "Starting niconavi.py inside pipenv..."
echo "Attempting to open ${APP_URL} in your browser..."
launch_browser_async
pipenv run python niconavi.py
