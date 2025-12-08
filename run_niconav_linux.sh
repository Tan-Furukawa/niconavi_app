#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

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

  echo "Google Chrome was not available; please open ${url} in your browser (clickable)."
  return 2
}

launch_browser_async() {
  (
    sleep 2
    open_browser "$APP_URL"
  ) &
}

if ! command -v uv >/dev/null 2>&1; then
  cat <<'EOF_MISSING_UV'
uv was not found on PATH. Please install it first, for example:

  curl -Ls https://astral.sh/uv/install.sh | sh
EOF_MISSING_UV
  exit 1
fi

echo "Syncing dependencies with uv (Python 3.12+)..."
uv sync

echo "Starting niconavi.py via uv..."
echo "Attempting to open ${APP_URL} in Google Chrome..."
launch_browser_async
uv run python niconavi.py
