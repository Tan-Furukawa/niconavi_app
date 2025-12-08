# NicoNavi

## What is this?
https://tan-furukawa.github.io/niconavi/

## Requirements
- uv installed and on PATH
- Google Chrome is recommended for the UI (scripts will try to open it)

## Install uv
- macOS/Linux:
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell):
  - `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Quick start
1) Ensure uv is installed (see above).
2) Run the launcher for your OS:
   - macOS: `./run_niconavi_mac.sh`
   - Linux: `./run_niconav_linux.sh`
   - Windows: `run_niconavi_Windows.bat`
   - Each launcher runs `uv sync` (if needed), then starts `uv run python niconavi.py` and tries to open Chrome; if Chrome is missing, it prints the app URL.

## Manual workflow
- Install deps: `uv sync`
- Run app: `uv run python niconavi.py`
- App URL: `http://localhost:8551/app`
