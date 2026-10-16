# NicoNavi v0.3.2-beta
[![DOI](https://zenodo.org/badge/1111962451.svg)](https://doi.org/10.5281/zenodo.17861277)

see https://tan-furukawa.github.io/niconavi/

## Installation

Download the appropriate package from [GitHub Releases](https://github.com/Tan-Furukawa/niconavi_app/releases/tag).

### Windows

Download `NicoNavi-Setup.exe`, run it, and follow the installer instructions.

### Linux x86_64

Download `NicoNavi-<version>-x86_64.AppImage`, make it executable, and run it:

```bash
chmod +x NicoNavi-*-x86_64.AppImage
./NicoNavi-*-x86_64.AppImage
```

### Python (macOS and other platforms)

On macOS or other operating systems without a prebuilt package, run NicoNavi from Python. This method can also be used on Windows and Linux. Install [uv](https://docs.astral.sh/uv/), clone this repository, and run NicoNavi with the locked dependencies:

```bash
git clone https://github.com/Tan-Furukawa/niconavi_app.git
cd niconavi_app
uv sync --frozen
uv run python niconavi.py
```
