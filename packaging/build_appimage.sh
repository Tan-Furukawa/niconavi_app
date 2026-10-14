#!/usr/bin/env bash
# Build a self-contained NicoNavi AppImage for Ubuntu/Linux x86_64.
#
# Usage:
#   packaging/build_appimage.sh [OUTPUT_DIR]
#
# Environment overrides:
#   PYTHON_VERSION=3.12.12
#   APP_VERSION=0.2.3
#   PACKAGING_CACHE_DIR=/path/to/cache
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.12}"
APP_VERSION="${APP_VERSION:-$(cd "$PROJECT_ROOT" && uv version --short)}"
OUTPUT_DIR="${1:-$PROJECT_ROOT/dist/appimage}"
PACKAGING_CACHE_DIR="${PACKAGING_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/niconavi-packaging}"
BUILD_ROOT="$PROJECT_ROOT/build/packaging/appimage"
APPDIR="$BUILD_ROOT/NicoNavi.AppDir"
PYTHON_STORE="$PACKAGING_CACHE_DIR/python"
REQUIREMENTS_FILE="$BUILD_ROOT/requirements.lock.txt"
NICONAVI_ROOT="$APPDIR/usr/lib/niconavi"
RUNTIME_DIR="$NICONAVI_ROOT/runtime"
LIB_DIR="$NICONAVI_ROOT/lib"

APPIMAGETOOL_VERSION="1.9.1"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
APPIMAGE_RUNTIME_VERSION="20251108"
APPIMAGE_RUNTIME_SHA256="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"
APPIMAGETOOL="$PACKAGING_CACHE_DIR/appimagetool-$APPIMAGETOOL_VERSION-x86_64.AppImage"
APPIMAGE_RUNTIME="$PACKAGING_CACHE_DIR/runtime-$APPIMAGE_RUNTIME_VERSION-x86_64"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "This script currently builds the Linux x86_64 AppImage only." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH. Install uv before running this script." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl was not found on PATH." >&2
    exit 1
fi

download_verified() {
    local url="$1"
    local destination="$2"
    local expected_sha256="$3"

    if [ -f "$destination" ] && printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --check --status; then
        return
    fi

    local temporary
    temporary="$(mktemp "${destination}.tmp.XXXXXX")"
    if ! curl --fail --location --retry 3 --output "$temporary" "$url"; then
        rm -f -- "$temporary"
        return 1
    fi
    printf '%s  %s\n' "$expected_sha256" "$temporary" | sha256sum --check --status
    mv -f -- "$temporary" "$destination"
}

rm -rf -- "$BUILD_ROOT"
mkdir -p "$RUNTIME_DIR" "$LIB_DIR" "$OUTPUT_DIR" "$PACKAGING_CACHE_DIR"

echo "==> Installing python-build-standalone CPython $PYTHON_VERSION"
UV_PYTHON_INSTALL_DIR="$PYTHON_STORE" UV_PYTHON_INSTALL_BIN=0 \
    uv python install --managed-python --no-bin --install-dir "$PYTHON_STORE" "$PYTHON_VERSION"
MANAGED_PYTHON="$({
    UV_PYTHON_INSTALL_DIR="$PYTHON_STORE" UV_PYTHON_INSTALL_BIN=0 \
        uv python find --no-project --managed-python "$PYTHON_VERSION"
} | tail -n 1)"
MANAGED_RUNTIME_DIR="$(dirname "$MANAGED_PYTHON")/.."
MANAGED_RUNTIME_DIR="$(cd "$MANAGED_RUNTIME_DIR" && pwd)"
cp -a "$MANAGED_RUNTIME_DIR/." "$RUNTIME_DIR/"

STAGE_PYTHON="$RUNTIME_DIR/bin/python3"
if [ ! -x "$STAGE_PYTHON" ]; then
    echo "The standalone runtime does not contain bin/python3." >&2
    exit 1
fi

echo "==> Exporting dependencies from uv.lock"
(
    cd "$PROJECT_ROOT"
    uv export --quiet --frozen --no-dev --no-emit-project \
        --python "$STAGE_PYTHON" --output-file "$REQUIREMENTS_FILE"
)

echo "==> Installing locked Linux dependencies"
uv pip install --quiet --python "$STAGE_PYTHON" --target "$LIB_DIR" \
    --require-hashes --only-binary :all: --requirements "$REQUIREMENTS_FILE"

echo "==> Installing NicoNavi"
uv pip install --quiet --python "$STAGE_PYTHON" --target "$LIB_DIR" \
    --no-deps "$PROJECT_ROOT"

install -m 0644 "$PROJECT_ROOT/packaging/desktop_launcher.py" "$NICONAVI_ROOT/launcher.py"
install -m 0644 "$PROJECT_ROOT/LICENSE" "$NICONAVI_ROOT/LICENSE.txt"
install -m 0755 "$PROJECT_ROOT/packaging/linux/AppRun" "$APPDIR/AppRun"
install -m 0644 "$PROJECT_ROOT/packaging/linux/io.github.tan_furukawa.NicoNavi.desktop" \
    "$APPDIR/io.github.tan_furukawa.NicoNavi.desktop"
install -m 0644 "$PROJECT_ROOT/src/niconavi_app/assets/favicon.png" "$APPDIR/niconavi.png"

mkdir -p \
    "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
    "$APPDIR/usr/share/metainfo"
install -m 0644 "$APPDIR/io.github.tan_furukawa.NicoNavi.desktop" \
    "$APPDIR/usr/share/applications/io.github.tan_furukawa.NicoNavi.desktop"
install -m 0644 "$PROJECT_ROOT/src/niconavi_app/assets/favicon.png" \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps/niconavi.png"
install -m 0644 "$PROJECT_ROOT/packaging/linux/io.github.tan_furukawa.NicoNavi.appdata.xml" \
    "$APPDIR/usr/share/metainfo/io.github.tan_furukawa.NicoNavi.appdata.xml"

echo "X-AppImage-Version=$APP_VERSION" >> "$APPDIR/io.github.tan_furukawa.NicoNavi.desktop"

echo "==> Running an import smoke test"
SMOKE_DATA="$BUILD_ROOT/smoke-data"
mkdir -p "$SMOKE_DATA/uploads"
PYTHONPATH="$LIB_DIR" FLET_UPLOAD_DIR="$SMOKE_DATA/uploads" \
    "$STAGE_PYTHON" -c 'import niconavi_app.main; print("NicoNavi import: OK")'
PYTHONPATH="$LIB_DIR" XDG_DATA_HOME="$SMOKE_DATA" \
    "$STAGE_PYTHON" -c 'import flet, runpy, sys; flet.app = lambda **kwargs: print("Packaged launcher: OK"); runpy.run_path(sys.argv[1], run_name="__main__")' \
    "$NICONAVI_ROOT/launcher.py"

if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$APPDIR/io.github.tan_furukawa.NicoNavi.desktop"
fi

echo "==> Downloading verified AppImage tooling"
download_verified \
    "https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-x86_64.AppImage" \
    "$APPIMAGETOOL" "$APPIMAGETOOL_SHA256"
download_verified \
    "https://github.com/AppImage/type2-runtime/releases/download/$APPIMAGE_RUNTIME_VERSION/runtime-x86_64" \
    "$APPIMAGE_RUNTIME" "$APPIMAGE_RUNTIME_SHA256"
chmod +x "$APPIMAGETOOL" "$APPIMAGE_RUNTIME"

OUTPUT_FILE="$OUTPUT_DIR/NicoNavi-$APP_VERSION-x86_64.AppImage"
rm -f -- "$OUTPUT_FILE"

echo "==> Creating AppImage"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 VERSION="$APP_VERSION" \
    "$APPIMAGETOOL" --runtime-file "$APPIMAGE_RUNTIME" "$APPDIR" "$OUTPUT_FILE"
chmod +x "$OUTPUT_FILE"

echo "Done: $OUTPUT_FILE"
du -h "$OUTPUT_FILE"
