#!/usr/bin/env bash
#
# Build a self-contained Linux bundle of the app.
#
# The target machine needs neither Python nor uv: the bundle carries its own
# interpreter (a python-build-standalone build, the same one uv installs when
# it manages a Python for you) and every dependency underneath it. uv does the
# two jobs it is good at - fetching that interpreter and resolving/installing
# the dependency tree - and the result is a directory you can copy anywhere.
#
#   packaging/build_linux.sh [OUTPUT_DIR]
#
# The layout it writes:
#
#   <bundle>/python/     the standalone CPython runtime
#   <bundle>/lib/        niconavi_app and every dependency
#   <bundle>/niconavi    the launcher
#   <bundle>/README.txt  how to run it
#
# Dependencies go in lib/ rather than into a virtualenv on purpose. A venv
# records the absolute path of the interpreter it was built from, so it breaks
# the moment the bundle is moved or unpacked somewhere else; a plain directory
# on PYTHONPATH has no such memory, and the runtime itself is relocatable by
# design.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
BUNDLE="${1:-$APP_DIR/dist/niconavi-linux-x86_64}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH. Install it first:" >&2
    echo "  curl -Ls https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "Building into $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"

# --managed-python so this never picks up a system or pyenv interpreter: the
# point of the bundle is that the runtime is the one we shipped.
echo "==> Fetching the standalone CPython $PYTHON_VERSION runtime"
UV_PYTHON_INSTALL_DIR="$BUNDLE/.runtime" uv python install \
    --managed-python "$PYTHON_VERSION"
# uv names the directory after the exact build it resolved
# (cpython-3.12.12-linux-x86_64-gnu); the bundle just calls it python.
runtime_dir="$(find "$BUNDLE/.runtime" -maxdepth 1 -mindepth 1 -type d -name 'cpython-*' | head -1)"
if [ -z "$runtime_dir" ]; then
    echo "uv did not install a CPython runtime under $BUNDLE/.runtime" >&2
    exit 1
fi
mv "$runtime_dir" "$BUNDLE/python"
rm -rf "$BUNDLE/.runtime"

echo "==> Installing the app and its dependencies"
# Resolved against the bundled interpreter, so the wheels picked are the ones
# that runtime can import - not the ones this machine's Python would want.
uv pip install --quiet \
    --python "$BUNDLE/python/bin/python3" \
    --target "$BUNDLE/lib" \
    "$APP_DIR"

# The console scripts a --target install drops in lib/bin carry this machine's
# shebang, so they would not run anywhere else. The launcher below calls the
# module instead, and leaving broken scripts in the bundle only invites
# someone to try them.
rm -rf "$BUNDLE/lib/bin"

echo "==> Writing the launcher"
cat > "$BUNDLE/niconavi" <<'LAUNCHER'
#!/usr/bin/env bash
#
# Run the bundled app. Every path is derived from this script's own location,
# so the bundle can live anywhere.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Uploads and other user data go under the user's data directory, not inside
# the bundle: a bundle unpacked into /opt is read-only for the user running it.
DATA_DIR="${NICONAVI_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/niconavi}"
export FLET_UPLOAD_DIR="${FLET_UPLOAD_DIR:-$DATA_DIR/uploads}"
mkdir -p "$FLET_UPLOAD_DIR"

export PYTHONPATH="$HERE/lib${PYTHONPATH:+:$PYTHONPATH}"

# The app serves a web UI unless --desktop is passed. Announce the address
# before handing over, because uvicorn's own line scrolls past under the
# app's logging.
case " $* " in
    *" --desktop "*) ;;
    *) echo "niconavi: http://localhost:8551/app" ;;
esac

exec "$HERE/python/bin/python3" -u -m niconavi_app.main "$@"
LAUNCHER
chmod +x "$BUNDLE/niconavi"

cat > "$BUNDLE/README.txt" <<'README'
niconavi - self-contained Linux build
=====================================

Nothing to install. Unpack this directory anywhere and run:

    ./niconavi

It serves the interface at http://localhost:8551/app - open that in a
browser. Chrome and Chromium are the ones it is tested against.

    ./niconavi --port 9000     serve somewhere else
    ./niconavi --host 0.0.0.0  reachable from another machine on the network
    ./niconavi --desktop       native window instead of a browser tab

Uploaded files and other user data are written to
$XDG_DATA_HOME/niconavi (in practice ~/.local/share/niconavi). Set
NICONAVI_DATA_DIR to put them somewhere else.

What is in here
---------------
    python/   a standalone CPython runtime - the bundle uses this one, not
              whatever Python the machine has
    lib/      the app and every library it needs
    niconavi  the launcher

Requires glibc 2.17 or newer (CentOS 7 and later, Ubuntu 14.04 and later) on
x86_64.
README

echo
echo "Done: $BUNDLE"
du -sh "$BUNDLE"
