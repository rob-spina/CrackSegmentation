#!/usr/bin/env bash
# Builds CrackSegmentation for Linux (Ubuntu/Debian) via PyInstaller, then
# wraps it into a proper installable .deb package. Must be run ON a Linux
# machine -- PyInstaller cannot cross-compile a Linux binary from macOS or
# Windows.
#
# Usage:
#   cd packaging
#   chmod +x build_linux.sh
#   ./build_linux.sh
#
# Output:
#   packaging/dist/CrackSegmentation/CrackSegmentation   (raw onedir build,
#                                                          runs directly)
#   packaging/dist/crackSegmentation_1.0.0_amd64.deb      (installable
#                                                          package -- adds
#                                                          it to the
#                                                          Applications
#                                                          menu too)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
APP_NAME="CrackSegmentation"
PKG_NAME="cracksegmentation"
# A timestamp-based Debian revision (the part after the hyphen) makes
# every build's version string strictly newer than the last -- without
# this, re-running the script and reinstalling produced a .deb with the
# exact same version apt had already seen, so `apt install` silently did
# nothing at all, leaving the OLD binary in place even though the new
# .deb on disk had a real fix in it (USER REPORT: a fix confirmed to be
# in the source, and in the freshly-built .deb, simply never took effect
# after "reinstalling").
PKG_VERSION="1.0.0-$(date +%Y%m%d%H%M%S)"
PKG_ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"

echo "== Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN") =="

# Unlike Windows/macOS, Python on Linux does NOT bundle Tkinter with the
# interpreter itself -- it's a separate system package. If this check
# fails, PyInstaller will either fail outright or silently produce a
# broken build (crack_segmentation_gui.py needs a REAL _tkinter to bundle,
# not just the pure-Python tkinter package).
if ! $PYTHON_BIN -c "import tkinter" >/dev/null 2>&1; then
    echo "ERROR: this Python doesn't have Tkinter available." >&2
    echo "Install it first (Debian/Ubuntu):" >&2
    echo "    sudo apt update && sudo apt install python3-tk" >&2
    exit 1
fi

echo "== Setting up an isolated build environment (.venv) =="
# A venv keeps this build's numpy/opencv/etc. fully separate from whatever
# is already on the system -- e.g. an apt-installed matplotlib built
# against a different numpy ABI than pip would install here, which is
# exactly what "ImportError: numpy.core.multiarray failed to import"
# means (a real, confirmed conflict on a fresh Ubuntu install). The venv
# is created from the same $PYTHON_BIN checked for Tkinter above, so it
# inherits that same real _tkinter binding.
if ! $PYTHON_BIN -c "import venv" >/dev/null 2>&1; then
    echo "ERROR: this Python doesn't have the venv module available." >&2
    echo "Install it first (Debian/Ubuntu):" >&2
    echo "    sudo apt update && sudo apt install python3-venv" >&2
    exit 1
fi
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
fi
PYTHON_BIN="$(pwd)/.venv/bin/python3"
echo "Using $($PYTHON_BIN --version) from .venv"

echo "== Installing/upgrading build dependencies (inside .venv) =="
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install --upgrade -r ../requirements.txt

echo "== Cleaning previous build artifacts =="
rm -rf build dist

echo "== Running PyInstaller =="
$PYTHON_BIN -m PyInstaller crack_segmentation_gui.spec --noconfirm

BUILD_DIR="dist/${APP_NAME}"
if [ ! -f "${BUILD_DIR}/${APP_NAME}" ]; then
    echo "ERROR: PyInstaller did not produce ${BUILD_DIR}/${APP_NAME} -- check the log above." >&2
    exit 1
fi

echo "== PyInstaller build done =="
echo "Folder: $(pwd)/${BUILD_DIR}/"
echo "Run directly with: $(pwd)/${BUILD_DIR}/${APP_NAME}"

echo "== Building the .deb package =="
if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb not found -- skipping .deb packaging (this is normally"
    echo "preinstalled on Ubuntu/Debian; on other distros install 'dpkg')."
    echo "The raw build above still runs fine on its own."
    exit 0
fi

STAGE_DIR="dist/${PKG_NAME}_${PKG_VERSION}_${PKG_ARCH}"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/DEBIAN"
mkdir -p "$STAGE_DIR/usr/lib/${PKG_NAME}"
mkdir -p "$STAGE_DIR/usr/bin"
mkdir -p "$STAGE_DIR/usr/share/applications"

# The full onedir build (the .exe-equivalent binary plus every bundled
# library it needs) goes under /usr/lib -- the standard place for a Linux
# package's own private files that aren't meant to be run directly from a
# user's $PATH.
cp -r "${BUILD_DIR}/." "$STAGE_DIR/usr/lib/${PKG_NAME}/"

# A tiny wrapper in /usr/bin is what actually ends up on $PATH -- running
# `crackSegmentation` from any terminal, or double-clicking the app's
# entry in the desktop's Applications menu (see the .desktop file below),
# both go through this.
cat > "$STAGE_DIR/usr/bin/${PKG_NAME}" << EOF
#!/usr/bin/env bash
exec "/usr/lib/${PKG_NAME}/${APP_NAME}" "\$@"
EOF
chmod +x "$STAGE_DIR/usr/bin/${PKG_NAME}"
chmod +x "$STAGE_DIR/usr/lib/${PKG_NAME}/${APP_NAME}"

cat > "$STAGE_DIR/usr/share/applications/${PKG_NAME}.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Crack Segmentation
Comment=Crack & detachment segmentation tool for facade inspection photos
Exec=/usr/bin/${PKG_NAME}
Terminal=false
Categories=Graphics;Science;
EOF

INSTALLED_SIZE_KB=$(du -sk "$STAGE_DIR/usr" | cut -f1)

cat > "$STAGE_DIR/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${PKG_VERSION}
Section: graphics
Priority: optional
Architecture: ${PKG_ARCH}
Installed-Size: ${INSTALLED_SIZE_KB}
Maintainer: Unspecified <unspecified@example.com>
Description: Crack & detachment segmentation tool
 Interactive tool for manual segmentation of cracks and detachments on
 building facade photos, with A*-guided tracing and a graphical
 control panel. Self-contained (PyInstaller onedir build) -- no separate
 Python install required on the target machine.
EOF

DEB_OUTPUT="dist/${PKG_NAME}_${PKG_VERSION}_${PKG_ARCH}.deb"
dpkg-deb --build --root-owner-group "$STAGE_DIR" "$DEB_OUTPUT"

echo
echo "== Done =="
echo "Installable package: $(pwd)/${DEB_OUTPUT}"
echo
echo "Install it with:"
echo "    sudo apt install $(pwd)/${DEB_OUTPUT}"
echo "(or: sudo dpkg -i $(pwd)/${DEB_OUTPUT} && sudo apt-get install -f"
echo " to also pull in any missing system dependency automatically)"
echo
echo "After installing, launch it by typing '${PKG_NAME}' in a"
echo "terminal, or find \"Crack Segmentation\" in the Applications menu."
echo
echo "This .deb is unsigned -- some systems may warn about installing a"
echo "package from an unknown source. That's expected for a locally built"
echo "package without a GPG-signed repository behind it, not a sign of a"
echo "problem with the build."
