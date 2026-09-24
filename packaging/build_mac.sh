#!/usr/bin/env bash
# Builds CrackSegmentation.app AND CrackSegmentation.dmg on macOS. Must be
# run ON a Mac -- PyInstaller (and hdiutil, used below for the .dmg) cannot
# cross-compile/cross-package from Windows or Linux.
#
# Usage:
#   cd packaging
#   ./build_mac.sh
#
# Output:
#   packaging/dist/CrackSegmentation.app   (the app bundle)
#   packaging/dist/CrackSegmentation.dmg   (double-click installer: drag
#                                            the app into Applications)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
APP_NAME="CrackSegmentation"

echo "== Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN") =="

if ! $PYTHON_BIN -c "import tkinter" >/dev/null 2>&1; then
    echo "ERROR: this Python doesn't have Tkinter available." >&2
    echo "The python.org installer for macOS includes it by default." >&2
    echo "If you installed Python via Homebrew, run: brew install python-tk" >&2
    exit 1
fi

echo "== Installing/upgrading build dependencies (venv recommended) =="
$PYTHON_BIN -m pip install --upgrade -r ../requirements.txt
# PyMuPDF (embedded PDF manual panel) is licensed AGPL-3.0/commercial --
# kept in its own file so it's a visible, separate choice. Installed here
# by default so the built app has a working manual panel; comment this
# line out to build without it (the panel then just shows a fallback
# message instead of the rendered PDF -- see requirements-optional.txt).
$PYTHON_BIN -m pip install --upgrade -r ../requirements-optional.txt
# PyInstaller is a build-time-only tool, not a feature dependency -- kept
# out of both requirements files above so a plain `pip install -r
# requirements.txt` (to just run the tool from source) never pulls it in.
$PYTHON_BIN -m pip install --upgrade "pyinstaller>=6.3"

echo "== Cleaning previous build artifacts =="
rm -rf build dist

echo "== Running PyInstaller =="
$PYTHON_BIN -m PyInstaller crack_segmentation_gui.spec --noconfirm

APP_PATH="dist/${APP_NAME}.app"
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: PyInstaller did not produce $APP_PATH -- check the log above." >&2
    exit 1
fi

echo "== Building the .dmg (drag-to-Applications installer) =="
DMG_STAGING="dist/.dmg_staging"
DMG_PATH="dist/${APP_NAME}.dmg"
rm -rf "$DMG_STAGING" "$DMG_PATH"
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

hdiutil create -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGING" \
    -ov -format UDZO \
    "$DMG_PATH"

rm -rf "$DMG_STAGING"

echo
echo "== Done =="
echo "App bundle: $(pwd)/${APP_PATH}"
echo "Installer:  $(pwd)/${DMG_PATH}"
echo
echo "To install: open the .dmg -- double-click it in Finder, or from a"
echo "terminal run 'open ${DMG_PATH}' (do NOT run it with ./${DMG_PATH##*/} or"
echo "'sh ${DMG_PATH##*/}': a .dmg is a disk image, not a script -- the shell"
echo "will try to read its binary contents as commands and print garbage/"
echo "syntax errors). Either way this mounts the disk image and opens a"
echo "Finder window -- drag CrackSegmentation into the Applications shortcut"
echo "shown, then eject the disk image."
echo
echo "First launch on a Mac other than the one that built it will likely be"
echo "blocked by Gatekeeper (the app isn't signed/notarized -- that requires"
echo "a paid Apple Developer account). The operator can allow it once via"
echo "System Settings > Privacy & Security > 'Open Anyway', or by"
echo "right-clicking the app and choosing Open."

