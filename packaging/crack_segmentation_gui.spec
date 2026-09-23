# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Crack & Detachment Segmentation tool (GUI shell).

IMPORTANT: PyInstaller does NOT cross-compile. Run this spec ONCE on a real
Mac to produce the .app, ONCE on a real Windows PC to produce the .exe, and
ONCE on a real Linux (Ubuntu/Debian) machine to produce the Linux binary
-- see build_mac.sh / build_windows.bat / build_linux.sh in this same
folder for the exact commands. There is no way to produce one platform's
build from another with PyInstaller.

Usage (from this "packaging" folder, after `pip install -r
../requirements.txt`):

    pyinstaller crack_segmentation_gui.spec

Output lands in packaging/dist/CrackSegmentation/ (onedir build -- see the
notes in build_mac.sh/build_windows.bat for why onedir was chosen over
onefile for this app).

CONSOLE WINDOW: back ON by default (console=True below) as a precaution
after a real-Mac crash report ("CrackSegmentation quit unexpectedly")
right after console=False was introduced. Not yet confirmed as the
actual cause, but it's the newest, most speculative change in that
delivery (a "nice to have" cleanup, not something requested) and a
windowed (no-console) macOS build can behave differently around
sys.stdout/sys.stderr -- reverting it removes one variable while the
real cause is tracked down, and having a console back also means any
Python-level traceback is visible directly instead of only in a system
crash log. The app's own window still has a status strip mirroring every
print()'d message either way (see _StatusBarStream in
crack_segmentation_gui.py) -- nothing about that is affected by this
setting. Once the crash is understood, this can safely go back to
console=False if that turns out to be unrelated.
"""
import sys
from pathlib import Path

block_cipher = None

# This spec lives in <project>/packaging/ -- the actual source files
# (crack_segmentation_gui.py, smart_segmentation.py, config.py,
# compare_and_filter_cracks.py) live one directory up.
SRC_DIR = str(Path(SPECPATH).resolve().parent)

a = Analysis(
    [str(Path(SRC_DIR) / "crack_segmentation_gui.py")],
    pathex=[SRC_DIR],
    binaries=[],
    # Bundles the PDF user manual and the ECHO-TWIN logo (startup dialog)
    # into the root of the packaged app, next to the executable --
    # _resolve_app_resource_dir() (crack_segmentation_gui.py) resolves to
    # that same folder in a frozen build, so both are found the exact same
    # way they are when run from source.
    datas=[
        (str(Path(SRC_DIR) / "Crack_Segmentation_User_Manual.pdf"), "."),
        (str(Path(SRC_DIR) / "echo_twin_logo.png"), "."),
    ],
    hiddenimports=[
        # scikit-image lazily imports several of its own compiled
        # extensions; PyInstaller's static analysis sometimes misses
        # these. If the built app crashes on startup with a
        # ModuleNotFoundError for anything under skimage.*, add the
        # missing dotted name to this list and rebuild.
        "skimage.morphology",
        "skimage.morphology._skeletonize",
        "skimage.filters",
        "skimage.filters.rank",
        "skimage.filters.rank.core_cy",
        "skimage.draw",
        "skimage.measure",
        "skimage._shared.geometry",
        # Tkinter's dialog/messagebox submodules are used by
        # crack_segmentation_gui.py but not always auto-detected.
        "tkinter.simpledialog",
        "tkinter.messagebox",
        "tkinter.ttk",
        # Pillow's Tk-image bridge, used to blit each finished frame into
        # the embedded canvas -- has its own PyInstaller quirks on some
        # platforms, so listed explicitly rather than relying on autodetect.
        "PIL.ImageTk",
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CrackSegmentation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # see the CONSOLE WINDOW note above
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CrackSegmentation",
)

# macOS only: also wrap the onedir build into a proper double-clickable
# CrackSegmentation.app bundle. On Windows/Linux this block is skipped;
# packaging/dist/CrackSegmentation/CrackSegmentation.exe (or the
# extension-less binary on Linux) is already the right thing to run.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CrackSegmentation.app",
        icon=None,  # drop an .icns file next to this spec and point this at it for a custom icon
        bundle_identifier="local.crackseg.gui",
        info_plist={
            "NSHighResolutionCapable": "True",
            "CFBundleShortVersionString": "1.0.0",
            "NSHumanReadableCopyright": "Internal tool",
        },
    )
