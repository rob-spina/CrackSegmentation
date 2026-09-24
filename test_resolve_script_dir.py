"""test_resolve_script_dir.py

Verifies resolve_script_dir() (see config.py), added so a PyInstaller-
packaged .app/.exe stores its Images / segmentated images / already
processed images / CSV report in a normal, writable, discoverable
per-user folder instead of inside the app bundle itself (where __file__
resolves when frozen -- often read-only, and never somewhere a user would
think to look).

Covers:
  - Non-frozen (source) mode: identical to the original bare
    Path(__file__).resolve().parent -- byte-for-byte the same folder
    every previous delivery used, proving `python3 smart_segmentation.py`
    is unaffected.
  - Frozen mode: resolves to ~/Documents/CrackSegmentation, creating it
    if needed.
  - Config.__init__ and smart_segmentation.py's own module-level
    CARTELLA_BASE/processed_folder sites all route through the SAME
    helper, so source and frozen mode behave consistently everywhere,
    not just in Config.

Run with:  python3 -m unittest test_resolve_script_dir -v
"""
import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest import mock

import config
from config import Config, resolve_script_dir


class TestResolveScriptDirSourceMode(unittest.TestCase):

    def test_matches_original_bare_file_computation_when_not_frozen(self):
        with mock.patch.object(sys, "frozen", False, create=True):
            result = resolve_script_dir()
        expected = Path(config.__file__).resolve().parent
        self.assertEqual(result, expected)

    def test_config_uses_it_for_script_dir(self):
        with mock.patch.object(sys, "frozen", False, create=True):
            cfg = Config()
        self.assertEqual(cfg.SCRIPT_DIR, Path(config.__file__).resolve().parent)

    def test_frozen_attribute_absent_behaves_like_not_frozen(self):
        # getattr(sys, 'frozen', False) -- on a normal (non-PyInstaller)
        # interpreter sys.frozen doesn't exist at all; that must behave
        # exactly like frozen=False, not raise or misbehave.
        if hasattr(sys, "frozen"):
            with mock.patch.object(sys, "frozen", False, create=True):
                result = resolve_script_dir()
        else:
            result = resolve_script_dir()
        self.assertEqual(result, Path(config.__file__).resolve().parent)


class TestResolveScriptDirFrozenMode(unittest.TestCase):

    def setUp(self):
        self.tmp_home = tempfile.mkdtemp(prefix="fakehome_")

    def tearDown(self):
        shutil.rmtree(self.tmp_home, ignore_errors=True)

    def test_frozen_resolves_to_documents_crackseg_and_creates_it(self):
        expected = Path(self.tmp_home) / "Documents" / "CrackSegmentation"
        self.assertFalse(expected.exists())
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(Path, "home", return_value=Path(self.tmp_home)):
            result = resolve_script_dir()
        self.assertEqual(result, expected)
        self.assertTrue(expected.is_dir(), "resolve_script_dir() should create the folder if missing")

    def test_frozen_is_idempotent_on_existing_folder(self):
        expected = Path(self.tmp_home) / "Documents" / "CrackSegmentation"
        expected.mkdir(parents=True)
        marker = expected / "Images"
        marker.mkdir()
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(Path, "home", return_value=Path(self.tmp_home)):
            result = resolve_script_dir()
        self.assertEqual(result, expected)
        self.assertTrue(marker.is_dir(), "an existing user Images subfolder must survive re-resolution")

    def test_config_uses_frozen_path_for_script_dir_and_image_folder(self):
        expected = Path(self.tmp_home) / "Documents" / "CrackSegmentation"
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(Path, "home", return_value=Path(self.tmp_home)):
            cfg = Config()
        self.assertEqual(cfg.SCRIPT_DIR, expected)
        self.assertEqual(cfg.IMAGE_FOLDER, expected)
        self.assertEqual(cfg.OUTPUT_FOLDER, os.path.join(str(expected), "segmentated images"))


class TestSmartSegmentationUsesTheSameHelper(unittest.TestCase):
    """smart_segmentation.py has several methods that independently used
    to compute os.path.dirname(os.path.abspath(__file__)) for the
    "already processed images" folder and the incremental CSV report.
    All of them were mechanically switched to call resolve_script_dir()
    too, so frozen mode is consistent everywhere, not just in Config."""

    def test_no_leftover_raw_file_dirname_computation(self):
        import smart_segmentation
        src = Path(smart_segmentation.__file__).read_text(encoding="utf-8")
        self.assertNotIn("os.path.dirname(os.path.abspath(__file__))", src,
                          "found a leftover call not routed through resolve_script_dir()")

    def test_smart_segmentation_imports_the_shared_helper(self):
        import smart_segmentation
        self.assertIs(smart_segmentation.resolve_script_dir, resolve_script_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
