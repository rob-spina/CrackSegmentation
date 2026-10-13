"""test_v109_reset_pipeline.py

Headless tests for the 1.0.9 [Reset pipeline training] button (code 30): every file of Training / Validation /
Test set goes back to 'Images', 'JSON files' and 'Binary files'; the stale cross-validation folds are removed.

Run with:  python3 -m unittest test_v109_reset_pipeline -v
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import pipeline_reset as pr
from config import Config
from smart_segmentation import CrackSegmentation
from test_v108_auto_split import _make


def _snapshot(root):
    """{relative path: size} of every file under root."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


class TestReset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v109_reset_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        _make(self.tmpdir, {"BLDG001": 5, "BLDG002": 4, "BLDG003": 3, "BLDG004": 2, "BLDG005": 2},
              negatives={"DJI_BLDG005_00__BLDG005.JPG"})
        for name in os.listdir(os.path.join(self.tmpdir, "Images")):   # give files some content
            with open(os.path.join(self.tmpdir, "Images", name), "w") as f:
                f.write(name)
        self.before = _snapshot(self.tmpdir)
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        self.app = CrackSegmentation(cfg)
        self.app.cfg.queue_idle = True

    def _auto_split(self, percent="70/15/15"):
        with mock.patch.object(self.app, "_ask_auto_split_percent", return_value=percent), \
                mock.patch.object(self.app, "_confirm_auto_split", return_value=True), \
                mock.patch.object(self.app, "_notify_auto_split"):
            self.assertTrue(self.app.run_auto_split())

    def _cv_folds(self):
        with mock.patch.object(self.app, "_ask_cv_fold_count", return_value=2), \
                mock.patch.object(self.app, "_confirm_cv_folds", return_value=True), \
                mock.patch.object(self.app, "_notify_cv_folds"):
            self.app.make_cv_folds()

    def _reset(self, confirm=True):
        with mock.patch.object(self.app, "_confirm_reset_pipeline", return_value=confirm) as asked, \
                mock.patch.object(self.app, "_notify_reset_pipeline") as notified:
            moved = self.app.run_reset_pipeline()
        return moved, asked, notified

    def test_auto_split_then_reset_gives_back_the_starting_folders(self):
        self._auto_split()
        self._cv_folds()
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "Training set", "folds")))
        moved, asked, _ = self._reset()
        self.assertTrue(moved)
        self.assertEqual(_snapshot(self.tmpdir), self.before, "every file back, same size, nothing left over")
        for name in ("Training set", "Validation set", "Test set"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, name)), f"{name} removed once empty")
        message = asked.call_args[0][0]
        self.assertIn("are removed: press CV folds again", message)
        self.assertIn("Training set:", message)

    def test_name_clash_is_never_overwritten(self):
        self._auto_split()
        test_photo = sorted(os.listdir(os.path.join(self.tmpdir, "Test set", "Images")))[0]
        clash = os.path.join(self.tmpdir, "Images", test_photo)
        with open(clash, "w") as f:
            f.write("a different photo with the same name")
        moved, asked, notified = self._reset()
        self.assertTrue(moved)
        with open(clash) as f:
            self.assertEqual(f.read(), "a different photo with the same name")
        self.assertIn(test_photo, os.listdir(os.path.join(self.tmpdir, "Test set", "Images")), "stays in its set")
        self.assertIn("already exist in the working folders", asked.call_args[0][0])
        self.assertIn(test_photo, notified.call_args[0][0])

    def test_cancel_moves_nothing(self):
        self._auto_split()
        after_split = _snapshot(self.tmpdir)
        moved, _, _ = self._reset(confirm=False)
        self.assertFalse(moved)
        self.assertEqual(_snapshot(self.tmpdir), after_split)

    def test_nothing_to_reset(self):
        moved, asked, notified = self._reset()
        self.assertFalse(moved)
        asked.assert_not_called()
        self.assertIn("already empty", notified.call_args[0][0])

    def test_from_mode2_the_queue_is_reloaded(self):
        self._auto_split()
        self.app.cfg.queue_idle = False
        self.app.cfg.CURRENT_IMAGE_PATH = ""
        with mock.patch.object(self.app, "_confirm_reset_pipeline", return_value=True), \
                mock.patch.object(self.app, "_notify_reset_pipeline"):
            self.assertTrue(self.app.process_keypress(pr.RESET_PIPELINE_CODE))
        self.assertTrue(self.app.cfg.reload_queue_requested)

    def test_from_the_idle_screen_the_queue_is_rebuilt(self):
        self._auto_split()
        self.app.cfg.modalita_scelta = "2"
        self.app._set_queue_folders()
        with mock.patch.object(self.app, "_confirm_reset_pipeline", return_value=True), \
                mock.patch.object(self.app, "_notify_reset_pipeline"):
            self.assertTrue(self.app._handle_idle_key(pr.RESET_PIPELINE_CODE))
        self.assertEqual(len(self.app.cfg.image_queue), 16, "every segmented photo is back in the Mode 2 queue")


if __name__ == "__main__":
    unittest.main()
