"""
Headless tests for 2.0.1 detachments in training: a [D] detachment is damage but not a crack, so a crack model
never learns it as background.
  1. [Save] writes <photo>-detachment_mask.png (filled polygon) and the JSON reloads it as a detachment;
  2. [CV folds] lists it in folds.csv (detachment_mask column, '' when the photo has none);
  3. active_learning_prep.py puts it in the 128 ignore band, also with --uncertain background;
  4. the active-learning score leaves its pixels out (active_learning.uncertain_mask).
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import cv2
import numpy as np

import active_learning as al
import active_learning_prep as prep
import cross_validation as cv
from test_gui_shell_parity import _MockedHighGui, _cleanup_real_module_dir_artifacts
from test_v105_link import _two_crack_app, H, W

SQUARE = [(60, 60), (120, 60), (120, 120), (60, 120)]


def _write(path, mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask)


class TestDetachmentExport(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_detach_")
        self.app = _two_crack_app(self.tmpdir)
        self.app.cfg.modalita_scelta = "2"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _cleanup_real_module_dir_artifacts()

    def _export(self):
        with mock.patch("smart_segmentation.resolve_script_dir", return_value=self.tmpdir), \
             mock.patch.object(self.app, "_play_save_beep"), _MockedHighGui():
            self.app.export_labelme_format()

    def test_filled_mask_and_reload(self):
        self.app.cfg.saved_detachments.append({"path": list(SQUARE), "click_nodes": list(SQUARE),
                                               "active": True, "session_id": "t"})
        self._export()
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-detachment_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (H, W))
        self.assertEqual(mask[90, 90], 255, "the inside of the polygon is filled, not just its outline")
        self.assertEqual(mask[150, 150], 0)
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            labels = [s["label"] for s in json.load(fh)["shapes"]]
        self.assertIn("detachment_wall.png", labels)
        self.app.cfg.saved_detachments = []
        self.app.load_labelme_format()
        self.assertEqual(len(self.app.cfg.saved_detachments), 1)


class TestFoldsColumn(unittest.TestCase):

    def test_detachment_mask_in_folds_csv(self):
        root = tempfile.mkdtemp(prefix="crackseg_detach_folds_")
        self.addCleanup(shutil.rmtree, root, True)
        for base in ("a__BLDG001", "b__BLDG002"):
            _write(os.path.join(root, "Images", base + ".jpg"), np.zeros((4, 4, 3), np.uint8))
            _write(os.path.join(root, "Binary files", base + "-crack_mask.png"), np.zeros((4, 4), np.uint8))
        _write(os.path.join(root, "Binary files", "a__BLDG001-detachment_mask.png"), np.zeros((4, 4), np.uint8))
        rows = {r["photo"]: r for r in cv.fold_rows(root, cv.assign_group_folds(["a__BLDG001.jpg", "b__BLDG002.jpg"], 2))}
        self.assertIn("detachment_mask", cv.FOLDS_FIELDS)
        self.assertEqual(rows["a__BLDG001.jpg"]["detachment_mask"], "Binary files/a__BLDG001-detachment_mask.png")
        self.assertEqual(rows["b__BLDG002.jpg"]["detachment_mask"], "")


class TestActiveLearningIgnoresDetachments(unittest.TestCase):

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="crackseg_detach_al_")
        self.out = tempfile.mkdtemp(prefix="crackseg_detach_alout_")
        self.addCleanup(shutil.rmtree, self.data, True)
        self.addCleanup(shutil.rmtree, self.out, True)
        bins = os.path.join(self.data, "Training set", "Binary files")
        _write(os.path.join(self.data, "Training set", "Images", "p.jpg"), np.zeros((10, 10, 3), np.uint8))
        crack = np.zeros((10, 10), np.uint8)
        crack[2] = 255
        det = np.zeros((10, 10), np.uint8)
        det[6:9, 0:5] = 255                              # bricks fallen out: damage, not a crack
        _write(os.path.join(bins, "p-crack_mask.png"), crack)
        _write(os.path.join(bins, "p-detachment_mask.png"), det)
        self.photo = os.path.join(self.data, "Training set", "Images", "p.jpg")

    def _exported(self, *extra):
        with redirect_stdout(io.StringIO()):
            prep.main(["--data-dir", self.data, "--out", self.out, *extra])
        return cv2.imread(os.path.join(self.out, "masks", "p.png"), cv2.IMREAD_GRAYSCALE)

    def test_prep_puts_detachments_in_the_ignore_band(self):
        for extra in ((), ("--uncertain", "background")):
            m = self._exported(*extra)
            self.assertTrue((m[2] == 255).all(), extra)
            self.assertTrue((m[6:9, 0:5] == 128).all(), f"detachment not ignored {extra}")
            self.assertTrue((m[6:9, 5:] == 0).all(), extra)

    def test_score_leaves_detachments_out(self):
        ign = al.uncertain_mask(self.data, self.photo)
        self.assertIsNotNone(ign)
        self.assertTrue((ign[6:9, 0:5] > 127).all())
        self.assertFalse((ign[2] > 127).any())


if __name__ == "__main__":
    unittest.main()
