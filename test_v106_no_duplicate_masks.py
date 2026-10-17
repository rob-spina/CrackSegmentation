"""test_v106_no_duplicate_masks.py

USER REPORT (v1.0.6): saving again after the photo's building group changed left the
overlay/binary PNGs of the previous group next to the new ones (2-3 versions per photo).
Now a group change renames them, and every save removes copies left under another group.

Run with:  python3 -m unittest test_v106_no_duplicate_masks -v
"""
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from config import Config
from smart_segmentation import CrackSegmentation


def _photo_app(tmpdir, name="DJI_5__BLDG020.JPG"):
    cfg = Config()
    cfg.SCRIPT_DIR = tmpdir
    app = CrackSegmentation(cfg)
    images = os.path.join(tmpdir, "Images")
    os.makedirs(images, exist_ok=True)
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    path = os.path.join(images, name)
    cv2.imwrite(path, img)
    app.cfg.CURRENT_IMAGE_PATH = path
    app.cfg.img_original = img
    app.cfg.H_img, app.cfg.W_img = 120, 160
    app.cfg.JSON_OUTPUT_PATH = os.path.join(app.cfg.OUTPUT_FOLDER, os.path.splitext(name)[0] + ".json")
    app.cfg.saved_cracks = [{'start': (10, 60), 'end': (150, 60), 'path': [(x, 60) for x in range(10, 151)],
                             'active': True, 'session_id': 'current'}]
    app.cfg.saved_detachments = []
    app.cfg.blue_visual_mask = np.zeros((120, 160), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((120, 160), dtype=np.uint8)
    app.recalculate_masks()
    return app


def _binary_folder(app):
    return os.path.join(str(app.cfg.SCRIPT_DIR), app.cfg.folder_seg_img)


def _save_outputs(app):
    app._remove_stale_photo_outputs()
    app._export_segmented_preview_image()
    app._export_binary_masks()


class TestNoDuplicateMasks(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_dupmask_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _names(self, app):
        return sorted(f for f in os.listdir(_binary_folder(app)) if not f.startswith("."))

    def test_group_change_renames_the_existing_files(self):
        app = _photo_app(self.tmpdir)
        _save_outputs(app)
        self.assertIn("DJI_5__BLDG020-crack_mask.png", self._names(app))
        self.assertTrue(app._apply_building_tag_to_current_file(37))
        names = self._names(app)
        self.assertIn("DJI_5__BLDG037-crack_mask.png", names)
        self.assertIn("DJI_5__BLDG037-seg.JPG", names)
        self.assertFalse(any("BLDG020" in n for n in names), names)
        _save_outputs(app)
        self.assertEqual(self._names(app), ["DJI_5__BLDG037-crack_mask.png", "DJI_5__BLDG037-seg.JPG"])

    def test_save_removes_copies_left_under_other_groups(self):
        app = _photo_app(self.tmpdir, "DJI_5__BLDG037.JPG")
        folder = _binary_folder(app)
        os.makedirs(folder, exist_ok=True)
        leftovers = ["DJI_5__BLDG020-crack_mask.png", "DJI_5__BLDG020-seg.JPG", "DJI_5-seg.JPG",
                     "DJI_5__BLDG031-crack_uncertain_mask.png"]
        others = ["DJI_50__BLDG020-crack_mask.png", "DJI_5b__BLDG020-seg.JPG"]  # different photos: untouched
        for f in leftovers + others:
            open(os.path.join(folder, f), "wb").close()
        _save_outputs(app)
        names = self._names(app)
        for f in leftovers:
            self.assertNotIn(f, names)
        for f in others:
            self.assertIn(f, names)
        self.assertIn("DJI_5__BLDG037-crack_mask.png", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
