"""test_crack_quality.py

Headless tests for the per-crack quality scores saved in the LabelMe JSON:
reliability_pct, confidence, the uncertain flag/label and the uncertain-crack mask.

Run with:  python3 -m unittest test_crack_quality -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from test_gui_shell_parity import _MockedHighGui, _cleanup_real_module_dir_artifacts, make_app


def _wall_with_cracks(h=300, w=400, seed=0):
    """Textured plaster with a strong crack at y=80, a faint one at y=160, and plain wall at y=240."""
    rng = np.random.default_rng(seed)
    wall = rng.normal(170, 12, (h, w)).astype(np.float32)
    wall = cv2.GaussianBlur(wall, (3, 3), 0)
    cv2.line(wall, (30, 80), (370, 85), 60, 2)     # clear crack
    cv2.line(wall, (30, 160), (370, 165), 150, 1)  # faint crack
    gray = np.clip(wall, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _app_on(img):
    app = make_app()
    app.cfg.img_original = img
    app.cfg.H_img, app.cfg.W_img = img.shape[:2]
    app._compute_binary_and_skeleton_masks()
    return app


STRONG = [(30, 80), (200, 82), (370, 85)]
FAINT = [(30, 160), (200, 162), (370, 165)]
PLAIN = [(30, 240), (200, 242), (370, 245)]


class TestCrackConfidence(unittest.TestCase):

    def setUp(self):
        self.app = _app_on(_wall_with_cracks())

    def test_confidence_separates_cracks_from_plain_wall(self):
        strong = self.app._compute_crack_confidence(STRONG)
        faint = self.app._compute_crack_confidence(FAINT)
        plain = self.app._compute_crack_confidence(PLAIN)
        for value in (strong, faint, plain):
            self.assertIsNotNone(value)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreater(strong, 0.95)
        self.assertGreater(faint, 0.8)
        self.assertLess(plain, 0.65, "plain wall must rank near 0.5")

    def test_tolerates_a_line_traced_a_couple_of_px_off(self):
        on = self.app._compute_crack_confidence(STRONG)
        off = self.app._compute_crack_confidence([(x, y + 2) for x, y in STRONG])
        self.assertGreater(off, 0.9)
        self.assertAlmostEqual(on, off, delta=0.1)

    def test_degenerate_paths_and_cache(self):
        self.assertIsNone(self.app._compute_crack_confidence([]))
        self.assertIsNone(self.app._compute_crack_confidence([(10, 10)]))
        first = self.app._compute_crack_confidence(STRONG)
        with mock.patch("smart_segmentation.sato", side_effect=AssertionError("must hit the cache")):
            self.assertEqual(self.app._compute_crack_confidence(STRONG), first)

    def test_densify_path_is_continuous(self):
        dense = self.app._densify_path([(0, 0), (10, 4), (10, 4), (3, 9)])
        self.assertEqual(dense[0], (0, 0))
        self.assertEqual(dense[-1], (3, 9))
        for (xa, ya), (xb, yb) in zip(dense[:-1], dense[1:]):
            self.assertLessEqual(max(abs(xa - xb), abs(ya - yb)), 1)


class TestUncertainRule(unittest.TestCase):

    def test_thresholds(self):
        app = make_app()
        app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = 50.0
        app.cfg.CRACK_UNCERTAIN_CONFIDENCE = None
        self.assertTrue(app._is_crack_uncertain(49.0, 0.99))
        self.assertFalse(app._is_crack_uncertain(50.0, 0.10))
        self.assertFalse(app._is_crack_uncertain(None, None))
        app.cfg.CRACK_UNCERTAIN_CONFIDENCE = 0.8
        self.assertTrue(app._is_crack_uncertain(90.0, 0.7))
        app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = None
        self.assertFalse(app._is_crack_uncertain(0.0, 0.9))


class TestExport(unittest.TestCase):

    def test_json_fields_label_and_uncertain_mask(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_quality_")
        try:
            img = _wall_with_cracks()
            img_path = os.path.join(tmpdir, "wall.png")
            cv2.imwrite(img_path, img)
            app = _app_on(img)
            app.cfg.SCRIPT_DIR = tmpdir
            app.cfg.CURRENT_IMAGE_PATH = img_path
            app.cfg.JSON_OUTPUT_PATH = os.path.join(tmpdir, "wall.json")
            app.cfg.modalita_scelta = "2"
            app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = None
            app.cfg.CRACK_UNCERTAIN_CONFIDENCE = 0.8
            app.cfg.saved_cracks = [
                {'start': STRONG[0], 'end': STRONG[-1], 'path': STRONG, 'active': True},
                {'start': PLAIN[0], 'end': PLAIN[-1], 'path': PLAIN, 'active': True},
            ]
            with mock.patch("smart_segmentation.resolve_script_dir", return_value=tmpdir), \
                 mock.patch.object(app, "_play_save_beep"), _MockedHighGui():
                app.export_labelme_format()

            with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
                shapes = json.load(fh)["shapes"]
            strong, plain = shapes
            self.assertEqual(strong["label"], "crack_wall.png")
            self.assertEqual(strong["flags"], {"uncertain": False})
            self.assertGreater(strong["confidence"], 0.9)
            self.assertIsInstance(strong["reliability_pct"], float)
            self.assertEqual(plain["label"], "crack_incerta_wall.png")
            self.assertEqual(plain["flags"], {"uncertain": True})
            self.assertLess(plain["confidence"], 0.65)

            seg = os.path.join(tmpdir, "segmentated images")
            full = cv2.imread(os.path.join(seg, "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
            unc = cv2.imread(os.path.join(seg, "wall-crack_uncertain_mask.png"), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(full[82, 200] and full[242, 200], "the crack mask still holds every crack")
            self.assertTrue(unc[242, 200])
            self.assertFalse(unc[82, 200])

            # Reloading the saved JSON still recognises both shapes as cracks.
            app.cfg.saved_cracks = []
            app.load_labelme_format()
            self.assertEqual(len(app.cfg.saved_cracks), 2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()


if __name__ == "__main__":
    unittest.main()
