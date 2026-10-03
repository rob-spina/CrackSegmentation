"""test_crack_multiview.py

Headless tests for the multi-view crack check: each crack is looked for in the
other photos of its building group, where a real crack stays in place while a
shadow seen in one photo only does not.

Run with:  python3 -m unittest test_crack_multiview -v
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

CRACK = [(200, 250), (380, 300), (520, 280), (700, 360), (900, 340)]
SHADOW = [(200, 650), (500, 640), (900, 660)]
H_A = np.array([[0.95, 0.04, 30.0], [-0.03, 0.97, 25.0], [2e-5, 1e-5, 1.0]])
H_B = np.array([[1.08, -0.02, -40.0], [0.03, 1.05, -10.0], [-1e-5, 2e-5, 1.0]])


def _textured_wall(h=900, w=1200, seed=1):
    """Plaster with stains and pebbles, enough detail for SIFT to align the views."""
    rng = np.random.default_rng(seed)
    img = rng.normal(165, 10, (h, w)).astype(np.float32)
    for _ in range(400):
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(3, 18)
        cv2.circle(img, (int(x), int(y)), int(r), float(rng.normal(150, 30)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _other_view(wall_with_crack, H, gain):
    """The same wall seen from another viewpoint (homography H) and under different light (gain)."""
    view = cv2.warpPerspective(wall_with_crack, H, (1300, 1000), borderMode=cv2.BORDER_REFLECT)
    return np.clip(view * gain, 0, 255).astype(np.uint8)


class _MultiViewScene:
    """Current photo with a real crack and a shadow line; two other photos of the group show only the crack."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="crackseg_multiview_")
        base = _textured_wall()
        with_crack = base.copy()
        cv2.polylines(with_crack, [np.int32(CRACK)], False, 70, 2)
        current = with_crack.copy()
        cv2.polylines(current, [np.int32(SHADOW)], False, 90, 3)
        self.current = np.clip(current, 0, 255).astype(np.uint8)
        cv2.imwrite(self.path("a__BLDG002.png"), _other_view(with_crack, H_A, 0.9))
        cv2.imwrite(self.path("b__BLDG002.png"), _other_view(with_crack, H_B, 1.1))
        cv2.imwrite(self.path("c__BLDG007.png"), _other_view(with_crack, H_B, 1.0))
        cv2.imwrite(self.path("cur__BLDG002.png"), self.current)

    def path(self, name):
        return os.path.join(self.dir, name)

    def make_app(self, current_name="cur__BLDG002.png"):
        app = make_app()
        app.cfg.img_original = cv2.cvtColor(self.current, cv2.COLOR_GRAY2BGR)
        app.cfg.H_img, app.cfg.W_img = self.current.shape[:2]
        app._compute_binary_and_skeleton_masks()
        app.cfg.SCRIPT_DIR = self.dir
        app.cfg.IMAGE_FOLDER = self.dir
        app.cfg.OUTPUT_FOLDER = os.path.join(self.dir, "segmentated images")
        app.cfg.CURRENT_IMAGE_PATH = self.path(current_name)
        app.cfg.saved_cracks = [
            {'start': CRACK[0], 'end': CRACK[-1], 'path': CRACK, 'active': True},
            {'start': SHADOW[0], 'end': SHADOW[-1], 'path': SHADOW, 'active': True},
        ]
        return app

    def script_dir_patch(self):
        return mock.patch("smart_segmentation.resolve_script_dir", return_value=self.dir)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestMultiViewCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scene = _MultiViewScene()

    @classmethod
    def tearDownClass(cls):
        cls.scene.cleanup()

    def _run(self, app):
        with self.scene.script_dir_patch():
            app.compute_multiview_scores()

    def test_group_photos_and_group_number(self):
        app = self.scene.make_app()
        open(self.scene.path("a__BLDG002-crack_mask.png"), "wb").close()
        open(self.scene.path("a__BLDG002-seg.jpg"), "wb").close()
        try:
            with self.scene.script_dir_patch():
                views = app._list_group_photo_paths(2)
            self.assertEqual([os.path.basename(v) for v in views], ["a__BLDG002.png", "b__BLDG002.png"])
            self.assertEqual(app._current_building_group(), 2)
            app.cfg.CURRENT_IMAGE_PATH = self.scene.path("untagged.png")
            self.assertIsNone(app._current_building_group())
        finally:
            os.remove(self.scene.path("a__BLDG002-crack_mask.png"))
            os.remove(self.scene.path("a__BLDG002-seg.jpg"))

    def test_real_crack_confirmed_shadow_not(self):
        app = self.scene.make_app()
        self._run(app)
        crack, shadow = app._crack_multiview(CRACK), app._crack_multiview(SHADOW)
        self.assertEqual((crack['views_checked'], crack['views_confirmed']), (2, 2))
        self.assertGreater(crack['confidence'], 0.9)
        self.assertEqual((shadow['views_checked'], shadow['views_confirmed']), (2, 0))
        self.assertLess(shadow['confidence'], 0.7)
        # The single-photo scores alone cannot tell the shadow apart.
        self.assertGreater(app._compute_crack_confidence(SHADOW), 0.9)
        self.assertEqual(app._build_crack_reliability_entries()[1].split()[-1], "v0/2")

    def test_residual_homography_error_is_corrected(self):
        app = self.scene.make_app()
        off = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]])
        with mock.patch.object(app, "_multiview_homography", side_effect=[off @ H_A, off @ H_B]):
            self._run(app)
        self.assertEqual(app._crack_multiview(CRACK)['views_confirmed'], 2)
        self.assertEqual(app._crack_multiview(SHADOW)['views_confirmed'], 0)

    def test_out_of_frame_and_failed_alignment_are_not_counted(self):
        app = self.scene.make_app()
        far = np.array([[1.0, 0.0, 5000.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with mock.patch.object(app, "_multiview_homography", side_effect=[far, None]):
            self._run(app)
        self.assertEqual(app._crack_multiview(CRACK), {'views_checked': 0, 'views_confirmed': 0, 'confidence': None})

    def test_results_are_cached_until_the_crack_changes(self):
        app = self.scene.make_app()
        self._run(app)
        with mock.patch("smart_segmentation.cv2.imread", side_effect=AssertionError("must hit the cache")):
            self._run(app)
        self.assertEqual(app._crack_multiview(CRACK)['views_confirmed'], 2)
        self.assertIsNone(app._crack_multiview([(x, y + 1) for x, y in CRACK]), "an edited crack has no stale result")

    def test_no_group_or_disabled_means_no_result(self):
        app = self.scene.make_app(current_name="untagged.png")
        self._run(app)
        self.assertIsNone(app._crack_multiview(CRACK))
        app = self.scene.make_app()
        app.cfg.CRACK_MULTIVIEW_ENABLED = False
        self._run(app)
        self.assertIsNone(app._crack_multiview(CRACK))


class TestMultiViewRule(unittest.TestCase):

    def test_ratio_threshold(self):
        app = make_app()
        app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = None
        app.cfg.CRACK_UNCERTAIN_MULTIVIEW_RATIO = None
        self.assertFalse(app._is_crack_uncertain(90.0, 0.9, {'views_checked': 3, 'views_confirmed': 0, 'confidence': 0.5}))
        app.cfg.CRACK_UNCERTAIN_MULTIVIEW_RATIO = 0.5
        self.assertTrue(app._is_crack_uncertain(90.0, 0.9, {'views_checked': 3, 'views_confirmed': 1, 'confidence': 0.6}))
        self.assertFalse(app._is_crack_uncertain(90.0, 0.9, {'views_checked': 4, 'views_confirmed': 2, 'confidence': 0.8}))
        self.assertFalse(app._is_crack_uncertain(90.0, 0.9, {'views_checked': 0, 'views_confirmed': 0, 'confidence': None}))
        self.assertFalse(app._is_crack_uncertain(90.0, 0.9, None))


class TestMultiViewExport(unittest.TestCase):

    def setUp(self):
        self.scene = _MultiViewScene()

    def tearDown(self):
        self.scene.cleanup()
        _cleanup_real_module_dir_artifacts()

    def _export(self, app):
        app.cfg.JSON_OUTPUT_PATH = os.path.join(self.scene.dir, "cur__BLDG002.json")
        app.cfg.modalita_scelta = "2"
        with self.scene.script_dir_patch(), mock.patch.object(app, "_play_save_beep"), _MockedHighGui():
            app.export_labelme_format()
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            return json.load(fh)["shapes"]

    def test_json_carries_multiview_and_marks_unconfirmed_cracks(self):
        app = self.scene.make_app()
        app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = None
        app.cfg.CRACK_UNCERTAIN_MULTIVIEW_RATIO = 0.5
        crack, shadow = self._export(app)
        self.assertEqual(crack["multiview"]["views_confirmed"], 2)
        self.assertEqual(crack["flags"], {"uncertain": False})
        self.assertEqual(shadow["multiview"]["views_confirmed"], 0)
        self.assertEqual(shadow["flags"], {"uncertain": True})
        self.assertTrue(shadow["label"].startswith("crack_incerta_"))

    def test_a_failing_check_never_blocks_the_save(self):
        app = self.scene.make_app()
        with mock.patch.object(app, "compute_multiview_scores", side_effect=RuntimeError("boom")):
            shapes = self._export(app)
        self.assertEqual(len(shapes), 2)
        self.assertIsNone(shapes[0]["multiview"])


if __name__ == "__main__":
    unittest.main()


class TestGroupFilterUsesPhotoGroup(unittest.TestCase):
    """[G] and the post-save check must compare the current photo's own group (its __BLDG tag),
    not CURRENT_BUILDING_INDEX, which tracks the highest group number seen so far."""

    def _tags_checked(self, app):
        seen = []
        with mock.patch.object(app, "_collect_group_json_paths", side_effect=lambda tag, folder: seen.append(tag) or []), \
             mock.patch("smart_segmentation.resolve_script_dir", return_value=tempfile.gettempdir()):
            app.filter_incompatible_cracks_for_current_group()
        return seen

    def test_photo_in_an_older_group(self):
        app = make_app()
        app.cfg.CURRENT_IMAGE_PATH = "/photos/IMG_0042__BLDG002.jpg"
        app.cfg.CURRENT_BUILDING_INDEX = 5
        self.assertEqual(self._tags_checked(app), ["__BLDG002"])

    def test_tagged_photo_without_index_and_untagged_photo(self):
        app = make_app()
        app.cfg.CURRENT_IMAGE_PATH = "/photos/IMG_0042__BLDG003.jpg"
        app.cfg.CURRENT_BUILDING_INDEX = None
        self.assertEqual(self._tags_checked(app), ["__BLDG003"])
        app.cfg.CURRENT_IMAGE_PATH = "/photos/IMG_0043.jpg"
        app.cfg.CURRENT_BUILDING_INDEX = 5
        self.assertEqual(self._tags_checked(app), [])


class TestSaveBeepDoesNotLeakPipes(unittest.TestCase):

    def test_macos_beep_discards_output(self):
        app = make_app()
        with mock.patch("smart_segmentation.sys.platform", "darwin"), mock.patch("subprocess.Popen") as popen:
            app._play_save_beep()
        import subprocess
        _, kwargs = popen.call_args
        self.assertEqual((kwargs["stdout"], kwargs["stderr"]), (subprocess.DEVNULL, subprocess.DEVNULL))
