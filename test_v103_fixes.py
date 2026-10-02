"""test_v103_fixes.py

Headless tests for the fixes reported on v1.0.2:
  1. Status banners (TIMEOUT, WARP SYNC, ...) auto-hide after 15 s and [.] hides them all.
  2. Each crack's start marker carries its number.
  3. Warp Sync: SIFT restricted to the source cracks' region, MAGSAC fit and a
     quality gate that rejects unreliable alignments instead of importing them.
  4. The HUD's "FILE: n/N" counter keeps counting up in Mode 1.

Run with:  python3 -m unittest test_v103_fixes -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from config import Config
from smart_segmentation import CrackSegmentation
from test_gui_shell_parity import (_MockedHighGui, _cleanup_real_module_dir_artifacts,
                                   make_app, make_synthetic_image_folder)


class TestBannerAutoHideAndDismiss(unittest.TestCase):

    def test_warp_banner_hides_itself_after_the_timeout(self):
        app = make_app()
        app.cfg.warp_jitter_triggered = True
        with mock.patch("smart_segmentation.time.time", return_value=1000.0):
            self.assertTrue(app._banner_still_visible('warp_jitter_triggered'))
        with mock.patch("smart_segmentation.time.time", return_value=1000.0 + app.cfg.BANNER_AUTOHIDE_SECONDS - 1):
            self.assertTrue(app._banner_still_visible('warp_jitter_triggered'))
        with mock.patch("smart_segmentation.time.time", return_value=1000.0 + app.cfg.BANNER_AUTOHIDE_SECONDS + 1):
            self.assertFalse(app._banner_still_visible('warp_jitter_triggered'))
        self.assertFalse(app.cfg.warp_jitter_triggered, "an expired banner must clear its own flag")

    def test_dot_key_hides_every_banner(self):
        app = make_app()
        app.cfg.pathfinding_timeout_triggered = True
        app.cfg.warp_jitter_triggered = True
        app.cfg.import_warning_triggered = True
        app.cfg.save_timestamp = 12345.0
        app.handle_keyboard(ord('.'))
        self.assertFalse(app.cfg.pathfinding_timeout_triggered)
        self.assertFalse(app.cfg.warp_jitter_triggered)
        self.assertFalse(app.cfg.import_warning_triggered)
        self.assertEqual(app.cfg.save_timestamp, 0.0)

    def test_banner_drawing_uses_the_custom_warp_message(self):
        app = make_app()
        app.cfg.warp_jitter_triggered = True
        app.cfg.warp_banner_message = "WARP SYNC: aligned (40/42 matches)"
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        with mock.patch("cv2.putText") as put_text:
            app._draw_timeout_and_warp_banners(win_out, app._compute_hud_font_metrics())
        self.assertIn("WARP SYNC: aligned (40/42 matches)", [c.args[1] for c in put_text.call_args_list])


class TestCrackNumberInStartMarker(unittest.TestCase):

    def _app_with_two_cracks(self):
        app = make_app()
        app.cfg.W_img, app.cfg.H_img = 1200, 900
        app.cfg.zoom_box = [0, 0, 1200, 900]
        app.cfg.saved_cracks = [
            {'start': (100, 100), 'end': (300, 100), 'path': [(100, 100), (300, 100)], 'active': True},
            {'start': (100, 400), 'end': (300, 400), 'path': [(100, 400), (300, 400)], 'active': False},
        ]
        return app

    def test_start_markers_show_the_one_based_crack_number(self):
        app = self._app_with_two_cracks()
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        with mock.patch("cv2.putText", wraps=cv2.putText) as put_text:
            app._draw_crack_markers(win_out, 22)
        drawn = [(c.args[1], c.args[2]) for c in put_text.call_args_list]
        self.assertEqual([t for t, _ in drawn], ["1", "2"])
        for text, (x, y) in drawn:
            self.assertTrue(80 <= x <= 120, f"number {text} must sit inside its start marker")

    def test_active_start_marker_is_yellow_filled_and_end_marker_unchanged(self):
        app = self._app_with_two_cracks()
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_crack_markers(win_out, 22)
        # Corner of the active start marker's inner area: yellow fill (no digit there).
        self.assertEqual(tuple(win_out[100 - 17, 100 - 17]), (0, 255, 255))
        # End marker keeps its small filled inner square.
        self.assertEqual(tuple(win_out[100, 300]), (0, 255, 255))
        # Soft-deleted crack: dark fill inside the start marker.
        self.assertEqual(tuple(win_out[400 - 17, 100 - 17]), (40, 40, 40))

    def test_three_digit_numbers_still_fit_inside_the_marker(self):
        app = make_app()
        win_out = np.full((200, 200, 3), 255, dtype=np.uint8)  # white, so black digit pixels stand out
        app._draw_crack_number_in_marker(win_out, 100, 100, 22, 123, True)
        ys, xs = np.where(np.all(win_out < 60, axis=2))
        self.assertTrue(xs.size > 0, "digits must be drawn")
        self.assertGreaterEqual(xs.min(), 100 - 22)
        self.assertLessEqual(xs.max(), 100 + 22)


def _textured_image(seed, size=(600, 800)):
    rng = np.random.default_rng(seed)
    img = (rng.random((size[0] // 8, size[1] // 8)) * 255).astype(np.uint8)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestWarpSyncAlignment(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_warp_")
        self.H_true = np.array([[1.05, 0.03, 25.0], [-0.02, 0.98, 15.0], [1e-5, 0.0, 1.0]])
        src = _textured_image(1)
        self.src_path = os.path.join(self.tmpdir, "src.png")
        self.dst_path = os.path.join(self.tmpdir, "dst.png")
        self.other_path = os.path.join(self.tmpdir, "other.png")
        cv2.imwrite(self.src_path, src)
        cv2.imwrite(self.dst_path, cv2.warpPerspective(src, self.H_true, (800, 600)))
        cv2.imwrite(self.other_path, _textured_image(2))
        self.crack = [[200.0, 200.0], [260.0, 260.0], [320.0, 300.0], [400.0, 380.0]]
        self.src_json = os.path.join(self.tmpdir, "src.json")
        with open(self.src_json, "w", encoding="utf-8") as f:
            json.dump({"shapes": [{"label": "crack", "points": self.crack, "shape_type": "linestrip", "flags": {}}],
                       "imagePath": "src.png", "imageHeight": 600, "imageWidth": 800}, f)
        self.out_json = os.path.join(self.tmpdir, "out.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_same_scene_projects_cracks_onto_the_right_place(self):
        app = make_app()
        ok = app.warp_and_adapt_json_to_new_image(self.src_path, self.dst_path, self.src_json, self.out_json)
        self.assertTrue(ok, app.cfg.import_warning_message)
        with open(self.out_json, encoding="utf-8") as f:
            projected = np.array(json.load(f)["shapes"][0]["points"], dtype=np.float32)
        expected = cv2.perspectiveTransform(np.array([self.crack], dtype=np.float32), self.H_true)[0]
        self.assertLess(float(np.max(np.linalg.norm(projected - expected, axis=1))), 3.0)
        self.assertTrue(app.cfg.warp_jitter_triggered)
        self.assertIn("matches", app.cfg.warp_banner_message)

    def test_unrelated_photo_is_rejected_and_nothing_is_written(self):
        app = make_app()
        ok = app.warp_and_adapt_json_to_new_image(self.src_path, self.other_path, self.src_json, self.out_json)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(self.out_json), "a rejected alignment must not import anything")
        self.assertFalse(app.cfg.warp_jitter_triggered, "no WARP SYNC banner for a rejected alignment")
        self.assertTrue(app.cfg.import_warning_triggered)

    def test_source_mask_covers_the_shapes_plus_a_margin_only(self):
        app = make_app()
        mask = app._build_warp_source_mask((600, 800), [{"points": self.crack}])
        self.assertEqual(mask[260, 260], 255, "inside the cracks' hull")
        self.assertEqual(mask[5, 790], 0, "far corner stays out of the mask")
        self.assertIsNone(app._build_warp_source_mask((600, 800), []), "no shapes: whole image is used")

    def test_degenerate_homographies_are_flagged(self):
        app = make_app()
        self.assertFalse(app._is_homography_degenerate(np.eye(3)))
        self.assertTrue(app._is_homography_degenerate(np.diag([-1.0, 1.0, 1.0])), "mirror flip")
        self.assertTrue(app._is_homography_degenerate(np.diag([10.0, 10.0, 1.0])), "implausible zoom")


class TestFileCounterKeepsCountingInMode1(unittest.TestCase):

    def test_counter_rises_while_saving_and_skipping(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_counter_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=5)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)
            shown = []
            original_start = app._start_image_session

            def _record(queue_pos, index, total_files):
                shown.append((index, total_files))
                return original_start(queue_pos, index, total_files)

            # save, save, skip, save, save
            keys = [ord('q'), ord('q'), 4, ord('q'), ord('q')]
            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 mock.patch.object(app, "_start_image_session", side_effect=_record), \
                 _MockedHighGui(key_sequence=keys), \
                 mock.patch.object(app, "_prompt_queue_exhausted", return_value=False):
                app.run()
            self.assertEqual(shown, [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()


class TestMode2SaveWritesOverlayAndMasksToSegmentatedImages(unittest.TestCase):
    """USER REPORT (v1.0.2): edits saved in Mode 2 did not show up in the overlay/binary masks --
    they were written next to the JSON in 'already processed images' instead of 'segmentated images'."""

    def test_mode2_save_updates_the_files_in_segmentated_images(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_mode2_save_")
        try:
            archive_dir = os.path.join(tmpdir, "already processed images")
            seg_dir = os.path.join(tmpdir, "segmentated images")
            os.makedirs(archive_dir)
            os.makedirs(seg_dir)
            img = np.full((220, 300, 3), 190, dtype=np.uint8)
            cv2.line(img, (20, 20), (280, 200), (30, 30, 30), 2)
            cv2.imwrite(os.path.join(archive_dir, "photo__BLDG001.png"), img)
            crack = {"label": "crack_photo__BLDG001.png", "points": [[20.0, 20.0], [150.0, 110.0], [280.0, 200.0]],
                     "group_id": 1, "shape_type": "linestrip", "flags": {}}
            with open(os.path.join(archive_dir, "photo__BLDG001.json"), "w", encoding="utf-8") as f:
                json.dump({"shapes": [crack], "imagePath": "photo__BLDG001.png", "imageHeight": 220, "imageWidth": 300}, f)
            # Stale outputs from the first (Mode 1) save, which Mode 2 must overwrite.
            stale_overlay = os.path.join(seg_dir, "photo__BLDG001-seg.png")
            cv2.imwrite(stale_overlay, np.zeros((5, 5, 3), dtype=np.uint8))

            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)
            # The archive step resolves its folder via resolve_script_dir(): point it at tmpdir like cfg.SCRIPT_DIR.
            with mock.patch("smart_segmentation.resolve_script_dir", return_value=tmpdir), \
                 mock.patch.object(app, "_prompt_mode_choice", return_value="2"), \
                 mock.patch.object(app, "render_scene"), \
                 _MockedHighGui(key_sequence=[ord('q')]):
                app.run()

            overlay = cv2.imread(stale_overlay)
            self.assertEqual(overlay.shape[:2], (220, 300), "the overlay in 'segmentated images' must be rewritten")
            self.assertTrue(os.path.exists(os.path.join(seg_dir, "photo__BLDG001-crack_mask.png")))
            self.assertEqual([f for f in os.listdir(archive_dir) if "-seg" in f or "_mask" in f], [],
                             "no overlay/mask files may land in 'already processed images'")
            self.assertTrue(os.path.exists(os.path.join(archive_dir, "photo__BLDG001.json")), "JSON stays in the archive")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()

    def test_removing_every_crack_deletes_the_outdated_mask(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_stale_mask_")
        try:
            app = make_app()
            stale = os.path.join(tmpdir, "x-crack_mask.png")
            cv2.imwrite(stale, np.full((5, 5), 255, dtype=np.uint8))
            self.assertFalse(app._write_mask_if_nonempty(np.zeros((5, 5), dtype=np.uint8), stale, "mask"))
            self.assertFalse(os.path.exists(stale))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_overlay_name_follows_a_building_group_rename(self):
        app = make_app()
        tmpdir = tempfile.mkdtemp(prefix="crackseg_seg_name_")
        try:
            app.cfg.SCRIPT_DIR = tmpdir
            app.cfg.CURRENT_IMAGE_PATH = os.path.join(tmpdir, "Images", "DJI_0201__BLDG003.JPG")
            self.assertEqual(os.path.basename(app._segmented_output_path("-seg", ".JPG")), "DJI_0201__BLDG003-seg.JPG")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
