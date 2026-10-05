"""test_v105_fixes.py

Headless tests for the changes requested on v1.0.4 (release 1.0.5):
  - Cut [1] / Join [2]: a completed operation closes the tool (no stale "click the second point").
  - The crack's un-numbered end is drawn as a tiny square (4 px at most).
  - New data layout: Images / Binary files / JSON files, with automatic migration of the old folders.

Run with:  python3 -m unittest test_v105_fixes -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from config import Config, migrate_legacy_folders
from smart_segmentation import CrackSegmentation
from test_gui_shell_parity import _MockedHighGui, make_app, make_synthetic_image_folder
from test_v104_fixes import _two_route_app


class TestCutJoinClosesAfterSecondPoint(unittest.TestCase):

    def _click(self, app, x, y):
        wx, wy = app.transform_real_to_window_coords(x, y)
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)

    def test_cut_closes_the_tool_after_the_second_point(self):
        app = _two_route_app()
        app.handle_keyboard(ord('1'))
        self._click(app, 100, 100)
        self._click(app, 200, 100)
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "a completed CUT must close the tool")
        self.assertIsNone(app.cfg.crack_cut_join_state["first"])

    def test_join_closes_the_tool_and_hides_the_indicator(self):
        app = _two_route_app()
        app._cut_crack_section((0, 50), (0, 210))
        piece_a, piece_b = (c['path'] for c in app.cfg.saved_cracks)
        app.handle_keyboard(ord('2'))
        self._click(app, *piece_a[-1])
        self._click(app, *piece_b[0])
        self.assertEqual(len(app.cfg.saved_cracks), 1, "the two pieces must be joined")
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "a completed JOIN must close the tool")
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_cut_join_indicator(win_out)
        self.assertFalse(win_out.any(), "no 'click the second point' text may remain on screen")

    def test_failed_operation_keeps_the_tool_open(self):
        app = _two_route_app()
        app.handle_keyboard(ord('1'))
        self._click(app, 100, 100)
        self._click(app, 101, 100)  # too close: nothing to remove
        self.assertEqual(app.cfg.crack_cut_join_state["mode"], 'cut', "a failed CUT stays active for a retry")
        self.assertIsNone(app.cfg.crack_cut_join_state["first"])


class TestCrackEndMarker(unittest.TestCase):

    def _draw(self, active=True):
        app = make_app()
        app.cfg.W_img, app.cfg.H_img = 1200, 900
        app.cfg.zoom_box = [0, 0, 1200, 900]
        crack = {'start': (200, 200), 'end': (800, 600), 'path': [(200, 200), (800, 600)], 'active': active}
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        with mock.patch.object(app, "transform_real_to_window_coords", side_effect=lambda x, y: (x, y)):
            app._draw_one_crack_marker(win_out, crack, 18, number=1)
        return win_out

    def test_end_marker_is_at_most_4_pixels(self):
        win_out = self._draw()
        ys, xs = np.nonzero(win_out[560:640, 760:840].any(axis=2))
        self.assertTrue(len(xs), "the end must still be marked")
        self.assertLessEqual(xs.max() - xs.min() + 1, 4)
        self.assertLessEqual(ys.max() - ys.min() + 1, 4)

    def test_start_marker_keeps_the_numbered_square(self):
        win_out = self._draw()
        ys, xs = np.nonzero(win_out[160:240, 160:240].any(axis=2))
        self.assertGreaterEqual(xs.max() - xs.min() + 1, 30, "the numbered start marker is unchanged")

    def test_soft_deleted_end_is_still_tiny(self):
        win_out = self._draw(active=False)
        ys, xs = np.nonzero(win_out[560:640, 760:840].any(axis=2))
        self.assertLessEqual(xs.max() - xs.min() + 1, 4)


class TestDataFolderLayout(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v15_layout_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _app(self):
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        return CrackSegmentation(cfg)

    def test_save_keeps_the_photo_in_images_and_splits_outputs(self):
        make_synthetic_image_folder(self.tmpdir, n_images=1)
        app = self._app()
        with mock.patch("smart_segmentation.resolve_script_dir", return_value=self.tmpdir), \
             mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
             mock.patch.object(app, "render_scene"), \
             mock.patch.object(app, "_prompt_queue_exhausted", return_value=False), \
             _MockedHighGui(key_sequence=[ord('q')]):
            app.run()
        images = os.listdir(os.path.join(self.tmpdir, "Images"))
        self.assertEqual(len(images), 1, "the original photo stays in 'Images'")
        jsons = os.listdir(os.path.join(self.tmpdir, "JSON files"))
        self.assertEqual([os.path.splitext(f)[1] for f in jsons], [".json"])
        binaries = os.listdir(os.path.join(self.tmpdir, "Binary files"))
        self.assertTrue(binaries and all(f.endswith(".png") for f in binaries))
        for legacy in ("already processed images", "segmentated images"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, legacy)))

    def test_mode1_skips_photos_that_already_have_a_json(self):
        make_synthetic_image_folder(self.tmpdir, n_images=3)
        json_dir = os.path.join(self.tmpdir, "JSON files")
        os.makedirs(json_dir)
        with open(os.path.join(json_dir, "img_01.json"), "w", encoding="utf-8") as f:
            json.dump({"shapes": [], "imagePath": "img_01.png"}, f)
        app = self._app()
        app._load_mode1_queue(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        self.assertEqual(sorted(os.path.basename(p) for p in app.cfg.image_queue), ["img_00.png", "img_02.png"])
        app.cfg.image_queue = []
        app._load_mode2_queue(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        self.assertEqual([os.path.basename(p) for p in app.cfg.image_queue], ["img_01.png"])
        self.assertEqual(app.cfg.OUTPUT_FOLDER, json_dir)

    def test_migration_moves_old_files_without_overwriting(self):
        archive = os.path.join(self.tmpdir, "already processed images")
        seg = os.path.join(self.tmpdir, "segmentated images")
        os.makedirs(archive)
        os.makedirs(seg)
        files = {
            os.path.join(archive, "a__BLDG001.JPG"): b"photo",
            os.path.join(archive, "a__BLDG001.json"): b'{"v": "archive"}',
            os.path.join(archive, "a__BLDG001.json.bak"): b'{"v": "backup"}',
            os.path.join(archive, "a__BLDG001-crack_mask.png"): b"stale mask",
            os.path.join(archive, ".DS_Store"): b"",
            os.path.join(seg, "a__BLDG001.json"): b'{"v": "old copy"}',
            os.path.join(seg, "a__BLDG001-seg.JPG"): b"overlay",
            os.path.join(seg, "a__BLDG001-crack_mask.png"): b"current mask",
            os.path.join(seg, "notes.txt"): b"keep me",
        }
        for path, data in files.items():
            with open(path, "wb") as f:
                f.write(data)

        moved = migrate_legacy_folders(self.tmpdir)

        self.assertEqual(moved, 5)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "Images", "a__BLDG001.JPG")))
        json_dir = os.path.join(self.tmpdir, "JSON files")
        with open(os.path.join(json_dir, "a__BLDG001.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["v"], "archive", "the archive's JSON (edited in Mode 2) wins")
        self.assertTrue(os.path.exists(os.path.join(json_dir, "a__BLDG001.json.bak")))
        bin_dir = os.path.join(self.tmpdir, "Binary files")
        self.assertEqual(sorted(os.listdir(bin_dir)), ["a__BLDG001-crack_mask.png", "a__BLDG001-seg.JPG"])
        with open(os.path.join(bin_dir, "a__BLDG001-crack_mask.png"), "rb") as f:
            self.assertEqual(f.read(), b"current mask", "the output folder's mask (rewritten on every save) wins")
        self.assertIn("a__BLDG001-crack_mask.png", os.listdir(archive), "the stale duplicate stays, never deleted")
        self.assertEqual(sorted(os.listdir(seg)), ["a__BLDG001.json", "notes.txt"])
        self.assertEqual(migrate_legacy_folders(self.tmpdir), 0, "running it again changes nothing")

    def test_migration_removes_a_folder_left_with_only_finder_metadata(self):
        archive = os.path.join(self.tmpdir, "already processed images")
        os.makedirs(archive)
        for name in ("b.JPG", ".DS_Store"):
            with open(os.path.join(archive, name), "wb") as f:
                f.write(b"x")
        migrate_legacy_folders(self.tmpdir)
        self.assertFalse(os.path.exists(archive))


class TestComparatorWritesToBinaryFiles(unittest.TestCase):

    def test_masks_of_a_json_in_json_files_go_to_binary_files(self):
        from compare_and_filter_cracks import CrackComparator
        tmpdir = tempfile.mkdtemp(prefix="crackseg_v15_cmp_")
        try:
            json_path = os.path.join(tmpdir, "JSON files", "a.json")
            out = CrackComparator.output_dir_for_json(json_path)
            self.assertEqual(out, os.path.join(tmpdir, "Binary files"))
            self.assertTrue(os.path.isdir(out))
            other = os.path.join(tmpdir, "elsewhere", "a.json")
            self.assertEqual(CrackComparator.output_dir_for_json(other), os.path.join(tmpdir, "elsewhere"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
