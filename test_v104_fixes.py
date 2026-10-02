"""test_v104_fixes.py

Headless tests for the changes requested on v1.0.3:
  - [S] saves and stays on the image (repeatable), [Enter] saves and moves on.
  - Cut [1] / Join [2] crack tools, with undo/redo.
  - J info panel follows the visible part of a scrolled canvas.
  - Smoother zoomed panning (clamped centre, crop-first rendering).

Run with:  python3 -m unittest test_v104_fixes -v
"""
import io
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


class TestSaveKeyStaysOnImage(unittest.TestCase):

    def test_s_twice_then_enter_saves_three_times_and_then_moves_on(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_s_stay_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=2)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)
            shown = []
            original_start = app._start_image_session

            def _record(queue_pos, index, total_files):
                shown.append((os.path.basename(app.cfg.image_queue[queue_pos])[:6], index, total_files))
                return original_start(queue_pos, index, total_files)

            saves = []
            original_export = app.export_labelme_format

            def _count_export():
                saves.append(app.cfg.CURRENT_IMAGE_PATH)
                return original_export()

            with mock.patch("smart_segmentation.resolve_script_dir", return_value=tmpdir), \
                 mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 mock.patch.object(app, "_start_image_session", side_effect=_record), \
                 mock.patch.object(app, "export_labelme_format", side_effect=_count_export), \
                 mock.patch.object(app, "_prompt_queue_exhausted", return_value=False), \
                 mock.patch("sys.stderr", new_callable=io.StringIO) as err, \
                 _MockedHighGui(key_sequence=[ord('s'), ord('s'), 13, 4]):
                app.run()

            self.assertEqual(shown, [("img_00", 1, 2), ("img_01", 2, 2)])
            self.assertEqual(len(saves), 3, "S, S and Enter must each save")
            archive = os.path.join(tmpdir, "already processed images")
            self.assertTrue(any(f.startswith("img_00") and f.endswith(".json") for f in os.listdir(archive)))
            self.assertTrue(any(f.startswith("img_00") and f.endswith(".png") for f in os.listdir(archive)))
            self.assertNotIn("IO ERROR", err.getvalue(), "a repeated save must not fail on already-archived files")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()


def _two_route_app():
    """A 200x300 photo whose skeleton offers two routes from (20, 100) to (280, 100):
    the straight one (a wall stain, say) and a detour bulging up to y=40."""
    app = make_app()
    h, w = 200, 300
    app.cfg.H_img, app.cfg.W_img = h, w
    app.cfg.zoom_box = [0, 0, w, h]
    skel = np.zeros((h, w), dtype=np.uint8)
    cv2.line(skel, (20, 100), (280, 100), 255, 1)                       # straight route
    cv2.polylines(skel, [np.array([[60, 100], [100, 40], [200, 40], [240, 100]], np.int32)], False, 255, 1)  # detour
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    straight = [(x, 100) for x in range(20, 281)]
    app.cfg.saved_cracks = [{'start': straight[0], 'end': straight[-1], 'path': list(straight),
                             'active': True, 'session_id': 'x',
                             'width_segments': [{'i0': 0, 'i1': 10, 'width_px': 2}, {'i0': 100, 'i1': 120, 'width_px': 2}]}]
    app.cfg.blue_visual_mask = np.zeros((h, w), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((h, w), dtype=np.uint8)
    return app


def _max_step(path):
    return max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(path, path[1:]))


class TestCutTool(unittest.TestCase):

    def test_cut_splits_the_crack_and_undo_redo_round_trip(self):
        app = _two_route_app()
        original = [dict(c) for c in app.cfg.saved_cracks]
        self.assertTrue(app._cut_crack_section((0, 80), (0, 180)))
        paths = [c['path'] for c in app.cfg.saved_cracks]
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0][-1], (100, 100))
        self.assertEqual(paths[1][0], (200, 100))
        self.assertEqual(len(app.cfg.cut_exclusions), 1)
        # The first width tract (0-10) survives; the one crossing the cut (100-120) is removed.
        self.assertEqual([(s['i0'], s['i1']) for s in app.cfg.saved_cracks[0].get('width_segments', [])], [(0, 10)])
        app._undo_last_action()
        self.assertEqual(app.cfg.saved_cracks[0]['path'], original[0]['path'])
        self.assertEqual(app.cfg.cut_exclusions, [])
        app._handle_redo_key()
        self.assertEqual(len(app.cfg.saved_cracks), 2)

    def test_cut_at_an_end_trims_instead_of_splitting(self):
        app = _two_route_app()
        app._cut_crack_section((0, 0), (0, 50))
        self.assertEqual(len(app.cfg.saved_cracks), 1)
        self.assertEqual(app.cfg.saved_cracks[0]['start'], (70, 100))

    def test_points_on_different_cracks_are_refused(self):
        app = _two_route_app()
        app.cfg.saved_cracks.append(dict(app.cfg.saved_cracks[0]))
        self.assertFalse(app._cut_crack_section((0, 10), (1, 50)))
        self.assertEqual(len(app.cfg.saved_cracks), 2)


class TestJoinTool(unittest.TestCase):

    def test_join_on_the_same_crack_takes_the_other_route(self):
        app = _two_route_app()
        self.assertTrue(app._join_crack_points((0, 30), (0, 230)))
        path = app.cfg.saved_cracks[0]['path']
        self.assertEqual(_max_step(path), 1, "the re-routed crack must stay continuous")
        self.assertTrue(any(y <= 45 for _, y in path), "the new route must follow the detour")
        self.assertEqual((path[0], path[-1]), ((20, 100), (280, 100)), "the untouched ends stay")

    def test_cut_then_join_reconnects_the_pieces_around_the_removed_part(self):
        app = _two_route_app()
        app._cut_crack_section((0, 50), (0, 210))
        piece_a, piece_b = (c['path'] for c in app.cfg.saved_cracks)
        self.assertTrue(app._join_crack_points((0, len(piece_a) - 1), (1, 0)))
        self.assertEqual(len(app.cfg.saved_cracks), 1)
        path = app.cfg.saved_cracks[0]['path']
        self.assertEqual(_max_step(path), 1)
        self.assertTrue(any(y <= 45 for _, y in path), "JOIN must not walk back over the removed stain")
        app._undo_last_action()
        self.assertEqual(len(app.cfg.saved_cracks), 2)

    def test_two_clicks_with_the_tool_keys(self):
        app = _two_route_app()
        app.handle_keyboard(ord('1'))
        self.assertEqual(app.cfg.crack_cut_join_state["mode"], 'cut')
        sx, sy = app.transform_real_to_window_coords(100, 100)
        ex, ey = app.transform_real_to_window_coords(200, 100)
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, sx, sy, 0, None)
            self.assertIsNotNone(app.cfg.crack_cut_join_state["first"])
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, ex, ey, 0, None)
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        app.handle_keyboard(ord('2'))
        self.assertEqual(app.cfg.crack_cut_join_state["mode"], 'join')
        app.handle_keyboard(ord('2'))
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "pressing the key again leaves the tool")

    def test_other_tools_switch_cut_join_off(self):
        app = _two_route_app()
        app.handle_keyboard(ord('1'))
        app.handle_keyboard(ord('c'))
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"])


class TestInfoPanelFollowsVisibleTop(unittest.TestCase):

    def test_panel_is_drawn_from_the_visible_top(self):
        app = make_app()
        app.cfg.show_info_overlay = True
        app.cfg.CURRENT_IMAGE_PATH = None
        win_out = np.full((900, 1200, 3), 255, dtype=np.uint8)
        hm = app._shift_hud_metrics(app._compute_hud_font_metrics(), 300)
        app._draw_info_overlay(win_out, "status", "metrics", hm)
        self.assertTrue(np.all(win_out[100, 400] == 255), "nothing drawn above the visible top")
        self.assertFalse(np.all(win_out[320, 400] == 255), "panel drawn just below the visible top")


class TestSmootherPanning(unittest.TestCase):

    def _zoomed_app(self):
        app = make_app()
        app.cfg.W_img, app.cfg.H_img = 4000, 3000
        app.cfg.zoom_factor = 4.0
        app.cfg.zoom_center = [2000, 1500]
        app.refresh_zoom_viewport()
        return app

    def test_centre_cannot_drift_past_the_edge(self):
        app = self._zoomed_app()
        for _ in range(50):
            app.handle_keyboard(3)  # Right
        right_box = list(app.cfg.zoom_box)
        self.assertEqual(right_box[2], 4000)
        app.handle_keyboard(2)  # one Left press must move the view immediately
        self.assertLess(app.cfg.zoom_box[2], 4000)

    def test_arrow_step_is_a_fraction_of_the_view(self):
        app = self._zoomed_app()
        x0 = app.cfg.zoom_box[0]
        app.handle_keyboard(3)
        self.assertEqual(app.cfg.zoom_box[0] - x0, int(0.08 * 4000 / 4.0))

    def test_pan_by_fraction_only_when_zoomed(self):
        app = self._zoomed_app()
        x0 = app.cfg.zoom_box[0]
        self.assertTrue(app.pan_view_by_fraction(0.1, 0.0))
        self.assertEqual(app.cfg.zoom_box[0] - x0, 100)
        app.cfg.zoom_factor = 1.0
        self.assertFalse(app.pan_view_by_fraction(0.1, 0.0))

    def test_crop_first_render_matches_a_full_frame_render(self):
        app = _two_route_app()
        app.cfg.img_background = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
        app.cfg.saved_detachments = [{'path': [(150, 20), (220, 20), (220, 80)], 'active': True}]
        app.cfg.temp_path = [(50 + i, 150) for i in range(40)]
        app.cfg.current_tool = 'crack'
        app.recalculate_masks()
        for zoom, centre in ((1.0, (150, 100)), (2.0, (150, 100)), (3.0, (10, 10)), (4.0, (299, 199))):
            app.cfg.zoom_factor, app.cfg.zoom_center = zoom, list(centre)
            app.refresh_zoom_viewport()
            full = app.cfg.img_background.copy()
            app._paint_temp_crack_preview(full)
            app._compute_scene_overlays(full, 200, 300)
            x0, y0, x1, y1 = app.cfg.zoom_box
            with mock.patch.object(app, "_resize_roi_to_window", side_effect=lambda roi: roi.copy()):
                fast = app._build_window_frame()
            self.assertTrue(np.array_equal(fast, full[y0:y1, x0:x1]), f"zoom {zoom} centre {centre}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
