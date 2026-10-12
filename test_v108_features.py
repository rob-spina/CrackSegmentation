"""test_v108_features.py

Headless tests for 1.0.8:
  - The sidebar title reads "More Features".
  - Eraser (sidebar, code 21): a window follows the mouse; dragging it over a crack erases that part
    (a crack crossed in the middle is split in two) and updates the JSON and the binary crack mask.
  - Remove uncertain (sidebar, code 22): deletes every crack marked [?] and updates the JSON and masks.
  (Validation available once every photo is segmented: see test_v106_features.TestValidationRun.)

Run with:  python3 -m unittest test_v108_features -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import crack_cleanup as cc
from test_v105_link import _two_crack_app


def _mouse(app, event, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.mouse_callback(event, wx, wy, 0, None)


def _drag(app, points):
    _mouse(app, cv2.EVENT_LBUTTONDOWN, *points[0])
    for p in points[1:]:
        _mouse(app, cv2.EVENT_MOUSEMOVE, *p)
    _mouse(app, cv2.EVENT_LBUTTONUP, *points[-1])


def _xs(app):
    return sorted(int(p[0]) for c in app.cfg.saved_cracks for p in c['path'])


class TestHelpers(unittest.TestCase):

    def test_kept_runs(self):
        self.assertEqual(cc.kept_runs([False, False, True, True, False]), [(0, 1), (4, 4)])
        self.assertEqual(cc.kept_runs([True, True]), [])
        self.assertEqual(cc.kept_runs([False, False]), [(0, 1)])

    def test_erased_runs(self):
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        self.assertEqual(cc.erased_runs(path, [True, False, True, True, False]), [[(0, 0)], [(2, 0), (3, 0)]])

    def test_sweep_centres_leave_no_gap(self):
        centres = cc.sweep_centres((0, 0), (100, 0), 10)
        self.assertEqual(centres[-1], (100, 0))
        steps = [centres[0][0]] + [b[0] - a[0] for a, b in zip(centres, centres[1:])]
        self.assertTrue(all(s <= 10 for s in steps))
        self.assertEqual(cc.sweep_centres(None, (5, 5), 10), [(5, 5)])


class TestEraser(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_eraser_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)

    def test_button_toggles_the_tool(self):
        self.assertTrue(self.app.cfg.eraser_state["active"])
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)
        self.assertFalse(self.app.cfg.eraser_state["active"])

    def test_drag_through_the_middle_splits_the_crack(self):
        _drag(self.app, [(60, 100), (65, 100), (70, 100), (80, 100)])
        xs = _xs(self.app)
        self.assertFalse(any(60 <= x <= 80 for x in xs), "the dragged stretch is gone")
        self.assertTrue(all(x in xs for x in list(range(20, 50)) + list(range(90, 121)) + list(range(160, 281))))
        self.assertEqual(len(self.app.cfg.saved_cracks), 3, "crack 1 split in two, crack 2 untouched")
        for c in self.app.cfg.saved_cracks:
            self.assertEqual((tuple(c['start']), tuple(c['end'])), (tuple(c['path'][0]), tuple(c['path'][-1])))

    def test_fast_drag_leaves_no_gap(self):
        # Only two mouse events far apart: everything between them is erased too.
        _drag(self.app, [(40, 100), (110, 100)])
        self.assertFalse(any(40 <= x <= 110 for x in _xs(self.app)))

    def test_release_updates_json_and_binary_mask(self):
        _drag(self.app, [(60, 100), (80, 100)])
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(len(shapes), 3)
        json_xs = {int(p[0]) for s in shapes for p in s["points"]}
        self.assertFalse(any(62 <= x <= 78 for x in json_xs))
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask[100, 70], 0, "erased stretch is background in the binary mask")
        self.assertEqual(mask[100, 30], 255)
        self.assertEqual(mask[100, 200], 255)

    def test_erasing_a_whole_crack_removes_it(self):
        _drag(self.app, [(155, 100), (200, 100), (240, 100), (285, 100)])
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertFalse(any(x >= 160 for x in _xs(self.app)))

    def test_undo_restores_the_crack(self):
        before = _xs(self.app)
        _drag(self.app, [(60, 100), (80, 100)])
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('u'))
        self.assertEqual(_xs(self.app), before)

    def test_click_away_from_cracks_changes_nothing(self):
        before = _xs(self.app)
        _drag(self.app, [(60, 30), (90, 30)])
        self.assertEqual(_xs(self.app), before)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH), "nothing to save")
        self.assertEqual(self.app.cfg.action_history, [])

    def test_hover_moves_the_window_without_erasing(self):
        before = _xs(self.app)
        _mouse(self.app, cv2.EVENT_MOUSEMOVE, 70, 100)
        self.assertEqual(_xs(self.app), before)
        self.assertIsNotNone(self.app.cfg.eraser_state["pos"])

    def test_brackets_resize_the_window(self):
        size = self.app.cfg.ERASER_SIZE_WIN_PX
        self.app.process_keypress(ord(']'))
        self.assertEqual(self.app.cfg.ERASER_SIZE_WIN_PX, size + self.app.cfg.ERASER_SIZE_STEP_WIN_PX)
        for _ in range(100):
            self.app.process_keypress(ord('['))
        self.assertEqual(self.app.cfg.ERASER_SIZE_WIN_PX, self.app.cfg.ERASER_MIN_SIZE_WIN_PX)

    def test_erases_while_zoom_window_is_on(self):
        self.app.cfg.zoom_window_state["active"] = True
        _drag(self.app, [(60, 100), (80, 100)])
        self.assertFalse(any(65 <= x <= 75 for x in _xs(self.app)))

    def test_other_tools_close_the_eraser(self):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('1'))
        self.assertFalse(self.app.cfg.eraser_state["active"])
        self.assertEqual(self.app.cfg.crack_cut_join_state["mode"], "cut")
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)
        self.assertTrue(self.app.cfg.eraser_state["active"])
        self.assertIsNone(self.app.cfg.crack_cut_join_state["mode"], "the eraser closes CUT")
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('c'))
        self.assertFalse(self.app.cfg.eraser_state["active"])

    def test_window_is_drawn(self):
        _mouse(self.app, cv2.EVENT_MOUSEMOVE, 70, 100)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_eraser(win)
        self.assertTrue(win.any())


class TestRemoveUncertain(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_unc_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        self.uncertain_path = self.app.cfg.saved_cracks[1]['path']
        quality = lambda path: (30.0, 0.4, True) if path == self.uncertain_path else (90.0, 0.9, False)
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_removes_only_uncertain_cracks_and_updates_files(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=True) as confirm, \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.REMOVE_UNCERTAIN_CODE)
        confirm.assert_called_once_with(1)
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertFalse(any(x >= 160 for x in _xs(self.app)))
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(len(shapes), 1)
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask[100, 200], 0)
        self.assertEqual(mask[100, 50], 255)

    def test_cancel_removes_nothing(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=False):
            self.assertFalse(self.app._remove_uncertain_cracks())
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH))

    def test_no_uncertain_crack_asks_nothing(self):
        self.app.cfg.saved_cracks.pop(1)
        with mock.patch.object(self.app, "_confirm_remove_uncertain") as confirm:
            self.assertFalse(self.app._remove_uncertain_cracks())
        confirm.assert_not_called()

    def test_undo_brings_them_back(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=True), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._remove_uncertain_cracks()
            self.app.process_keypress(ord('u'))
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)


class TestSidebar(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_title_reads_more_features(self):
        import inspect
        src = inspect.getsource(self.gm.ShortcutsSidebar.__init__)
        self.assertIn('"More Features"', src)
        self.assertNotIn("More Shortcuts", src)

    def test_new_buttons_send_their_codes(self):
        for label, code in (("Eraser", cc.ERASER_CODE), ("Remove uncertain", cc.REMOVE_UNCERTAIN_CODE)):
            btn = next(b for b in self.app._sidebar.buttons if label in b.text)
            self.app._pending_keys.clear()
            btn.command()
            self.assertEqual(self.app._pending_keys, [code])

    def test_eraser_button_looks_pressed_while_on(self):
        cfg = self.app.cfg
        btn = next(b for b in self.app._sidebar.buttons if "Eraser" in b.text)
        cfg.eraser_state["active"] = True
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "sunken")
        cfg.eraser_state["active"] = False
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "raised")

    def test_new_commands_are_in_the_menu(self):
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(cc.ERASER_CODE, menu_codes)
        self.assertIn(cc.REMOVE_UNCERTAIN_CODE, menu_codes)


if __name__ == "__main__":
    unittest.main()
