"""test_v110_uncertain_trace.py

Headless tests for the 1.1.0 [Uncertain crack] button (right sidebar, Tools menu, code 7):
  - pressed before tracing, every new crack is saved as uncertain by the operator ('uncertain_manual'),
    and JSON and binary masks are written as soon as the trace is completed;
  - pressed again, tracing goes back to normal;
  - the uncertain cracks are drawn red (on screen and in the -seg overlay), the others stay blue;
  - the flag survives reloading the JSON, LINK / JOIN merges, and Remove uncertain removes them.

Run with:  python3 -m unittest test_v110_uncertain_trace -v
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
import uncertain_trace as ut
from test_v105_link import _two_crack_app

RED, BLUE = ut.UNCERTAIN_MANUAL_BGR, (255, 0, 0)


def _good_quality(path):
    return 90.0, 0.9, False


class _AppCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v110_unc_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        self.app.cfg.current_tool = 'crack'
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _press(self):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ut.UNCERTAIN_TRACE_CODE)

    def _trace(self, start, end):
        self.app.cfg.temp_start = start
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._complete_new_crack_trace(end)
        return self.app.cfg.saved_cracks[-1]

    def _json_cracks(self):
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            return [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]

    def _mask(self, suffix):
        return cv2.imread(os.path.join(self.tmpdir, "Binary files", f"wall{suffix}.png"), cv2.IMREAD_GRAYSCALE)


class TestToggle(_AppCase):

    def test_off_by_default_and_button_toggles(self):
        self.assertFalse(self.app.cfg.uncertain_trace_mode)
        self._press()
        self.assertTrue(self.app.cfg.uncertain_trace_mode)
        self._press()
        self.assertFalse(self.app.cfg.uncertain_trace_mode)

    def test_code_is_not_a_printable_key(self):
        self.assertLess(ut.UNCERTAIN_TRACE_CODE, 32, "must not shadow a typed character (32 is Space)")
        self.assertNotIn(ut.UNCERTAIN_TRACE_CODE, (0, 1, 2, 3, 8, 9, 10, 13, 27), "arrows / Tab / Enter / etc.")


class TestTracing(_AppCase):

    def test_crack_traced_in_uncertain_mode_is_saved_right_away(self):
        self.app.cfg.saved_cracks = []
        self._press()
        crack = self._trace((20, 100), (120, 100))
        self.assertTrue(crack.get('uncertain_manual'))
        shapes = self._json_cracks()
        self.assertEqual(len(shapes), 1)
        self.assertTrue(shapes[0]["label"].startswith(self.app.cfg.CRACK_UNCERTAIN_LABEL))
        self.assertEqual(shapes[0]["flags"], {"uncertain": True, "uncertain_manual": True})
        self.assertEqual(self._mask("-crack_uncertain_mask")[100, 70], 255)
        self.assertEqual(self._mask("-crack_mask")[100, 70], 255, "the crack mask still holds every crack")

    def test_normal_mode_is_unchanged(self):
        self.app.cfg.saved_cracks = []
        crack = self._trace((20, 100), (120, 100))
        self.assertNotIn('uncertain_manual', crack)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH), "no save before [S], as before")

    def test_pressing_again_goes_back_to_normal(self):
        self.app.cfg.saved_cracks = []
        self._press()
        first = self._trace((20, 100), (120, 100))
        self._press()
        second = self._trace((160, 100), (280, 100))
        self.assertTrue(first.get('uncertain_manual'))
        self.assertNotIn('uncertain_manual', second)
        self.app._save_annotations_now()
        labels = sorted(s["flags"].get("uncertain_manual", False) for s in self._json_cracks())
        self.assertEqual(labels, [False, True])
        unc = self._mask("-crack_uncertain_mask")
        self.assertEqual((unc[100, 70], unc[100, 220]), (255, 0))

    def test_undo_removes_the_uncertain_trace(self):
        self.app.cfg.saved_cracks = []
        self._press()
        self._trace((20, 100), (120, 100))
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('u'))
        self.assertEqual(self.app.cfg.saved_cracks, [])


class TestColours(_AppCase):

    def _scene(self):
        self.app.recalculate_masks()
        scene = np.zeros((self.app.cfg.H_img, self.app.cfg.W_img, 3), dtype=np.uint8)
        self.app._compute_scene_overlays(scene, self.app.cfg.H_img, self.app.cfg.W_img)
        return scene

    def test_uncertain_crack_is_red_the_other_blue(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        scene = self._scene()
        self.assertEqual(tuple(scene[100, 60]), BLUE)
        self.assertEqual(tuple(scene[100, 220]), RED)

    def test_red_also_in_a_zoomed_region(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        self.app.recalculate_masks()
        region = (90, 110, 200, 240)
        scene = np.zeros((20, 40, 3), dtype=np.uint8)
        self.app._compute_scene_overlays(scene, self.app.cfg.H_img, self.app.cfg.W_img, region)
        self.assertEqual(tuple(scene[10, 20]), RED)

    def test_overlay_off_hides_red_too(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        self.app.cfg.show_crack_overlay = False
        self.assertFalse(self._scene().any())

    def test_trace_in_progress_is_red_in_uncertain_mode(self):
        self.assertEqual(self.app._temp_trace_bgr(), (0, 165, 255))
        self.app.cfg.uncertain_trace_mode = True
        self.assertEqual(self.app._temp_trace_bgr(), RED)

    def test_seg_overlay_export_is_red(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        self.app.cfg.SAVE_SEG_IMAGE = True
        self.app.recalculate_masks()
        self.app._export_segmented_preview_image()
        seg = cv2.imread(self.app.cfg.EXPORT_IMAGE_PATH)
        self.assertEqual(tuple(seg[100, 220]), RED)
        self.assertEqual(tuple(seg[100, 60]), BLUE)


class TestFlagKept(_AppCase):

    def test_reloaded_from_json(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        self.app._save_annotations_now()
        shapes = self._json_cracks()
        loaded = [self.app._build_loaded_crack(s, [(int(p[0]), int(p[1])) for p in s["points"]]) for s in shapes]
        self.assertEqual(sorted(bool(c.get('uncertain_manual')) for c in loaded), [False, True])

    def test_auto_uncertain_crack_is_not_turned_into_manual(self):
        shape = {"label": "crack_incerta_wall.png", "points": [[0, 0], [5, 0]], "shape_type": "linestrip",
                 "flags": {"uncertain": True}}
        self.assertNotIn('uncertain_manual', self.app._build_loaded_crack(shape, [(0, 0), (5, 0)]))

    def test_link_with_an_uncertain_crack_stays_uncertain(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        self.assertTrue(self.app._link_cracks((0, 100), (1, 0)))
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertTrue(self.app.cfg.saved_cracks[0].get('uncertain_manual'))

    def test_remove_uncertain_removes_manual_ones(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=True), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.REMOVE_UNCERTAIN_CODE)
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[0])

    def test_info_list_marks_it(self):
        self.app.cfg.saved_cracks[1]['uncertain_manual'] = True
        entries = self.app._build_crack_reliability_entries()
        self.assertTrue(entries[1].endswith("[?]"))
        self.assertFalse(entries[0].endswith("[?]"))


class TestHud(_AppCase):

    def test_label_only_in_uncertain_mode(self):
        win = np.zeros((300, 700, 3), dtype=np.uint8)
        self.app._draw_uncertain_trace_label(win)
        self.assertFalse(win.any())
        self.app.cfg.uncertain_trace_mode = True
        self.app._draw_uncertain_trace_label(win)
        self.assertTrue(win.any())


class TestSidebar(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def _button(self):
        return next(b for b in self.app._sidebar.buttons if "Uncertain crack" in b.text)

    def test_button_sends_its_code(self):
        self.app._pending_keys.clear()
        self._button().command()
        self.assertEqual(self.app._pending_keys, [ut.UNCERTAIN_TRACE_CODE])

    def test_button_looks_pressed_while_on(self):
        cfg, btn = self.app.cfg, self._button()
        cfg.uncertain_trace_mode = True
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "sunken")
        cfg.uncertain_trace_mode = False
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "raised")

    def test_in_the_menu_and_code_unique(self):
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertEqual(menu_codes.count(ut.UNCERTAIN_TRACE_CODE), 1)
        sidebar_codes = [entry[3] for entry in self.gm._SIDEBAR_SHORTCUTS]
        self.assertEqual(sidebar_codes.count(ut.UNCERTAIN_TRACE_CODE), 1)


if __name__ == "__main__":
    unittest.main()
