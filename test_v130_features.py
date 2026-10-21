"""test_v130_features.py

Headless tests for 1.3.0:
  - [Crack labels] (code 11): hovering a crack number or line shows a temporary window with the crack's data;
    it disappears when the cursor moves away; pressed again, hovering shows nothing;
  - the CRACK_UNCERTAIN_* rule no longer marks cracks by itself (CRACK_UNCERTAIN_AUTO = False);
  - Ctrl/Cmd + click with the Crack tool next to a crack marker starts a trace instead of deleting the crack;
  - [Warn uncertain] (code 12): after a trace, a crack below the limits is proposed as uncertain (Yes/No).

Run with:  python3 -m unittest test_v130_features -v
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

import crack_labels as cl
import warn_uncertain as wu
from test_v105_link import _two_crack_app


def _weak_quality(path):
    return 20.0, 0.55, False


def _good_quality(path):
    return 90.0, 0.9, False


class _AppCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v130_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        self.app.cfg.current_tool = 'crack'

    def _key(self, code):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(code)

    def _move(self, real_x, real_y):
        wx, wy = self.app.transform_real_to_window_coords(real_x, real_y)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)

    def _click(self, real_x, real_y, flags=0):
        wx, wy = self.app.transform_real_to_window_coords(real_x, real_y)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, flags, None)

    def _trace(self, start, end):
        self.app.cfg.temp_start = start
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._complete_new_crack_trace(end)
        return self.app.cfg.saved_cracks[-1]

    def _json_cracks(self):
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            return [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]


class TestCodes(unittest.TestCase):

    def test_codes_are_free_non_printable_keys(self):
        for code in (cl.CRACK_LABELS_CODE, wu.WARN_UNCERTAIN_CODE):
            self.assertLess(code, 32)
            self.assertNotIn(code, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 27))
        self.assertNotEqual(cl.CRACK_LABELS_CODE, wu.WARN_UNCERTAIN_CODE)


class TestCrackLabels(_AppCase):

    def test_off_by_default_and_button_toggles(self):
        self.assertFalse(self.app.cfg.crack_labels_mode)
        self._key(cl.CRACK_LABELS_CODE)
        self.assertTrue(self.app.cfg.crack_labels_mode)
        self._key(cl.CRACK_LABELS_CODE)
        self.assertFalse(self.app.cfg.crack_labels_mode)

    def test_hover_on_the_line_shows_the_crack_and_moving_away_hides_it(self):
        self._key(cl.CRACK_LABELS_CODE)
        self._move(220, 101)
        self.assertEqual(self.app._hovered_crack()[0], 2)
        self._move(220, 160)
        self.assertEqual(self.app._hovered_crack(), (None, None))

    def test_hover_on_the_number_marker(self):
        self._key(cl.CRACK_LABELS_CODE)
        wx, wy = self.app.transform_real_to_window_coords(20, 100)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx + 15, wy - 15, 0, None)
        self.assertEqual(self.app._hovered_crack()[0], 1)
        self.assertEqual(self.app.cfg.crack_label_hover["anchor"], (20, 100))

    def test_pressing_again_stops_the_labels(self):
        self._key(cl.CRACK_LABELS_CODE)
        self._move(220, 100)
        self._key(cl.CRACK_LABELS_CODE)
        self.assertEqual(self.app._hovered_crack(), (None, None))
        self._move(220, 100)
        self.assertIsNone(self.app.cfg.crack_label_hover)

    def test_hover_does_not_stop_tracing_preview(self):
        self._key(cl.CRACK_LABELS_CODE)
        self.app.cfg.temp_start = (20, 150)
        self.app.cfg.temp_path = [(20, 150)]
        self._move(220, 100)
        last = self.app.cfg.temp_path[-1]
        self.assertTrue(abs(last[0] - 220) <= 1 and abs(last[1] - 100) <= 1, "the move still reaches the Crack tool")

    def test_label_content(self):
        crack = self.app.cfg.saved_cracks[0]
        with mock.patch.object(self.app, "_crack_quality", side_effect=_weak_quality):
            title, lines = self.app.build_crack_label(1, crack)
        text = "\n".join(t for t, _ in lines)
        self.assertEqual(title, "CRACK 1")
        self.assertIn("Length: 5.0 cm (100 px)", text)
        self.assertIn("Width: mean", text)
        self.assertIn("Reliability: 20%", text)
        self.assertIn("Confidence: 0.55", text)
        self.assertIn("Status: normal crack", text)
        self.assertIn("Below threshold", text)
        self.assertNotIn("IGNORED", text)

    def test_label_shows_uncertain_and_ignored(self):
        crack = self.app.cfg.saved_cracks[1]
        crack['uncertain_manual'] = True
        self.app.cfg.saved_ignores = [{'path': [(150, 80), (290, 80), (290, 120), (150, 120)], 'active': True}]
        with mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality):
            _, lines = self.app.build_crack_label(2, crack)
        text = "\n".join(t for t, _ in lines)
        self.assertIn("UNCERTAIN", text)
        self.assertIn("IGNORED: traced inside an ignore region", text)

    def test_partly_ignored(self):
        self.app.cfg.saved_ignores = [{'path': [(20, 80), (70, 80), (70, 120), (20, 120)], 'active': True}]
        share = self.app._crack_ignored_share(self.app.cfg.saved_cracks[0]['path'])
        self.assertAlmostEqual(share, 0.5, delta=0.05)

    def test_width_follows_the_band(self):
        mean_w, max_w = self.app._crack_width_px({'path': [(x, 100) for x in range(20, 121)], 'active': True})
        self.assertGreater(mean_w, 1.0)
        self.assertGreaterEqual(max_w, mean_w - 0.5)

    def test_window_is_drawn_only_while_hovering(self):
        self._key(cl.CRACK_LABELS_CODE)
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_crack_label(win_out)
        self.assertFalse(win_out.any(), "nothing under the cursor: no window")
        self._move(220, 100)
        with mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality):
            self.app._draw_crack_label(win_out)
        self.assertTrue(win_out.any())

    def test_new_photo_forgets_the_hover(self):
        self._key(cl.CRACK_LABELS_CODE)
        self._move(220, 100)
        self.app._reset_crack_labels_for_new_image()
        self.assertIsNone(self.app.cfg.crack_label_hover)
        self.assertTrue(self.app.cfg.crack_labels_mode, "the toggle stays on")


class TestNoAutomaticUncertain(_AppCase):

    def test_weak_crack_is_saved_normal_by_default(self):
        self.app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = 101.0   # every crack would fail the old rule
        self.assertFalse(self.app.cfg.CRACK_UNCERTAIN_AUTO)
        self.assertFalse(self.app._crack_quality(self.app.cfg.saved_cracks[0]['path'])[2])
        self.app._save_annotations_now()
        for shape in self._json_cracks():
            self.assertEqual(shape["flags"], {"uncertain": False})
            self.assertFalse(shape["label"].startswith(self.app.cfg.CRACK_UNCERTAIN_LABEL))

    def test_old_behaviour_can_be_switched_back_on(self):
        self.app.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT = 101.0
        self.app.cfg.CRACK_UNCERTAIN_AUTO = True
        self.assertTrue(self.app._crack_quality(self.app.cfg.saved_cracks[0]['path'])[2])


class TestCtrlClickKeepsCracks(_AppCase):

    def test_plain_click_on_a_marker_still_deletes(self):
        self._click(122, 102)
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)

    def test_ctrl_click_on_a_marker_starts_a_trace(self):
        self._click(122, 102, cv2.EVENT_FLAG_CTRLKEY)
        self.assertEqual(len(self.app.cfg.saved_cracks), 2, "no crack deleted")
        self.assertIsNotNone(self.app.cfg.temp_start)

    def test_gui_reports_ctrl_like_opencv(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        import crack_segmentation_gui as gm
        ev = type("Ev", (), {"state": 0x0004})()
        self.assertEqual(gm.EmbeddedCanvas._mouse_flags(ev), cv2.EVENT_FLAG_CTRLKEY)
        ev.state = 0x0005
        self.assertEqual(gm.EmbeddedCanvas._mouse_flags(ev), cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY)
        ev.state = 0x0008   # Command on macOS, Num Lock / Alt elsewhere
        expected = cv2.EVENT_FLAG_CTRLKEY if sys.platform == "darwin" else 0
        self.assertEqual(gm.EmbeddedCanvas._mouse_flags(ev), expected)


class TestWarnUncertain(_AppCase):

    def setUp(self):
        super().setUp()
        self.app.cfg.saved_cracks = []

    def _trace_with(self, quality, answer):
        with mock.patch.object(self.app, "_crack_quality", side_effect=quality), \
             mock.patch.object(self.app, "_confirm_warn_uncertain", return_value=answer) as ask:
            crack = self._trace((20, 100), (120, 100))
        return crack, ask

    def test_off_by_default_and_button_toggles(self):
        self.assertFalse(self.app.cfg.warn_uncertain_mode)
        self._key(wu.WARN_UNCERTAIN_CODE)
        self.assertTrue(self.app.cfg.warn_uncertain_mode)
        self._key(wu.WARN_UNCERTAIN_CODE)
        self.assertFalse(self.app.cfg.warn_uncertain_mode)

    def test_off_never_asks(self):
        crack, ask = self._trace_with(_weak_quality, True)
        ask.assert_not_called()
        self.assertNotIn('uncertain_manual', crack)

    def test_weak_crack_accepted_is_saved_uncertain_right_away(self):
        self._key(wu.WARN_UNCERTAIN_CODE)
        crack, ask = self._trace_with(_weak_quality, True)
        ask.assert_called_once()
        reasons = ask.call_args[0][1]
        self.assertTrue(any("reliability" in r for r in reasons) and any("confidence" in r for r in reasons))
        self.assertTrue(crack.get('uncertain_manual'))
        shapes = self._json_cracks()
        self.assertEqual(shapes[0]["flags"], {"uncertain": True, "uncertain_manual": True})

    def test_weak_crack_refused_stays_normal(self):
        self._key(wu.WARN_UNCERTAIN_CODE)
        crack, ask = self._trace_with(_weak_quality, False)
        ask.assert_called_once()
        self.assertNotIn('uncertain_manual', crack)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH), "no save before [S], as before")

    def test_good_crack_shows_no_window(self):
        self._key(wu.WARN_UNCERTAIN_CODE)
        crack, ask = self._trace_with(_good_quality, True)
        ask.assert_not_called()
        self.assertNotIn('uncertain_manual', crack)

    def test_already_uncertain_mode_is_not_asked_again(self):
        self._key(wu.WARN_UNCERTAIN_CODE)
        self.app.cfg.uncertain_trace_mode = True
        crack, ask = self._trace_with(_weak_quality, False)
        ask.assert_not_called()
        self.assertTrue(crack.get('uncertain_manual'))

    def test_limits_can_be_disabled(self):
        self.app.cfg.WARN_UNCERTAIN_RELIABILITY_PCT = None
        self.app.cfg.WARN_UNCERTAIN_CONFIDENCE = None
        self.assertEqual(self.app._warn_uncertain_reasons(5.0, 0.1), [])
        self.assertEqual(self.app._warn_uncertain_reasons(None, None), [])


class TestModeLinesInStatusBar(unittest.TestCase):
    """In the GUI, the mode lines (WARN UNCERTAIN, CRACK LABELS, UNCERTAIN MODE ...) go to the status bar, not the photo."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        import crack_segmentation_gui as gm
        self.gm = gm

    def test_gui_collects_instead_of_drawing(self):
        app = object.__new__(self.gm.GuiCrackSegmentation)
        app._status_bar = mock.Mock()
        app._canvas = mock.Mock()
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._put_hud_text(win_out, "WARN UNCERTAIN: on", (15, 200), (0, 140, 255))
        app._put_hud_text(win_out, "CRACK LABELS: on", (15, 230), (0, 230, 255))
        self.assertFalse(win_out.any(), "nothing drawn on the photo")
        app._display_frame("w", win_out)
        app._status_bar.set_modes.assert_called_once_with(["WARN UNCERTAIN: on", "CRACK LABELS: on"])
        app._display_frame("w", win_out)
        app._status_bar.set_modes.assert_called_with([])

    def test_headless_still_draws_on_the_photo(self):
        app = _two_crack_app()
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._put_hud_text(win_out, "WARN UNCERTAIN: on", (15, 200), (0, 140, 255))
        self.assertTrue(win_out.any())

    def test_status_bar_shows_and_clears_the_modes(self):
        bar = self.gm.StatusBar(self.gm.tk.Tk())
        bar.set_modes(["A", "B", "A"])
        self.assertEqual(bar.mode_label.cget("text"), "A\nB")
        bar.set_modes([])
        self.assertEqual(bar.mode_label.cget("text"), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
