"""test_v130_relabel.py

Headless tests for the 1.3.0 Make uncertain / Make normal commands (crack_relabel.py):
  - Ctrl+C (Cmd+C on macOS), Ctrl/Cmd+click on the sidebar [Crack tool] button or the Tools menu: the next clicks
    on a blue crack (line or number) make it uncertain (red);
  - Ctrl+U (Cmd+U), Ctrl/Cmd+click on [Uncertain crack] or the Tools menu: a red crack becomes normal (blue);
  - JSON and masks are written after the change, [U] undoes it; one crack per press, so the crack can then be
    deleted by its marker as usual;
  - other Ctrl+letters no longer fire unrelated commands through their control characters;
  - cracks are never marked uncertain automatically (CRACK_UNCERTAIN_AUTO off).

Run with:  python3 -m unittest test_v130_relabel -v
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported

import crack_relabel as cr
from test_v105_link import _two_crack_app


def _good_quality(path):
    return 90.0, 0.9, False


def _key(keysym, char="", state=0):
    return types.SimpleNamespace(keysym=keysym, char=char, state=state)


class _AppCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v130_relabel_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)   # crack 1: x 20..120, crack 2: x 160..280, both on y=100
        self.app.cfg.current_tool = 'crack'
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _press(self, code):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(code)

    def _click_real(self, x, y):
        wx, wy = self.app.transform_real_to_window_coords(x, y)
        self._click_window(wx, wy)

    def _click_window(self, wx, wy):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)

    def _json_flags(self):
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]
        return [(s["label"], s["flags"]) for s in shapes]

    def _mask(self, suffix):
        return cv2.imread(os.path.join(self.tmpdir, "Binary files", f"wall{suffix}.png"), cv2.IMREAD_GRAYSCALE)


class TestMode(_AppCase):

    def test_codes_are_never_typed_characters(self):
        for code in (cr.MARK_UNCERTAIN_CODE, cr.MARK_NORMAL_CODE):
            self.assertTrue(128 <= code < 160, "C1 control range: no key, accent or symbol types it")

    def test_off_by_default_toggles_and_switches(self):
        self.assertIsNone(self.app.cfg.crack_relabel_mode)
        self._press(cr.MARK_UNCERTAIN_CODE)
        self.assertEqual(self.app.cfg.crack_relabel_mode, 'uncertain')
        self._press(cr.MARK_NORMAL_CODE)
        self.assertEqual(self.app.cfg.crack_relabel_mode, 'normal')
        self._press(cr.MARK_NORMAL_CODE)
        self.assertIsNone(self.app.cfg.crack_relabel_mode)

    def test_mode_line_is_shown(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        texts = []
        with mock.patch.object(self.app, "_put_hud_text", side_effect=lambda w, t, *a, **k: texts.append(t)):
            self.app._draw_crack_relabel_label(np.zeros((900, 1200, 3), dtype=np.uint8))
        self.assertTrue(texts and texts[0].startswith("MAKE UNCERTAIN"))


class TestRelabel(_AppCase):

    def test_click_on_a_blue_line_makes_it_uncertain_and_saves(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._click_real(70, 100)
        first, second = self.app.cfg.saved_cracks
        self.assertTrue(first.get('uncertain_manual'))
        self.assertNotIn('uncertain_manual', second)
        (label_a, flags_a), (label_b, flags_b) = self._json_flags()
        self.assertTrue(label_a.startswith(self.app.cfg.CRACK_UNCERTAIN_LABEL))
        self.assertEqual(flags_a, {"uncertain": True, "uncertain_manual": True})
        self.assertTrue(label_b.startswith("crack_") and not label_b.startswith(self.app.cfg.CRACK_UNCERTAIN_LABEL))
        self.assertEqual(flags_b, {"uncertain": False})
        unc = self._mask("-crack_uncertain_mask")
        self.assertEqual((unc[100, 70], unc[100, 220]), (255, 0))

    def test_click_on_the_number_works_too(self):
        self.app.cfg.show_markers = True
        self._press(cr.MARK_UNCERTAIN_CODE)
        wx, wy = self.app.transform_real_to_window_coords(160, 100)   # start marker (number) of crack 2
        self._click_window(wx + 8, wy - 8)   # inside the number square, off the line
        self.assertTrue(self.app.cfg.saved_cracks[1].get('uncertain_manual'))

    def test_one_crack_per_press(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._click_real(70, 100)
        self.assertIsNone(self.app.cfg.crack_relabel_mode, "the click on a crack ends the mode")
        self._click_real(220, 100)
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[1], "the next click is an ordinary click")

    def test_relabelled_crack_can_be_deleted_by_its_marker(self):
        # USER REPORT: after making a crack uncertain, clicking its marker did not delete it (mode still on).
        for start_mode, code in ((False, cr.MARK_UNCERTAIN_CODE), (True, cr.MARK_NORMAL_CODE)):
            app = _two_crack_app(tempfile.mkdtemp(dir=self.tmpdir))
            app.cfg.current_tool = 'crack'
            if start_mode:
                app.cfg.saved_cracks[0]['uncertain_manual'] = True
            with mock.patch.object(app, "_crack_quality", side_effect=_good_quality), \
                    mock.patch.object(app, "refresh_zoom_viewport"):
                app.process_keypress(code)
                for x in (70, 20):   # relabel on the line, then click the start marker
                    wx, wy = app.transform_real_to_window_coords(x, 100)
                    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)
            self.assertEqual(len(app.cfg.saved_cracks), 1, "the relabelled crack was deleted")
            self.assertEqual(app.cfg.saved_cracks[0]['start'], (160, 100))

    def test_choosing_cut_cancels_and_the_click_cuts(self):
        # USER REPORT: Cmd+C, then Cut, then a click on the crack still made it red instead of cutting.
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._press(ord('1'))
        self.assertIsNone(self.app.cfg.crack_relabel_mode)
        self.assertEqual(self.app.cfg.crack_cut_join_state["mode"], 'cut')
        self._click_real(50, 100)
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[0])
        self.assertEqual(self.app.cfg.crack_cut_join_state["first"][0], 0, "the click went to Cut")

    def test_other_commands_cancel_view_keys_do_not(self):
        for key in (ord('2'), ord('3'), ord('c'), ord('d'), 21, 7):
            self.app.cfg.crack_relabel_mode = 'uncertain'
            self.app._cancel_crack_relabel_for_key(key)
            self.assertIsNone(self.app.cfg.crack_relabel_mode, key)
        for key in (0, 3, ord('+'), ord('-'), ord('0'), ord(' '), ord('n')):
            self.app.cfg.crack_relabel_mode = 'normal'
            self.app._cancel_crack_relabel_for_key(key)
            self.assertEqual(self.app.cfg.crack_relabel_mode, 'normal', key)

    def test_click_away_keeps_waiting(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._click_real(150, 30)
        self.assertEqual(self.app.cfg.crack_relabel_mode, 'uncertain')
        self._click_real(70, 100)
        self.assertTrue(self.app.cfg.saved_cracks[0].get('uncertain_manual'))

    def test_click_on_a_red_line_makes_it_normal_again(self):
        self.app.cfg.saved_cracks[0]['uncertain_manual'] = True
        self._press(cr.MARK_NORMAL_CODE)
        self._click_real(70, 100)
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[0])
        self.assertEqual([flags for _, flags in self._json_flags()], [{"uncertain": False}] * 2)
        self.assertIsNone(self._mask("-crack_uncertain_mask"), "no uncertain crack left: mask removed")

    def test_already_in_that_class_changes_nothing(self):
        self._press(cr.MARK_NORMAL_CODE)
        self._click_real(70, 100)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH))
        self.assertEqual(self.app.cfg.action_history, [])

    def test_click_away_from_cracks_does_nothing_and_does_not_trace(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._click_real(150, 30)
        self.assertIsNone(self.app.cfg.temp_start, "no trace started while the mode is on")
        self.assertFalse(any(c.get('uncertain_manual') for c in self.app.cfg.saved_cracks))

    def test_undo_restores_the_previous_class(self):
        self._press(cr.MARK_UNCERTAIN_CODE)
        self._click_real(70, 100)
        self._press(ord('u'))
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[0])

    def test_hidden_crack_is_not_picked(self):
        self.app.cfg.crack_filter = {2}
        with mock.patch.object(self.app, "_crack_shown", side_effect=lambda f: f is self.app.cfg.saved_cracks[1]):
            self._press(cr.MARK_UNCERTAIN_CODE)
            self._click_real(70, 100)
        self.assertNotIn('uncertain_manual', self.app.cfg.saved_cracks[0])

    def test_automatic_rule_stays_off(self):
        self.assertFalse(self.app.cfg.CRACK_UNCERTAIN_AUTO)


class TestGuiShortcuts(unittest.TestCase):

    def setUp(self):
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_ctrl_c_and_ctrl_u(self):
        self.assertEqual(self.gm.tk_key_event_to_code(_key("c", "\x03", 0x4)), cr.MARK_UNCERTAIN_CODE)
        self.assertEqual(self.gm.tk_key_event_to_code(_key("u", "\x15", 0x4)), cr.MARK_NORMAL_CODE)
        self.assertEqual(self.gm.tk_key_event_to_code(_key("C", "\x03", 0x5)), cr.MARK_UNCERTAIN_CODE, "with Shift too")

    def test_cmd_on_macos(self):
        with mock.patch.object(self.gm.sys, "platform", "darwin"):
            self.assertEqual(self.gm.tk_key_event_to_code(_key("c", "c", 0x8)), cr.MARK_UNCERTAIN_CODE)
            self.assertEqual(self.gm.tk_key_event_to_code(_key("u", "u", 0x8)), cr.MARK_NORMAL_CODE)

    def test_plain_keys_unchanged(self):
        self.assertEqual(self.gm.tk_key_event_to_code(_key("c", "c")), ord('c'))
        self.assertEqual(self.gm.tk_key_event_to_code(_key("u", "u")), ord('u'))

    def test_other_ctrl_letters_do_not_fire_unrelated_commands(self):
        self.assertIsNone(self.gm.tk_key_event_to_code(_key("n", "\x0e", 0x4)), "Ctrl+N would be 14 = HWAV")
        self.assertIsNone(self.gm.tk_key_event_to_code(_key("s", "\x13", 0x4)), "Ctrl+S would be 19 = filter")

    def test_keypress_queues_the_code(self):
        self.app._pending_keys.clear()
        self.app._on_keypress(_key("c", "\x03", 0x4))
        self.assertEqual(self.app._pending_keys, [cr.MARK_UNCERTAIN_CODE])

    def test_ctrl_click_on_sidebar_buttons(self):
        buttons = self.app._sidebar.buttons_by_code
        for code, expected in ((ord('c'), cr.MARK_UNCERTAIN_CODE), (7, cr.MARK_NORMAL_CODE)):
            self.app._pending_keys.clear()
            result = buttons[code].fire("<Control-Button-1>")
            self.assertEqual(result, "break", "the button's own command must not run")
            self.assertEqual(self.app._pending_keys, [expected])

    def test_plain_click_on_sidebar_buttons_unchanged(self):
        buttons = self.app._sidebar.buttons_by_code
        self.app._pending_keys.clear()
        buttons[ord('c')].command()
        buttons[7].command()
        self.assertEqual(self.app._pending_keys, [ord('c'), 7])

    def test_tools_menu_entries(self):
        codes = [code for _, code in self.gm._TOOL_COMMANDS]
        self.assertIn(cr.MARK_UNCERTAIN_CODE, codes)
        self.assertIn(cr.MARK_NORMAL_CODE, codes)


if __name__ == "__main__":
    unittest.main()
