"""test_v140_features.py

Headless tests for the 1.4.0 features:
  - [Edit extremes] (crack_extremes.py, code 132, key [4]): crack ends become large red dots without numbers; a dot
    clicked (or dragged) hides the other dots, and a click elsewhere lengthens the crack along the edge, a click on
    the crack shortens it; [U] undoes it; pressing the button / [4] again (or choosing another command) writes JSON
    and masks; clicks off the dots trace new cracks; Cmd+C / Cmd+U and Zoom window work with the mode on;
  - [Hide uncertain overlay] / [Hide normal overlay] (crack_class_view.py, codes 130 / 131): hide the fill of one
    class only, [N] still hides both.

Run with:  python3 -m unittest test_v140_features -v
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported

import crack_segmentation_gui as gm
from crack_extremes import CRACK_EXTREMES_CODE, CRACK_EXTREMES_KEY
from crack_class_view import HIDE_UNCERTAIN_OVERLAY_CODE, HIDE_NORMAL_OVERLAY_CODE
from crack_relabel import MARK_UNCERTAIN_CODE, MARK_NORMAL_CODE
from test_v104_fixes import _max_step
from test_v105_link import _two_crack_app


def _mouse(app, event, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.mouse_callback(event, wx, wy, 0, None)


def _click(app, x, y):
    _mouse(app, cv2.EVENT_LBUTTONDOWN, x, y)


def _press(app, code):
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.process_keypress(code)


def _extremes_app(tmpdir=None):
    app = _two_crack_app(tmpdir)
    _press(app, CRACK_EXTREMES_CODE)
    return app


def _red_at(app, win, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    return tuple(win[wy, wx]) == (0, 0, 255)


class TestEditExtremesMode(unittest.TestCase):

    def test_button_toggles_the_mode(self):
        app = _extremes_app()
        self.assertTrue(app.cfg.crack_extremes_state["active"])
        with mock.patch.object(app, "_save_annotations_now") as save:
            _press(app, CRACK_EXTREMES_CODE)
        self.assertFalse(app.cfg.crack_extremes_state["active"])
        save.assert_not_called()   # nothing changed: nothing to write

    def test_key_4_toggles_the_mode(self):
        app = _two_crack_app()
        _press(app, CRACK_EXTREMES_KEY)
        self.assertTrue(app.cfg.crack_extremes_state["active"])
        _press(app, CRACK_EXTREMES_KEY)
        self.assertFalse(app.cfg.crack_extremes_state["active"])

    def test_key_4_and_button_mix(self):
        app = _extremes_app()
        _press(app, CRACK_EXTREMES_KEY)
        self.assertFalse(app.cfg.crack_extremes_state["active"])

    def test_numbers_hidden_and_red_dots_drawn_while_on(self):
        app = _extremes_app()
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        with mock.patch.object(app, "_draw_one_crack_marker") as numbered:
            app._draw_crack_markers(win, 22)
        numbered.assert_not_called()
        app._draw_crack_extremes(win)
        self.assertTrue(_red_at(app, win, 120, 100), "the end of crack 1 is a red dot")

    def test_picking_a_dot_hides_the_others(self):
        app = _extremes_app()
        _click(app, 120, 100)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_crack_extremes(win)
        for x, shown in ((120, True), (20, False), (160, False), (280, False)):
            self.assertEqual(_red_at(app, win, x, 100), shown, x)

    def test_releasing_the_dot_shows_them_all_again(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 120, 100)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_crack_extremes(win)
        for x in (20, 120, 160, 280):
            self.assertTrue(_red_at(app, win, x, 100), x)

    def test_all_dots_back_after_lengthening(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 145, 100)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_crack_extremes(win)
        for x in (20, 145, 160, 280):
            self.assertTrue(_red_at(app, win, x, 100), x)

    def test_numbers_come_back_when_off(self):
        app = _extremes_app()
        _press(app, CRACK_EXTREMES_CODE)
        with mock.patch.object(app, "_draw_one_crack_marker") as numbered:
            app._draw_crack_markers(np.zeros((900, 1200, 3), dtype=np.uint8), 22)
        self.assertEqual(numbered.call_count, 2)

    def test_click_away_from_dots_starts_a_trace(self):
        app = _extremes_app()
        _click(app, 60, 150)
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])
        self.assertEqual(app.cfg.temp_start, app.transform_window_to_real_coords(
            *app.transform_real_to_window_coords(60, 150)))

    def test_new_crack_traced_while_on(self):
        app = _extremes_app()
        _click(app, 200, 100)
        _mouse(app, cv2.EVENT_MOUSEMOVE, 230, 100)
        _click(app, 250, 100)
        self.assertEqual(len(app.cfg.saved_cracks), 3)
        self.assertIsNone(app.cfg.temp_start)
        self.assertTrue(app.cfg.crack_extremes_state["active"], "the mode stays on after the trace")

    def test_trace_end_on_a_dot_completes_the_trace(self):
        app = _extremes_app()
        _click(app, 140, 100)
        _click(app, 120, 100)   # crack 1's end dot: ends the trace, does not pick the dot
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])
        self.assertIsNone(app.cfg.temp_start)

    def test_click_near_a_marker_does_not_delete_the_crack(self):
        app = _extremes_app()
        _click(app, 120, 112)   # off the dot, but within the delete-by-marker distance
        self.assertEqual(len(app.cfg.saved_cracks), 2)


class TestLengthenAndShorten(unittest.TestCase):

    def test_end_lengthened_along_the_edge(self):
        app = _extremes_app()
        _click(app, 120, 100)
        self.assertEqual(app.cfg.crack_extremes_state["grab"], (0, 'end'))
        _click(app, 145, 100)
        crack = app.cfg.saved_cracks[0]
        self.assertEqual(crack['path'][0], (20, 100))
        self.assertEqual(crack['path'][-1], (145, 100))
        self.assertEqual(crack['end'], (145, 100))
        self.assertEqual(_max_step(crack['path']), 1)
        self.assertEqual(crack['width_segments'], [{'i0': 0, 'i1': 10, 'width_px': 3}])
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])
        self.assertTrue(app.cfg.crack_extremes_state["changed"])

    def test_drag_then_release_waits_for_the_click(self):
        app = _extremes_app()
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 120, 100)
        _mouse(app, cv2.EVENT_MOUSEMOVE, 135, 100)
        _mouse(app, cv2.EVENT_LBUTTONUP, 135, 100)
        self.assertEqual(len(app.cfg.saved_cracks[0]['path']), 101, "releasing changes nothing yet")
        _click(app, 145, 100)
        self.assertEqual(app.cfg.saved_cracks[0]['path'][-1], (145, 100))

    def test_start_lengthened_and_width_tracts_follow(self):
        app = _extremes_app()
        before = app.cfg.saved_cracks[1]
        tract_pts = (before['path'][100], before['path'][110])
        _click(app, 160, 100)
        _click(app, 140, 100)
        crack = app.cfg.saved_cracks[1]
        self.assertEqual(crack['path'][0], (140, 100))
        self.assertEqual(crack['start'], (140, 100))
        self.assertEqual(crack['path'][-1], (280, 100))
        self.assertEqual(_max_step(crack['path']), 1)
        seg = crack['width_segments'][0]
        self.assertEqual((crack['path'][seg['i0']], crack['path'][seg['i1']]), tract_pts)

    def test_click_on_the_crack_shortens_it(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 80, 100)
        crack = app.cfg.saved_cracks[0]
        self.assertEqual(crack['path'][0], (20, 100))
        self.assertEqual(crack['path'][-1], (80, 100))
        self.assertEqual(crack['width_segments'], [{'i0': 0, 'i1': 10, 'width_px': 3}])

    def test_shortening_start_drops_the_tract_it_cuts(self):
        app = _extremes_app()
        _click(app, 20, 100)
        _click(app, 25, 100)
        crack = app.cfg.saved_cracks[0]
        self.assertEqual(crack['path'][0], (25, 100))
        self.assertNotIn('width_segments', crack)

    def test_same_dot_again_releases_it(self):
        app = _extremes_app()
        path = list(app.cfg.saved_cracks[0]['path'])
        _click(app, 120, 100)
        _click(app, 120, 100)
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])
        self.assertEqual(app.cfg.saved_cracks[0]['path'], path)

    def test_undo_restores_the_crack(self):
        app = _extremes_app()
        path = list(app.cfg.saved_cracks[0]['path'])
        _click(app, 120, 100)
        _click(app, 145, 100)
        _press(app, ord('u'))
        self.assertTrue(app.cfg.crack_extremes_state["active"], "Undo keeps the mode on")
        self.assertEqual(app.cfg.saved_cracks[0]['path'], path)

    def test_no_route_keeps_the_dot_picked(self):
        app = _extremes_app()
        _click(app, 120, 100)
        with mock.patch.object(app, "a_star_pathfinding", return_value=[]):
            _click(app, 140, 180)
        self.assertEqual(app.cfg.crack_extremes_state["grab"], (0, 'end'))
        self.assertEqual(len(app.cfg.saved_cracks[0]['path']), 101)


class TestEditExtremesSaving(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crackseg_v140_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_second_press_writes_json_and_masks(self):
        app = _extremes_app(self.tmp)
        _click(app, 120, 100)
        _click(app, 145, 100)
        _press(app, CRACK_EXTREMES_CODE)
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            shapes = [s for s in json.load(fh)["shapes"] if s["shape_type"] == "linestrip"]
        ends = {tuple(map(int, s["points"][-1])) for s in shapes}
        self.assertIn((145, 100), ends)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "Binary files", "wall-crack_mask.png")))

    def test_key_4_saves_too(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 145, 100)
        with mock.patch.object(app, "_save_annotations_now") as save:
            _press(app, CRACK_EXTREMES_KEY)
        self.assertFalse(app.cfg.crack_extremes_state["active"])
        save.assert_called_once()

    def test_another_command_closes_and_saves(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 145, 100)
        with mock.patch.object(app, "_save_annotations_now") as save:
            _press(app, ord('1'))
        self.assertFalse(app.cfg.crack_extremes_state["active"])
        save.assert_called_once()

    def test_leaving_without_saving_does_not_write(self):
        app = _extremes_app()
        _click(app, 120, 100)
        _click(app, 145, 100)
        with mock.patch.object(app, "_save_annotations_now") as save, \
                mock.patch.object(app, "_handle_navigate_without_saving", return_value=True):
            _press(app, 4)
        self.assertFalse(app.cfg.crack_extremes_state["active"])
        save.assert_not_called()

    def test_view_keys_keep_the_mode(self):
        app = _extremes_app()
        _press(app, ord('n'))
        _press(app, HIDE_UNCERTAIN_OVERLAY_CODE)
        self.assertTrue(app.cfg.crack_extremes_state["active"])


class TestEditExtremesWithRelabel(unittest.TestCase):
    """USER REPORT: Cmd+C during Edit extremes closed the mode, so the next click on a crack end hit the
    numbered marker of the normal Crack tool and deleted the crack."""

    def test_cmd_c_keeps_the_mode_and_relabels(self):
        app = _extremes_app()
        _press(app, MARK_UNCERTAIN_CODE)
        self.assertTrue(app.cfg.crack_extremes_state["active"])
        _click(app, 60, 100)
        self.assertTrue(app.cfg.saved_cracks[0].get('uncertain_manual'))
        self.assertTrue(app.cfg.crack_extremes_state["active"])

    def test_click_on_an_end_after_cmd_c_never_deletes(self):
        app = _extremes_app()
        _press(app, MARK_UNCERTAIN_CODE)
        _click(app, 60, 100)
        _click(app, 120, 100)   # red dot: picked, not deleted
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertEqual(app.cfg.crack_extremes_state["grab"], (0, 'end'))

    def test_cmd_u_keeps_the_mode(self):
        app = _extremes_app()
        app.cfg.saved_cracks[0]['uncertain_manual'] = True
        _press(app, MARK_NORMAL_CODE)
        _click(app, 60, 100)
        self.assertFalse(app.cfg.saved_cracks[0].get('uncertain_manual'))
        self.assertTrue(app.cfg.crack_extremes_state["active"])


class TestEditExtremesWithZoomWindow(unittest.TestCase):
    """USER REPORT: with Zoom window on, pressing a red dot and dragging it drew a zoom rectangle instead."""

    def _app(self):
        app = _extremes_app()
        app.cfg.zoom_window_state["active"] = True
        return app

    def test_drag_a_dot_and_click_the_target(self):
        app = self._app()
        box = list(app.cfg.zoom_box)
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 120, 100)
        _mouse(app, cv2.EVENT_MOUSEMOVE, 140, 100)
        _mouse(app, cv2.EVENT_LBUTTONUP, 140, 100)
        self.assertEqual(list(app.cfg.zoom_box), box, "no zoom")
        self.assertEqual(app.cfg.crack_extremes_state["grab"], (0, 'end'))
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 145, 100)
        _mouse(app, cv2.EVENT_LBUTTONUP, 145, 100)
        self.assertEqual(app.cfg.saved_cracks[0]['path'][-1], (145, 100))
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])

    def test_plain_click_on_a_dot_then_target(self):
        app = self._app()
        for x in (120, 145):
            _mouse(app, cv2.EVENT_LBUTTONDOWN, x, 100)
            _mouse(app, cv2.EVENT_LBUTTONUP, x, 100)
        self.assertEqual(app.cfg.saved_cracks[0]['path'][-1], (145, 100))

    def test_drag_off_the_dots_still_zooms(self):
        app = self._app()
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 40, 30)
        _mouse(app, cv2.EVENT_MOUSEMOVE, 200, 170)
        _mouse(app, cv2.EVENT_LBUTTONUP, 200, 170)
        self.assertGreater(app.cfg.zoom_factor, 1.0)
        self.assertIsNone(app.cfg.crack_extremes_state["grab"])

    def test_dot_works_inside_the_zoomed_view(self):
        app = self._app()
        app.cfg.zoom_box = [80, 60, 200, 150]
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 120, 100)
        _mouse(app, cv2.EVENT_LBUTTONUP, 120, 100)
        _mouse(app, cv2.EVENT_LBUTTONDOWN, 145, 100)
        _mouse(app, cv2.EVENT_LBUTTONUP, 145, 100)
        self.assertEqual(app.cfg.saved_cracks[0]['path'][-1], (145, 100))


class TestCrackClassView(unittest.TestCase):

    def _app(self):
        app = _two_crack_app()
        app.cfg.saved_cracks[1]['uncertain_manual'] = True   # crack 2 red
        app.recalculate_masks()
        return app

    def test_both_shown_by_default(self):
        app = self._app()
        mask = app._display_blue_mask()
        self.assertEqual(mask[100, 60], 255)
        self.assertEqual(mask[100, 220], 255)

    def test_hide_uncertain_keeps_the_normal_fill(self):
        app = self._app()
        _press(app, HIDE_UNCERTAIN_OVERLAY_CODE)
        mask = app._display_blue_mask()
        self.assertEqual(mask[100, 60], 255)
        self.assertEqual(mask[100, 220], 0)
        scene = np.zeros((200, 300, 3), dtype=np.uint8)
        app._paint_red_uncertain_overlay(scene, np.ones((3, 3), np.uint8), 200, 300)
        self.assertFalse(np.any(scene), "no red fill while uncertain cracks are hidden")

    def test_hide_normal_keeps_the_uncertain_fill(self):
        app = self._app()
        _press(app, HIDE_NORMAL_OVERLAY_CODE)
        mask = app._display_blue_mask()
        self.assertEqual(mask[100, 60], 0)
        self.assertEqual(mask[100, 220], 255)

    def test_pressing_again_shows_them_back(self):
        app = self._app()
        _press(app, HIDE_NORMAL_OVERLAY_CODE)
        _press(app, HIDE_NORMAL_OVERLAY_CODE)
        self.assertIs(app._display_blue_mask(), app.cfg.blue_visual_mask)

    def test_n_still_hides_both(self):
        app = self._app()
        _press(app, ord('n'))
        self.assertFalse(app.cfg.show_crack_overlay)
        self.assertTrue(app.cfg.show_uncertain_overlay and app.cfg.show_normal_overlay)

    def test_exported_masks_unaffected(self):
        app = self._app()
        _press(app, HIDE_UNCERTAIN_OVERLAY_CODE)
        _press(app, HIDE_NORMAL_OVERLAY_CODE)
        self.assertEqual(app.cfg.blue_visual_mask[100, 60], 255)
        self.assertEqual(app.cfg.blue_visual_mask[100, 220], 255)


class TestCrackNumberToggle(unittest.TestCase):
    """v1.4.0: with the Crack tool, a click ON a crack line hides / shows its markers and number."""

    def _app(self):
        app = _two_crack_app()
        app.cfg.current_tool = 'crack'
        return app

    def _numbers_drawn(self, app):
        with mock.patch.object(app, "_draw_crack_number_in_marker") as num:
            app._draw_crack_markers(np.zeros((900, 1200, 3), dtype=np.uint8), 22)
        return [c.args[4] for c in num.call_args_list]

    def test_click_on_the_crack_hides_its_number(self):
        app = self._app()
        _click(app, 70, 100)
        self.assertTrue(app.cfg.saved_cracks[0].get('number_hidden'))
        self.assertIsNone(app.cfg.temp_start, "no trace started")
        self.assertEqual(self._numbers_drawn(app), [2])

    def test_second_click_shows_it_again(self):
        app = self._app()
        _click(app, 70, 100)
        _click(app, 75, 100)
        self.assertFalse(app.cfg.saved_cracks[0].get('number_hidden'))
        self.assertEqual(self._numbers_drawn(app), [1, 2])

    def test_whole_marker_disappears(self):
        app = self._app()
        _click(app, 70, 100)
        with mock.patch.object(app, "_draw_crack_end_dot") as dot:
            win = np.zeros((900, 1200, 3), dtype=np.uint8)
            app._draw_crack_markers(win, 22)
        self.assertEqual(dot.call_count, 1, "only crack 2's end dot")
        for x in (20, 120):
            wx, wy = app.transform_real_to_window_coords(x, 100)
            self.assertFalse(np.any(win[wy - 23:wy + 24, wx - 23:wx + 24]), f"nothing drawn around x={x}")

    def test_hidden_marker_cannot_delete_the_crack(self):
        app = self._app()
        _click(app, 70, 100)
        _click(app, 122, 102)   # where crack 1's end marker was: off the line, starts a trace
        self.assertEqual(len(app.cfg.saved_cracks), 2)

    def test_click_where_the_start_marker_was_shows_it_again(self):
        app = self._app()
        _click(app, 70, 100)
        _click(app, 20, 100)
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertFalse(app.cfg.saved_cracks[0].get('number_hidden'))

    def test_other_cracks_keep_their_numbers(self):
        app = self._app()
        _click(app, 220, 100)
        self.assertEqual(self._numbers_drawn(app), [1])

    def test_click_off_the_cracks_still_traces(self):
        app = self._app()
        _click(app, 70, 160)
        self.assertIsNotNone(app.cfg.temp_start)

    def test_shift_click_on_the_crack_starts_a_trace(self):
        app = self._app()
        wx, wy = app.transform_real_to_window_coords(70, 100)
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, cv2.EVENT_FLAG_SHIFTKEY, None)
        self.assertIsNotNone(app.cfg.temp_start)
        self.assertFalse(app.cfg.saved_cracks[0].get('number_hidden'))

    def test_click_on_the_crack_ends_a_trace_in_progress(self):
        app = self._app()
        wx, wy = app.transform_real_to_window_coords(250, 100)
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, cv2.EVENT_FLAG_SHIFTKEY, None)
        _mouse(app, cv2.EVENT_MOUSEMOVE, 100, 100)
        _click(app, 70, 100)   # on crack 1: completes the trace, does not toggle its number
        self.assertIsNone(app.cfg.temp_start)
        self.assertEqual(len(app.cfg.saved_cracks), 3)
        self.assertFalse(app.cfg.saved_cracks[0].get('number_hidden'))

    def test_plain_click_on_a_marker_still_deletes(self):
        app = self._app()
        _click(app, 122, 102)
        self.assertEqual(len(app.cfg.saved_cracks), 1)

    def test_hidden_number_survives_an_extremes_edit(self):
        app = self._app()
        _click(app, 70, 100)
        _press(app, CRACK_EXTREMES_CODE)
        _click(app, 120, 100)
        _click(app, 145, 100)
        _press(app, CRACK_EXTREMES_CODE)
        self.assertTrue(app.cfg.saved_cracks[0].get('number_hidden'))

    def test_never_exported(self):
        tmp = tempfile.mkdtemp(prefix="crackseg_v140_num_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        app = _two_crack_app(tmp)
        app.cfg.current_tool = 'crack'
        _click(app, 70, 100)
        app._save_annotations_now("TEST", "the change")
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            self.assertNotIn("number_hidden", fh.read())


class TestGuiEntries(unittest.TestCase):

    def test_new_commands_in_menu_and_sidebar(self):
        menu_codes = [code for _, commands in gm._ALL_MENU_GROUPS for _, code in commands]
        sidebar_codes = [code for *_, code in gm._SIDEBAR_SHORTCUTS]
        for code in (CRACK_EXTREMES_CODE, HIDE_UNCERTAIN_OVERLAY_CODE, HIDE_NORMAL_OVERLAY_CODE):
            self.assertIn(code, menu_codes)
            self.assertIn(code, sidebar_codes)
            self.assertIn(code, gm._SIDEBAR_TOGGLE_STATES)

    def test_key_4_shown_on_the_sidebar_button(self):
        entry = [e for e in gm._SIDEBAR_SHORTCUTS if e[-1] == CRACK_EXTREMES_CODE][0]
        self.assertEqual(entry[2], "4")

    def test_typed_4_reaches_the_toggle(self):
        event = mock.Mock(keysym="4", char="4", state=0)
        self.assertEqual(gm.tk_key_event_to_code(event), CRACK_EXTREMES_KEY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
