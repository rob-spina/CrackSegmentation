"""test_v107_features.py

Headless tests for 1.0.7:
  - LINK [3] also accepts a click anywhere ON a crack line, not only at its ends. Inside a crack, its
    longer part joins the merged crack and the shorter leftover stays a separate crack (nothing is lost).
  - Alternative route (sidebar, code 18): other edge routes between the ends of the crack traced last,
    one per press, each in its own color; the arrow turns left when none is left; [$] applies one.
  - Show cracks by number (sidebar, code 19): type e.g. 3;7;12 to show only those cracks (display only).
  - Crack tracing: Shift+click between the start and the end adds up to two intermediate points the
    route must pass through (a plain start-end trace is unchanged).
  - Import [W]/[L] between very different viewpoints: the target is re-seen in memory from the source viewpoint
    and matched again (perspective_match.py); cracks leaving the target view are cut, not squashed on its border.
  - Building portion: turning it on tints orange the part of the photo shared with the other photos of the group
    (compatible_area.py) and pre-sets the window around it.
  - Building portion (sidebar, code 20): drag a window over the part of the building shared with the other
    photos of the group; [W]/[L] then align on it and import only inside it.

Run with:  python3 -m unittest test_v107_features -v
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

from test_gui_shell_parity import make_app
from test_v104_fixes import _max_step

H, W = 200, 300


def _t_junction_app(tmpdir=None):
    """Crack 1 along y=100 (x 20..280); crack 2 going down x=150 (y 20..80), ending 20 px above crack 1."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    skel = np.zeros((H, W), dtype=np.uint8)
    cv2.line(skel, (20, 100), (280, 100), 255, 1)
    cv2.line(skel, (150, 20), (150, 100), 255, 1)
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    a = [(x, 100) for x in range(20, 281)]
    b = [(150, y) for y in range(20, 81)]
    app.cfg.saved_cracks = [
        {'start': a[0], 'end': a[-1], 'path': list(a), 'active': True, 'session_id': 'current',
         'width_segments': [{'i0': 0, 'i1': 10, 'width_px': 3}, {'i0': 200, 'i1': 210, 'width_px': 5},
                            {'i0': 120, 'i1': 140, 'width_px': 7}]},
        {'start': b[0], 'end': b[-1], 'path': list(b), 'active': True, 'session_id': 'current'},
    ]
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    if tmpdir:
        img = np.full((H, W, 3), 200, dtype=np.uint8)
        cv2.line(img, (20, 100), (280, 100), (30, 30, 30), 2)
        cv2.line(img, (150, 20), (150, 100), (30, 30, 30), 2)
        images = os.path.join(tmpdir, "Images")
        os.makedirs(images)
        img_path = os.path.join(images, "wall.png")
        cv2.imwrite(img_path, img)
        app.cfg.SCRIPT_DIR = tmpdir
        app.cfg.OUTPUT_FOLDER = os.path.join(tmpdir, "JSON files")
        app.cfg.CURRENT_IMAGE_PATH = img_path
        app.cfg.JSON_OUTPUT_PATH = os.path.join(app.cfg.OUTPUT_FOLDER, "wall.json")
        app.cfg.img_original = img
    return app


def _click(app, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)


def _all_points(app):
    return {tuple(p) for c in app.cfg.saved_cracks for p in c['path']}


class TestLinkOnTheLine(unittest.TestCase):

    def test_end_to_a_point_inside_another_crack(self):
        app = _t_junction_app()
        before = _all_points(app)
        self.assertTrue(app._link_cracks((1, 60), (0, 130)))   # end of crack 2 -> x=150 on crack 1
        self.assertEqual(len(app.cfg.saved_cracks), 2, "merged crack + the leftover of crack 1")
        merged, spur = app.cfg.saved_cracks
        self.assertEqual(_max_step(merged['path']), 1, "the merged crack has no gaps")
        self.assertEqual(merged['path'][0], (150, 20), "the first clicked crack leads")
        self.assertIn((150, 90), merged['path'], "the gap is bridged")
        self.assertEqual(merged['path'][-1], (20, 100), "the longer side of crack 1 follows")
        self.assertEqual({spur['path'][0], spur['path'][-1]}, {(150, 100), (280, 100)})
        self.assertTrue(before <= _all_points(app), "no traced point is lost")
        self.assertEqual(merged['links'], [{"from": [150, 80], "to": [150, 100], "method": "edge"}])

    def test_width_tracts_split_at_the_junction(self):
        app = _t_junction_app()
        app._link_cracks((1, 60), (0, 130))
        merged, spur = app.cfg.saved_cracks
        tracts = lambda c: {(tuple(sorted((c['path'][s['i0']], c['path'][s['i1']]))), s['width_px'])
                            for s in c.get('width_segments') or []}
        self.assertEqual(tracts(merged), {(((20, 100), (30, 100)), 3)})
        self.assertEqual(tracts(spur), {(((220, 100), (230, 100)), 5)})
        # the x=140..160 tract ran across the junction and is the only one removed

    def test_two_points_inside_both_cracks(self):
        app = _t_junction_app()
        before = _all_points(app)
        self.assertTrue(app._link_cracks((0, 130), (1, 20)))   # x=150 on crack 1 -> y=40 on crack 2
        self.assertEqual(len(app.cfg.saved_cracks), 3, "merged crack + one leftover per crack")
        self.assertTrue(before <= _all_points(app))
        for c in app.cfg.saved_cracks:
            self.assertEqual(_max_step(c['path']), 1)

    def test_ends_still_merge_whole_cracks(self):
        app = _t_junction_app()
        app.cfg.saved_cracks[1]['path'] = [(x, 100) for x in range(290, 299)][::-1]
        self.assertTrue(app._link_cracks((0, 260), (1, 8)))
        self.assertEqual(len(app.cfg.saved_cracks), 1)

    def test_undo_restores_the_original_cracks(self):
        app = _t_junction_app()
        app._link_cracks((1, 60), (0, 130))
        app._undo_last_action()
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertEqual(len(app.cfg.saved_cracks[0]['path']), 261)


class TestLinkOnTheLineClicks(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v107_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_clicks_merge_and_save_json_and_masks(self):
        app = _t_junction_app(self.tmpdir)
        app.handle_keyboard(ord('3'))
        _click(app, 151, 79)    # near the END of crack 2
        _click(app, 152, 101)   # ON crack 1, far from its ends
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "a completed LINK closes the tool")
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = json.load(f)["shapes"]
        self.assertEqual(len(shapes), 2, "merged crack + leftover, both in the JSON")
        self.assertEqual(sum(1 for s in shapes if s.get("links")), 1)
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(mask[90, 150], "the bridged gap is part of the binary mask")
        self.assertTrue(mask[100, 250], "the leftover crack is still in the binary mask")

    def test_an_end_wins_over_a_nearby_line_point(self):
        app = _t_junction_app()
        app.handle_keyboard(ord('3'))
        _click(app, 24, 102)   # within the end tolerance of crack 1's start, also on its line
        self.assertEqual(app.cfg.crack_cut_join_state["first"], (0, 0))


def _three_route_app():
    """Three edge routes from (20, 100) to (280, 100): straight, over the top (y=60) and underneath (y=140).
    One crack is traced with two clicks along the straight one."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    skel = np.zeros((H, W), dtype=np.uint8)
    cv2.line(skel, (20, 100), (280, 100), 255, 1)
    for y in (60, 140):
        cv2.polylines(skel, [np.array([(20, 100), (60, y), (240, y), (280, 100)], np.int32)], False, 255, 1)
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.refresh_zoom_viewport = lambda *a, **k: None
    _click(app, 20, 100)
    _click(app, 280, 100)
    return app


def _press(app, code):
    app.process_keypress(code)
    return app.cfg.alt_route_state


def _ys(route):
    return {p[1] for p in route}


class TestAlternativeRoute(unittest.TestCase):

    def setUp(self):
        self.app = _three_route_app()
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertEqual(_ys(self.app.cfg.saved_cracks[0]['path']), {100}, "traced along the straight edge")

    def test_presses_walk_forward_then_back(self):
        st = _press(self.app, 18)
        self.assertTrue(st["active"])
        self.assertEqual((st["index"], st["direction"]), (1, 1), "first press: first alternative, arrow right")
        first = st["routes"][1]
        self.assertTrue(min(_ys(first)) <= 60 or max(_ys(first)) >= 140, "a different edge")
        _press(self.app, 18)
        self.assertEqual((st["index"], st["direction"]), (2, -1), "last route found: the arrow turns left")
        self.assertTrue(st["exhausted"])
        self.assertNotEqual(min(_ys(st["routes"][2])) <= 60, min(_ys(first)) <= 60, "the third route is the other edge")
        _press(self.app, 18)
        self.assertEqual((st["index"], st["direction"]), (1, -1), "going back")
        _press(self.app, 18)
        self.assertEqual((st["index"], st["direction"]), (0, 1), "back on the blue route: the arrow turns right")
        _press(self.app, 18)
        self.assertEqual(st["index"], 1, "and forward again, without searching anew")
        self.assertEqual(len(st["routes"]), 3)

    def test_dollar_applies_the_route_on_screen(self):
        _press(self.app, 18)
        st = _press(self.app, 18)
        chosen = list(st["routes"][2])
        _press(self.app, ord('$'))
        self.assertFalse(st["active"], "[$] closes the tool")
        crack = self.app.cfg.saved_cracks[0]
        self.assertEqual(crack['path'], chosen)
        self.assertEqual((crack['start'], crack['end']), (chosen[0], chosen[-1]))
        self.assertTrue(self.app.cfg.blue_visual_mask.any(), "masks recalculated")
        self.app._undo_last_action()
        self.assertEqual(_ys(self.app.cfg.saved_cracks[0]['path']), {100}, "[U] restores the old route")

    def test_dollar_on_the_blue_route_changes_nothing(self):
        before = list(self.app.cfg.saved_cracks[0]['path'])
        _press(self.app, 18)
        _press(self.app, 18)
        _press(self.app, 18)
        _press(self.app, 18)   # back on the original route
        _press(self.app, ord('$'))
        self.assertFalse(self.app.cfg.alt_route_state["active"])
        self.assertEqual(self.app.cfg.saved_cracks[0]['path'], before)

    def test_width_tracts_are_dropped_with_the_old_route(self):
        self.app.cfg.saved_cracks[0]['width_segments'] = [{'i0': 0, 'i1': 10, 'width_px': 3}]
        _press(self.app, 18)
        _press(self.app, ord('$'))
        self.assertNotIn('width_segments', self.app.cfg.saved_cracks[0])

    def test_a_click_on_the_photo_closes_the_tool(self):
        before = list(self.app.cfg.saved_cracks[0]['path'])
        _press(self.app, 18)
        _click(self.app, 150, 30)
        self.assertFalse(self.app.cfg.alt_route_state["active"])
        self.assertEqual(self.app.cfg.saved_cracks[0]['path'], before)
        self.assertIsNone(self.app.cfg.temp_start, "the closing click does not start a new trace")

    def test_works_on_the_crack_traced_last(self):
        self.app.cfg.saved_cracks.insert(0, {'start': (20, 60), 'end': (60, 60), 'active': True,
                                             'path': [(x, 60) for x in range(20, 61)]})
        st = _press(self.app, 18)
        self.assertIs(st["crack"], self.app.cfg.saved_cracks[1])

    def test_undo_while_routes_are_shown_starts_over(self):
        _press(self.app, 18)
        self.app.cfg.saved_cracks[0] = dict(self.app.cfg.saved_cracks[0])  # the crack changed meanwhile
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_alt_route(win_out)
        self.assertFalse(win_out.any(), "nothing is drawn for a crack that changed")
        st = _press(self.app, 18)
        self.assertEqual(st["index"], 1, "the next press starts over on the current crack")

    def test_route_is_drawn_in_its_color(self):
        st = _press(self.app, 18)
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_alt_route(win_out)
        self.assertTrue((win_out.reshape(-1, 3) == (0, 0, 255)).all(axis=1).any(), "the first alternative is red")
        _press(self.app, 18)
        win_out[:] = 0
        self.app._draw_alt_route(win_out)
        self.assertTrue((win_out.reshape(-1, 3) == (0, 200, 0)).all(axis=1).any(), "the second is green")
        self.assertEqual(st["index"], 2)


class TestAlternativeRouteEdgeCases(unittest.TestCase):

    def test_no_crack_no_tool(self):
        app = make_app()
        app.refresh_zoom_viewport = lambda *a, **k: None
        self.assertFalse(_press(app, 18)["active"])

    def test_single_edge_has_no_alternative(self):
        app = _t_junction_app()
        app.refresh_zoom_viewport = lambda *a, **k: None
        self.assertFalse(_press(app, 18)["active"], "the vertical crack has no other edge to follow")

    def test_dollar_without_the_tool_does_nothing(self):
        app = _three_route_app()
        before = list(app.cfg.saved_cracks[0]['path'])
        _press(app, ord('$'))
        self.assertEqual(app.cfg.saved_cracks[0]['path'], before)

    def test_new_photo_forgets_the_traced_crack(self):
        app = _three_route_app()
        _press(app, 18)
        app._reset_alt_routes_for_new_image()
        self.assertFalse(app.cfg.alt_route_state["active"])
        self.assertIsNone(app.cfg.last_traced_crack)


class TestAlternativeRouteButton(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        self.app = make_gui_app(self)
        self.btn = next(b for b in self.app._sidebar.buttons if "Alternative route" in b.text)

    def test_button_sends_code_18(self):
        self.app._pending_keys.clear()
        self.btn.command()
        self.assertEqual(self.app._pending_keys, [18])

    def test_arrow_turns_left_and_button_looks_pressed(self):
        cfg = self.app.cfg
        refresh = lambda: self.app._sidebar.refresh_states(cfg, self.app._validation_available())
        refresh()
        self.assertEqual(self.btn.kw.get("relief"), "raised")
        self.assertTrue(self.btn.kw["text"].startswith("\u25B6"))
        cfg.alt_route_state.update(active=True, direction=-1)
        refresh()
        self.assertEqual(self.btn.kw.get("relief"), "sunken")
        self.assertTrue(self.btn.kw["text"].startswith("\u25C0"))
        cfg.alt_route_state.update(active=False, direction=1)
        refresh()
        self.assertTrue(self.btn.kw["text"].startswith("\u25B6"))


from crack_filter import parse_crack_numbers
from test_v103_fixes import _textured_image


def _four_crack_app():
    """Four short horizontal cracks, numbered 1..4 from top to bottom (y = 30, 70, 110, 150)."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.saved_cracks = [{'start': (40, y), 'end': (260, y), 'path': [(x, y) for x in range(40, 261)],
                             'active': True, 'session_id': 'current'} for y in (30, 70, 110, 150)]
    app.refresh_zoom_viewport = lambda *a, **k: None
    app.recalculate_masks()
    return app


def _filter(app, text):
    with mock.patch.object(app, "_prompt_crack_numbers", return_value=text):
        app.process_keypress(19)


class TestCrackNumberParsing(unittest.TestCase):

    def test_semicolons_spaces_commas_and_duplicates(self):
        self.assertEqual(parse_crack_numbers("12; 3;7"), [3, 7, 12])
        self.assertEqual(parse_crack_numbers("3,3; 4;"), [3, 4])
        self.assertEqual(parse_crack_numbers("  "), [])

    def test_anything_else_is_refused(self):
        for bad in ("a", "3;x", "0", "-2", "2.5"):
            with self.assertRaises(ValueError, msg=bad):
                parse_crack_numbers(bad)


class TestShowCracksByNumber(unittest.TestCase):

    def setUp(self):
        self.app = _four_crack_app()

    def test_only_the_typed_cracks_are_shown(self):
        _filter(self.app, "2;4")
        shown = [self.app._crack_shown(c) for c in self.app.cfg.saved_cracks]
        self.assertEqual(shown, [False, True, False, True])
        mask = self.app._display_blue_mask()
        self.assertFalse(mask[30, 100] or mask[110, 100], "cracks 1 and 3 hidden")
        self.assertTrue(mask[70, 100] and mask[150, 100], "cracks 2 and 4 shown")
        self.assertTrue(self.app.cfg.blue_visual_mask[30, 100], "the export mask still holds every crack")

    def test_markers_of_hidden_cracks_are_not_drawn(self):
        _filter(self.app, "2")
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_crack_markers(win_out, 14)
        _, wy1 = self.app.transform_real_to_window_coords(40, 30)
        _, wy2 = self.app.transform_real_to_window_coords(40, 70)
        self.assertFalse(win_out[wy1 - 20:wy1 + 20].any(), "no marker for crack 1")
        self.assertTrue(win_out[wy2 - 20:wy2 + 20].any(), "marker of crack 2")

    def test_hidden_cracks_cannot_be_clicked(self):
        _filter(self.app, "2")
        self.assertEqual(self.app._pick_active_crack_point(100, 30), (None, None))
        self.assertEqual(self.app._pick_active_crack_point(100, 70)[0], 1)
        self.assertFalse(self.app._delete_crack_near(40, 30), "a hidden crack is never deleted by a click")
        self.assertEqual(len(self.app.cfg.saved_cracks), 4)

    def test_an_edited_crack_stays_visible(self):
        _filter(self.app, "2")
        self.app.cfg.saved_cracks[0] = dict(self.app.cfg.saved_cracks[0], path=[(x, 30) for x in range(40, 200)])
        self.assertTrue(self.app._crack_shown(self.app.cfg.saved_cracks[0]))

    def test_empty_answer_shows_everything_cancel_and_errors_change_nothing(self):
        _filter(self.app, "2")
        _filter(self.app, None)
        self.assertEqual(self.app.cfg.crack_filter["numbers"], [2], "cancel keeps the filter")
        _filter(self.app, "two")
        self.assertEqual(self.app.cfg.crack_filter["numbers"], [2], "a typo keeps the filter")
        _filter(self.app, "9")
        self.assertEqual(self.app.cfg.crack_filter["numbers"], [2], "no such crack: filter unchanged")
        _filter(self.app, "")
        self.assertIsNone(self.app.cfg.crack_filter)
        self.assertTrue(self.app._display_blue_mask()[30, 100])

    def test_numbers_beyond_the_last_crack_are_ignored(self):
        _filter(self.app, "1;9")
        self.assertEqual(self.app.cfg.crack_filter["numbers"], [1])

    def test_label_and_new_photo(self):
        _filter(self.app, "3")
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_crack_filter_label(win_out)
        self.assertTrue(win_out.any())
        self.app._reset_crack_filter()
        self.assertIsNone(self.app.cfg.crack_filter)

    def test_files_keep_every_crack(self):
        tmp = tempfile.mkdtemp(prefix="crackseg_v107_filter_")
        self.addCleanup(shutil.rmtree, tmp, True)
        app = _t_junction_app(tmp)
        app.refresh_zoom_viewport = lambda *a, **k: None
        app.recalculate_masks()
        _filter(app, "1")
        self.assertTrue(app._save_annotations_now())
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["shapes"]), 2)
        mask = cv2.imread(os.path.join(tmp, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(mask[50, 150], "the hidden crack is still in the binary mask")


def _drag(app, p, q, steps=4):
    with mock.patch.object(app, "refresh_zoom_viewport"):
        w0 = app.transform_real_to_window_coords(*p)
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, w0[0], w0[1], 0, None)
        for t in range(1, steps + 1):
            x = p[0] + (q[0] - p[0]) * t / steps
            y = p[1] + (q[1] - p[1]) * t / steps
            wx, wy = app.transform_real_to_window_coords(x, y)
            app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)
        w1 = app.transform_real_to_window_coords(*q)
        app.mouse_callback(cv2.EVENT_LBUTTONUP, w1[0], w1[1], 0, None)


class TestBuildingPortionWindow(unittest.TestCase):

    def setUp(self):
        self.app = _four_crack_app()
        self.app.process_keypress(20)
        self.st = self.app.cfg.portion_state

    def test_drag_draws_moves_and_click_removes(self):
        self.assertTrue(self.st["active"])
        _drag(self.app, (20, 20), (150, 120))
        self.assertEqual(self.st["rect"], (20, 20, 150, 120))
        _drag(self.app, (80, 60), (130, 80))   # inside: moved by (+50, +20)
        self.assertEqual(self.st["rect"], (70, 40, 200, 140))
        _drag(self.app, (100, 100), (400, 400))   # inside: kept within the photo
        x0, y0, x1, y1 = self.st["rect"]
        self.assertEqual((x1, y1), (W - 1, H - 1))
        self.assertEqual((x1 - x0, y1 - y0), (130, 100), "moving keeps the size")
        _drag(self.app, (5, 5), (5, 5))   # plain click outside
        self.assertIsNone(self.st["rect"])

    def test_key_5_toggles_the_tool_like_the_button(self):
        self.app.process_keypress(ord('5'))
        self.assertFalse(self.st["active"])
        self.app.process_keypress(ord('5'))
        self.assertTrue(self.st["active"])

    def test_tiny_window_is_refused(self):
        _drag(self.app, (20, 20), (24, 23))
        self.assertIsNone(self.st["rect"])

    def test_second_press_closes_the_tool_and_keeps_the_window(self):
        _drag(self.app, (20, 20), (150, 120))
        self.app.process_keypress(20)
        self.assertFalse(self.st["active"])
        self.assertIsNotNone(self.st["rect"])
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_building_portion(win_out)
        self.assertTrue(win_out.any(), "the window stays visible")
        self.app._reset_building_portion()
        self.assertIsNone(self.st["rect"], "a new photo starts without a window")

    def test_target_mask_and_shape_filter(self):
        _drag(self.app, (0, 0), (149, 199))
        mask = self.app._portion_target_mask((H, W))
        self.assertEqual((mask[100, 100], mask[100, 200]), (255, 0))
        mask2 = self.app._portion_target_mask((2 * H, 2 * W))
        self.assertEqual((mask2[200, 290], mask2[200, 310]), (255, 0), "scaled to the photo as read")
        shapes = [{"points": [[10, 10], [100, 10]]}, {"points": [[120, 50], [200, 50]]}, {"points": [[160, 9], [290, 9]]}]
        kept, dropped = self.app._keep_shapes_in_portion(shapes, W, H)
        self.assertEqual((len(kept), dropped), (2, 1), "half inside is enough, fully outside is left out")


class TestBuildingPortionImport(unittest.TestCase):
    """Target photo: left half = the source scene (moved slightly), right half = another building."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v107_portion_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.H_true = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]])
        src = _textured_image(1)
        dst = cv2.warpPerspective(src, self.H_true, (800, 600))
        dst[:, 400:] = _textured_image(7)[:, 400:]
        self.src_path = os.path.join(self.tmpdir, "src.png")
        self.dst_path = os.path.join(self.tmpdir, "dst.png")
        cv2.imwrite(self.src_path, src)
        cv2.imwrite(self.dst_path, dst)
        self.left = [[100.0, 200.0], [160.0, 260.0], [220.0, 330.0]]
        self.right = [[520.0, 200.0], [600.0, 280.0], [680.0, 350.0]]
        self.src_json = os.path.join(self.tmpdir, "src.json")
        with open(self.src_json, "w", encoding="utf-8") as f:
            json.dump({"shapes": [{"label": "crack", "points": pts, "shape_type": "linestrip", "flags": {}}
                                  for pts in (self.left, self.right)],
                       "imagePath": "src.png", "imageHeight": 600, "imageWidth": 800}, f)
        self.out_json = os.path.join(self.tmpdir, "out.json")
        self.app = make_app()
        self.app.cfg.H_img, self.app.cfg.W_img = 600, 800
        self.app.cfg.portion_state["rect"] = (0, 0, 380, 599)

    def test_import_aligns_on_the_window_and_keeps_only_what_is_inside(self):
        spy = mock.patch.object(self.app, "_find_good_sift_matches", wraps=self.app._find_good_sift_matches)
        with spy as m:
            ok = self.app.warp_and_adapt_json_to_new_image(self.src_path, self.dst_path, self.src_json, self.out_json)
        self.assertTrue(ok, self.app.cfg.import_warning_message)
        dst_mask = m.call_args_list[0][0][3]   # the first (whole-photo) matching; a pre-warp may follow
        self.assertEqual((dst_mask[300, 100], dst_mask[300, 600]), (255, 0), "target features only in the window")
        with open(self.out_json, encoding="utf-8") as f:
            shapes = json.load(f)["shapes"]
        self.assertEqual(len(shapes), 1, "the crack of the other building is left out")
        projected = np.array(shapes[0]["points"], dtype=np.float32)
        expected = np.array(self.left, dtype=np.float32) + [12.0, 8.0]
        self.assertLess(float(np.max(np.linalg.norm(projected - expected, axis=1))), 3.0)
        self.assertIn("building portion", self.app.cfg.warp_banner_message)

    def test_without_a_window_the_import_is_unchanged(self):
        self.app.cfg.portion_state["rect"] = None
        with mock.patch.object(self.app, "_find_good_sift_matches", wraps=self.app._find_good_sift_matches) as m:
            self.app.warp_and_adapt_json_to_new_image(self.src_path, self.dst_path, self.src_json, self.out_json)
        self.assertIsNone(m.call_args_list[0][0][3])
        self.assertNotIn("building portion", self.app.cfg.warp_banner_message)


class TestPerspectivePrewarp(unittest.TestCase):
    """Source: a close-up of a textured facade. Target: the same facade seen from much farther and at an angle
    (area scale ~0.15, below HOMOGRAPHY_MIN_AREA_SCALE), so the first alignment alone is rejected."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v107_prewarp_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        src = _textured_image(3, size=(900, 1200))
        # Far and oblique: scale ~0.4, a perspective tilt, and an unrelated background around the facade.
        self.H_true = np.array([[0.42, 0.05, 300.0], [0.01, 0.38, 180.0], [0.00012, 0.00002, 1.0]])
        dst = _textured_image(11, size=(900, 1200))
        warped = cv2.warpPerspective(src, self.H_true, (1200, 900))
        inside = cv2.warpPerspective(np.full((900, 1200), 255, np.uint8), self.H_true, (1200, 900)) > 0
        dst[inside] = warped[inside]
        self.src_path = os.path.join(self.tmpdir, "src.png")
        self.dst_path = os.path.join(self.tmpdir, "dst.png")
        cv2.imwrite(self.src_path, src)
        cv2.imwrite(self.dst_path, dst)
        self.crack = [[300.0, 300.0], [420.0, 380.0], [560.0, 430.0], [700.0, 560.0]]
        self.src_json = os.path.join(self.tmpdir, "src.json")
        with open(self.src_json, "w", encoding="utf-8") as f:
            json.dump({"shapes": [{"label": "crack", "points": self.crack, "shape_type": "linestrip", "flags": {}}],
                       "imagePath": "src.png", "imageHeight": 900, "imageWidth": 1200}, f)
        self.out_json = os.path.join(self.tmpdir, "out.json")

    def _import(self, refine=True):
        app = make_app()
        app.cfg.H_img, app.cfg.W_img = 900, 1200
        app.cfg.WARP_PREWARP_REFINE = refine
        ok = app.warp_and_adapt_json_to_new_image(self.src_path, self.dst_path, self.src_json, self.out_json)
        return app, ok

    def test_far_oblique_view_is_imported_on_the_right_place(self):
        app, ok = self._import()
        self.assertTrue(ok, app.cfg.import_warning_message)
        with open(self.out_json, encoding="utf-8") as f:
            projected = np.array(json.load(f)["shapes"][0]["points"], dtype=np.float32)
        expected = cv2.perspectiveTransform(np.array([self.crack], dtype=np.float32), self.H_true)[0]
        self.assertLess(float(np.max(np.linalg.norm(projected - expected, axis=1))), 2.0)

    def test_without_the_prewarp_the_same_pair_is_rejected(self):
        app, ok = self._import(refine=False)
        self.assertFalse(ok, "the first alignment alone is refused (scale too far from 1)")

    def test_unusable_starts_are_not_refined(self):
        app = make_app()
        self.assertFalse(app._is_usable_prewarp_start(np.diag([-1.0, 1.0, 1.0]), (100, 100)), "mirror")
        self.assertFalse(app._is_usable_prewarp_start(np.array([[1, 0, 0], [0, 1, 0], [-0.02, 0, 1.0]]), (100, 100)),
                         "a corner behind the camera")
        self.assertTrue(app._is_usable_prewarp_start(np.diag([0.4, 0.4, 1.0]), (100, 100)))


class TestCompatibleArea(unittest.TestCase):
    """Group BLDG001: a close-up already segmented, an unrelated photo also segmented, and the current wide view."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v107_compat_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        images, jsons = os.path.join(self.tmpdir, "Images"), os.path.join(self.tmpdir, "JSON files")
        os.makedirs(images)
        os.makedirs(jsons)
        src = _textured_image(3, size=(900, 1200))
        self.H_true = np.array([[0.42, 0.05, 300.0], [0.01, 0.38, 180.0], [0.00012, 0.00002, 1.0]])
        cur = _textured_image(11, size=(900, 1200))
        inside = cv2.warpPerspective(np.full((900, 1200), 255, np.uint8), self.H_true, (1200, 900)) > 0
        cur[inside] = cv2.warpPerspective(src, self.H_true, (1200, 900))[inside]
        for name, img in (("close__BLDG001", src), ("other__BLDG001", _textured_image(21, size=(900, 1200))),
                          ("wide__BLDG001", cur)):
            cv2.imwrite(os.path.join(images, name + ".png"), img)
        for name in ("close__BLDG001", "other__BLDG001"):
            with open(os.path.join(jsons, name + ".json"), "w", encoding="utf-8") as f:
                json.dump({"shapes": [{"label": "crack", "points": [[300, 300], [700, 560]], "shape_type": "linestrip"}],
                           "imagePath": name + ".png", "imageHeight": 900, "imageWidth": 1200}, f)
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR, cfg.IMAGE_FOLDER = self.tmpdir, images
        cfg.CURRENT_IMAGE_PATH = os.path.join(images, "wide__BLDG001.png")
        cfg.img_original = cur
        cfg.H_img, cfg.W_img = 900, 1200
        cfg.zoom_box = [0, 0, 1200, 900]
        self.app.refresh_zoom_viewport = lambda *a, **k: None

    def test_shared_area_is_found_tinted_and_framed(self):
        self.app.process_keypress(ord('5'))
        st = self.app.cfg.portion_state
        self.assertEqual(len(st["compatible"]), 1, "the close-up is found, the unrelated photo is not")
        expected = cv2.perspectiveTransform(np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]]).reshape(-1, 1, 2),
                                            self.H_true).reshape(-1, 2)
        x0, y0, x1, y1 = st["rect"]
        self.assertLess(abs(x0 - expected[:, 0].min()), 6)
        self.assertLess(abs(y1 - expected[:, 1].max()), 6)
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_building_portion(win_out)
        cx, cy = (int(v) for v in expected.mean(axis=0))
        b, g, r = win_out[cy, cx]
        self.assertTrue(r > g > b, "orange glass inside the shared area")
        self.assertFalse(win_out[880, 20].any(), "nothing outside it")

    def test_searched_once_per_photo(self):
        self.app.process_keypress(ord('5'))
        self.app.process_keypress(ord('5'))
        with mock.patch.object(self.app, "_find_compatible_area") as m:
            self.app.process_keypress(ord('5'))
        m.assert_not_called()
        self.app._reset_building_portion()
        self.assertIsNone(self.app.cfg.portion_state["compatible"], "a new photo searches again")

    def test_a_window_drawn_by_hand_is_kept(self):
        self.app.cfg.portion_state["rect"] = (10, 10, 200, 200)
        self.app.process_keypress(ord('5'))
        self.assertEqual(self.app.cfg.portion_state["rect"], (10, 10, 200, 200))

    def test_photo_without_group_just_opens_the_tool(self):
        self.app.cfg.CURRENT_IMAGE_PATH = os.path.join(self.tmpdir, "Images", "nogroup.png")
        self.app.process_keypress(ord('5'))
        self.assertTrue(self.app.cfg.portion_state["active"])
        self.assertEqual(self.app.cfg.portion_state["compatible"], [])
        self.assertIsNone(self.app.cfg.portion_state["rect"])


class TestProjectionAtTheBorder(unittest.TestCase):

    def test_crack_leaving_the_view_is_cut_not_squashed(self):
        app = make_app()
        H = np.eye(3)
        shape = {"label": "crack", "shape_type": "linestrip", "points": [[10, 50], [60, 50], [90, 50], [150, 50], [190, 50]]}
        kept, implausible = app._project_one_shape(shape, H, 100, 100, 0.3, 3.0)
        self.assertFalse(implausible)
        self.assertEqual([p[0] for p in kept["points"]], [10, 60, 90], "cut where it leaves the 100 px wide photo")

    def test_shape_mostly_outside_is_left_out(self):
        app = make_app()
        shape = {"label": "crack", "shape_type": "linestrip", "points": [[90, 50], [150, 50], [190, 50]]}
        self.assertEqual(app._project_one_shape(shape, np.eye(3), 100, 100, 0.3, 3.0), (None, False))

    def test_span_check_follows_the_alignment_scale(self):
        app = make_app()
        shape = {"label": "crack", "shape_type": "linestrip", "points": [[10, 10], [60, 60], [90, 90]]}
        kept, implausible = app._project_one_shape(shape, np.diag([0.2, 0.2, 1.0]), 100, 100, 0.3, 3.0)
        self.assertIsNotNone(kept, "a crack 5x smaller in a photo taken 5x farther is plausible")


class TestNewSidebarButtons(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def _button(self, label):
        return next(b for b in self.app._sidebar.buttons if label in b.text)

    def test_buttons_send_their_codes(self):
        for label, code in (("Show cracks by number", 19), ("Building portion", 20)):
            self.app._pending_keys.clear()
            self._button(label).command()
            self.assertEqual(self.app._pending_keys, [code])

    def test_prompt_uses_a_text_dialog(self):
        self.gm.simpledialog.next_string = "3;7"
        self.assertEqual(self.app._prompt_crack_numbers("1"), "3;7")
        self.assertIn("cracks by number", self.gm.simpledialog.last_call[0])

    def test_buttons_look_pressed_while_on(self):
        cfg = self.app.cfg
        cfg.crack_filter = {"numbers": [2], "hidden_keys": set()}
        cfg.portion_state["rect"] = (0, 0, 10, 10)
        self.app._sidebar.refresh_states(cfg, self.app._validation_available())
        self.assertEqual(self._button("Show cracks by number").kw.get("relief"), "sunken")
        self.assertEqual(self._button("Building portion").kw.get("relief"), "sunken")


def _shadow_app():
    """A short straight 'shadow' edge y=100 and the real crack arching over it through y=40."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    skel = np.zeros((H, W), dtype=np.uint8)
    cv2.line(skel, (20, 100), (280, 100), 255, 1)
    cv2.polylines(skel, [np.array([(20, 100), (80, 40), (220, 40), (280, 100)], np.int32)], False, 255, 1)
    cv2.line(skel, (150, 40), (150, 20), 255, 1)   # a side branch, so a waypoint can sit off the arc too
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    # The photo: a broad shadow below y=100 (its edge is what the skeleton follows) and a FAINT thin crack on the arc.
    img = np.full((H, W, 3), 200, dtype=np.uint8)
    img[101:] = 150
    cv2.polylines(img, [np.array([(20, 100), (80, 40), (220, 40), (280, 100)], np.int32)], False, (170, 170, 170), 1)
    cv2.line(img, (150, 40), (150, 20), (170, 170, 170), 1)
    app.cfg.img_original = img
    app.refresh_zoom_viewport = lambda *a, **k: None
    return app


def _turns_back(path, min_gap=12, near=2.5):
    """True if the route comes back within `near` px of where it was `min_gap`+ steps earlier (out and back)."""
    pts = np.asarray(path, dtype=np.float64)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    i, j = np.triu_indices(len(pts), k=min_gap)
    return bool(np.any(d[i, j] < near))


def _click_flags(app, x, y, flags=0):
    wx, wy = app.transform_real_to_window_coords(x, y)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, flags, None)


SHIFT = cv2.EVENT_FLAG_SHIFTKEY


class TestTraceWaypoints(unittest.TestCase):

    def setUp(self):
        self.app = _shadow_app()

    def test_two_clicks_trace_exactly_as_before(self):
        expected = self.app.a_star_pathfinding((20, 100), (280, 100), 'crack')
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 280, 100)
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertEqual(self.app.cfg.saved_cracks[0]['path'], expected)
        self.assertEqual({p[1] for p in expected}, {100}, "unconstrained, the route takes the shadow edge")

    def test_shift_click_forces_the_route_through_the_point(self):
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 150, 40, SHIFT)
        self.assertEqual(len(self.app.cfg.saved_cracks), 0, "Shift+click does not end the trace")
        self.assertEqual(self.app.cfg.trace_waypoints, [(150, 40)])
        self.assertIn((150, 40), self.app.cfg.temp_path, "the route so far is shown")
        _click_flags(self.app, 280, 100)
        crack = self.app.cfg.saved_cracks[0]
        self.assertIn((150, 40), crack['path'])
        self.assertEqual(_max_step(crack['path']), 1, "no gaps at the intermediate point")
        self.assertFalse(_turns_back(crack['path']), "the route passes THROUGH the point, it does not turn back")
        self.assertLessEqual(max(p[1] for p in crack['path'] if 80 <= p[0] <= 220), 42, "it follows the faint crack")
        self.assertEqual((crack['start'], crack['end']), ((20, 100), (280, 100)))
        self.assertEqual(self.app.cfg.trace_waypoints, [], "ready for the next crack")

    def test_two_intermediate_points_and_no_more(self):
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 80, 40, SHIFT)
        _click_flags(self.app, 220, 40, SHIFT)
        _click_flags(self.app, 150, 20, SHIFT)   # a third one is refused
        wps = list(self.app.cfg.trace_waypoints)
        self.assertEqual(len(wps), 2)
        self.assertLess(wps[0][0], wps[1][0])
        _click_flags(self.app, 280, 100)
        path = self.app.cfg.saved_cracks[0]['path']
        self.assertLess(path.index(wps[0]), path.index(wps[1]), "points visited in click order")

    def test_undo_removes_the_last_point_then_the_trace(self):
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 80, 40, SHIFT)
        _click_flags(self.app, 220, 40, SHIFT)
        self.app._undo_last_action()
        self.assertEqual(len(self.app.cfg.trace_waypoints), 1)
        self.assertIsNotNone(self.app.cfg.temp_start, "the trace is still in progress")
        self.app._undo_last_action()
        self.app._undo_last_action()
        self.assertIsNone(self.app.cfg.temp_start)
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 280, 100)
        self.assertEqual({p[1] for p in self.app.cfg.saved_cracks[0]['path']}, {100}, "no leftover point")

    def test_point_slightly_off_the_crack_still_passes_through(self):
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 150, 43, SHIFT)   # 3 px below the faint crack
        _click_flags(self.app, 280, 100)
        path = self.app.cfg.saved_cracks[0]['path']
        self.assertIn((150, 40), path, "the point is snapped onto the crack line")
        self.assertFalse(_turns_back(path))

    def test_shift_on_the_first_click_just_starts(self):
        _click_flags(self.app, 20, 100, SHIFT)
        self.assertEqual(self.app.cfg.temp_start, (20, 100))
        self.assertEqual(self.app.cfg.trace_waypoints, [])

    def test_markers_and_preview(self):
        _click_flags(self.app, 20, 100)
        _click_flags(self.app, 150, 40, SHIFT)
        wx, wy = self.app.transform_real_to_window_coords(200, 60)
        self.app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)
        self.assertEqual(self.app.cfg.temp_path[-1], (200, 60))
        self.assertIn((150, 40), self.app.cfg.temp_path, "the preview keeps the route traced so far")
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_temp_crack_start_marker(win_out, 14)
        px, py = self.app.transform_real_to_window_coords(150, 40)
        self.assertTrue(win_out[py - 12:py + 12, px - 12:px + 12].any(), "the intermediate point is marked")

    def test_gui_reports_shift_like_opencv(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        import crack_segmentation_gui as gm
        ev = type("Ev", (), {"state": 0x0001})()
        self.assertEqual(gm.EmbeddedCanvas._mouse_flags(ev), SHIFT)
        ev.state = 0
        self.assertEqual(gm.EmbeddedCanvas._mouse_flags(ev), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
