"""test_v107_ignore.py

Headless tests for the v1.0.7 ignore-region tool [I]:
  1. [I] selects the tool; [+]/[=] still zoom, [I] no longer does
  2. clicking corners and closing with [Y] saves a polygon; [U]/[R] undo and redo it
  3. click inside a saved polygon deletes it (undoable)
  4. JSON export / reload round trip, -ignore_mask.png, never mistaken for a detachment
  5. [X] Clear All, W/L import projection, training validation counts
  6. GUI: sidebar button and Tools menu entry
  7. options dialog: rectangle / square by two clicks, with sides extended to the photo edges
  8. reshaping: drag a polygon corner, or a rectangle corner/side; undo/redo

Run with:  python3 -m unittest test_v107_ignore -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import training_validation as tv
from test_gui_shell_parity import _MockedHighGui, _cleanup_real_module_dir_artifacts
from test_v105_link import _two_crack_app, H, W

SQUARE = [(40, 40), (140, 40), (140, 140), (40, 140)]


def _poly_app(tmpdir=None):
    """App with the ignore tool in free-polygon mode (the dialog default is rectangle)."""
    app = _two_crack_app(tmpdir)
    app.cfg.ignore_options = {"shape": "polygon", "edges": {e: False for e in ("left", "right", "top", "bottom")}}
    return app


def _rect_app(shape="rectangle", **edges):
    app = _two_crack_app()
    app.cfg.ignore_options = {"shape": shape,
                              "edges": {e: bool(edges.get(e)) for e in ("left", "right", "top", "bottom")}}
    return app


def _click(app, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)


def _draw_square(app, corners=SQUARE):
    app.process_keypress(ord('i'))
    for x, y in corners:
        _click(app, x, y)
    app.process_keypress(ord('y'))


def _close(a, b, tol=3):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


class TestIgnoreToolKeys(unittest.TestCase):

    def test_i_selects_the_ignore_tool(self):
        app = _poly_app()
        app.process_keypress(ord('i'))
        self.assertEqual(app.cfg.current_tool, 'ignore')
        app.process_keypress(ord('c'))
        self.assertEqual(app.cfg.current_tool, 'crack')
        app.process_keypress(ord('I'))
        self.assertEqual(app.cfg.current_tool, 'ignore')

    def test_i_no_longer_zooms_but_plus_still_does(self):
        app = _poly_app()
        with mock.patch.object(app, "_zoom_in") as zoom_in:
            app.process_keypress(ord('i'))
            zoom_in.assert_not_called()
            app.process_keypress(ord('+'))
            app.process_keypress(ord('='))
        self.assertEqual(zoom_in.call_count, 2)

    def test_help_menu_lists_the_i_key(self):
        app = _poly_app()
        self.assertTrue(any("[I]" in line for line in app._help_menu_commands()))


class TestIgnorePolygon(unittest.TestCase):

    def test_corners_and_y_save_a_straight_polygon(self):
        app = _poly_app()
        _draw_square(app)
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        path = app.cfg.saved_ignores[0]['path']
        self.assertEqual(len(path), 4, "straight edges: one point per click, no A* snapping")
        for got, want in zip(path, SQUARE):
            self.assertTrue(_close(got, want), (got, want))
        self.assertEqual(app.cfg.temp_nodes, [])
        self.assertEqual(app.cfg.saved_detachments, [], "never stored as a detachment")

    def test_y_needs_three_corners(self):
        app = _poly_app()
        _draw_square(app, SQUARE[:2])
        self.assertEqual(app.cfg.saved_ignores, [])
        self.assertEqual(len(app.cfg.temp_nodes), 2, "the open polygon stays in progress")

    def test_u_removes_the_last_corner_while_drawing(self):
        app = _poly_app()
        app.process_keypress(ord('i'))
        for x, y in SQUARE[:3]:
            _click(app, x, y)
        app.process_keypress(ord('u'))
        self.assertEqual(len(app.cfg.temp_nodes), 2)

    def test_undo_and_redo_a_saved_polygon(self):
        app = _poly_app()
        _draw_square(app)
        app.process_keypress(ord('u'))
        self.assertEqual(app.cfg.saved_ignores, [])
        app.process_keypress(ord('r'))
        self.assertEqual(len(app.cfg.saved_ignores), 1)

    def test_click_inside_deletes_and_undo_restores(self):
        app = _poly_app()
        _draw_square(app)
        _click(app, 90, 90)
        self.assertEqual(app.cfg.saved_ignores, [])
        self.assertEqual(app.cfg.temp_nodes, [], "a delete click must not start a new polygon")
        app.process_keypress(ord('u'))
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        app.process_keypress(ord('r'))
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_click_outside_starts_a_new_polygon(self):
        app = _poly_app()
        _draw_square(app)
        _click(app, 220, 170)
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        self.assertEqual(len(app.cfg.temp_nodes), 1)

    def test_corners_can_be_placed_inside_a_region_while_drawing(self):
        app = _poly_app()
        _draw_square(app)
        _click(app, 200, 100)
        _click(app, 90, 90)  # inside the saved square, but a polygon is in progress
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        self.assertEqual(len(app.cfg.temp_nodes), 2)

    def test_overlapping_regions_delete_the_most_recent(self):
        app = _poly_app()
        _draw_square(app)
        # The first corner must lie outside existing regions (a click there deletes); the others can go anywhere.
        _draw_square(app, [(200, 160), (100, 160), (100, 60), (200, 60)])
        self.assertEqual(len(app.cfg.saved_ignores), 2)
        _click(app, 120, 100)  # inside both
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        self.assertTrue(_close(app.cfg.saved_ignores[0]['path'][0], SQUARE[0]), "the older square survives")

    def test_switching_to_detachment_drops_the_open_ignore_polygon(self):
        app = _poly_app()
        app.process_keypress(ord('i'))
        _click(app, *SQUARE[0])
        _click(app, *SQUARE[1])
        app.process_keypress(ord('d'))
        self.assertEqual(app.cfg.temp_nodes, [])

    def test_y_still_closes_detachments(self):
        app = _poly_app()
        app.process_keypress(ord('d'))
        with mock.patch.object(app, "_close_detachment_polygon") as close_det:
            app.process_keypress(ord('y'))
        close_det.assert_called_once()

    def test_clear_all_is_undoable_for_ignores(self):
        app = _poly_app()
        app.cfg.JSON_OUTPUT_PATH = "/tmp/nonexistent_test_clear_all_ignore.json"
        _draw_square(app)
        app._clear_all()
        self.assertEqual(app.cfg.saved_ignores, [])
        app._undo_last_action()
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        app._handle_redo_key()
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_undo_of_an_old_clear_all_backup_without_ignores(self):
        app = _poly_app()
        app.cfg.saved_ignores = [{'path': SQUARE, 'active': True}]
        app._undo_clear_all({'cracks': [], 'detachments': []})
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_rendering_tints_the_region_and_leaves_the_rest(self):
        app = _poly_app()
        _draw_square(app)
        win = np.full((900, 1200, 3), 200, dtype=np.uint8)
        app._draw_ignore_regions(win, 22)
        cx, cy = app.transform_real_to_window_coords(90, 90)
        ox, oy = app.transform_real_to_window_coords(220, 170)
        self.assertFalse((win[cy, cx] == 200).all(), "inside the polygon is tinted")
        self.assertTrue((win[oy, ox] == 200).all(), "outside stays untouched")


class TestIgnoreEdgeSnap(unittest.TestCase):

    def test_corners_near_the_edges_snap_onto_them(self):
        app = _poly_app()  # whole 300x200 photo on a 1200x900 window: 15 win px = ~3.75 x / 3.3 y image px
        self.assertEqual(app._snap_to_photo_edges(2, 3), (0, 0))
        self.assertEqual(app._snap_to_photo_edges(W - 3, H - 2), (W - 1, H - 1))
        self.assertEqual(app._snap_to_photo_edges(50, 60), (50, 60), "corners away from the edges stay put")

    def test_snap_tolerance_shrinks_when_zoomed_in(self):
        app = _poly_app()
        self.assertEqual(app._snap_to_photo_edges(3, 3), (0, 0), "whole photo: 3 px is within tolerance")
        app.cfg.zoom_box = [0, 0, W // 4, H // 4]  # 4x zoom: tolerance below 1 image px
        self.assertEqual(app._snap_to_photo_edges(3, 3), (3, 3), "zoomed in, the same corner stays put")

    def test_mask_reaches_the_photo_edges(self):
        app = _poly_app()
        _draw_square(app, [(2, H - 60), (W - 3, H - 60), (W - 2, H - 2), (3, H - 3)])
        mask = app._build_ignore_area_mask()
        self.assertTrue(mask[H - 1, :].all(), "no unmasked strip along the bottom edge")
        self.assertTrue(mask[H - 30, 0] and mask[H - 30, W - 1], "left and right edges covered")


class TestIgnoreRectangle(unittest.TestCase):

    def _two_clicks(self, app, p1, p2):
        app.process_keypress(ord('i'))
        _click(app, *p1)
        _click(app, *p2)

    def test_rectangle_is_the_default_shape(self):
        app = _two_crack_app()
        self.assertEqual(app._ignore_options()["shape"], "rectangle")

    def test_two_clicks_save_a_free_rectangle(self):
        app = _rect_app()
        self._two_clicks(app, (60, 50), (160, 120))
        self.assertEqual(len(app.cfg.saved_ignores), 1)
        xs = [p[0] for p in app.cfg.saved_ignores[0]['path']]
        ys = [p[1] for p in app.cfg.saved_ignores[0]['path']]
        self.assertTrue(abs(min(xs) - 60) <= 3 and abs(max(xs) - 160) <= 3)
        self.assertTrue(abs(min(ys) - 50) <= 3 and abs(max(ys) - 120) <= 3)
        self.assertEqual(app.cfg.temp_nodes, [])

    def test_corners_in_any_order(self):
        app = _rect_app()
        a = app._ignore_rect_corners((160, 120), (60, 50))
        b = app._ignore_rect_corners((60, 50), (160, 120))
        self.assertEqual(a, b)

    def test_ticked_edges_reach_the_photo_border(self):
        app = _rect_app(right=True, bottom=True)
        corners = app._ignore_rect_corners((60, 50), (160, 120))
        self.assertEqual(corners, [(60, 50), (W - 1, 50), (W - 1, H - 1), (60, H - 1)])
        app = _rect_app(left=True, top=True)
        self.assertEqual(app._ignore_rect_corners((60, 50), (160, 120))[0], (0, 0))

    def test_all_four_edges_cover_the_whole_photo(self):
        app = _rect_app(left=True, right=True, top=True, bottom=True)
        self._two_clicks(app, (100, 80), (120, 90))
        mask = app._build_ignore_area_mask()
        self.assertTrue(mask.all())

    def test_safety_net_band_across_the_bottom(self):
        app = _rect_app(left=True, right=True, bottom=True)
        self._two_clicks(app, (150, 140), (151, 141))  # only the top edge (y=140) matters
        mask = app._build_ignore_area_mask()
        self.assertTrue(mask[140:, :].all(), "band from y=140 down, edge to edge")
        self.assertFalse(mask[:139, :].any())

    def test_square_uses_the_longer_side(self):
        app = _rect_app("square")
        corners = app._ignore_rect_corners((50, 50), (130, 80))
        self.assertEqual(corners, [(50, 50), (130, 50), (130, 130), (50, 130)])
        corners = app._ignore_rect_corners((100, 100), (80, 40))
        self.assertEqual(corners, [(40, 40), (100, 40), (100, 100), (40, 100)])

    def test_preview_follows_the_cursor(self):
        app = _rect_app()
        app.process_keypress(ord('i'))
        _click(app, 60, 50)
        wx, wy = app.transform_real_to_window_coords(160, 120)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)
        self.assertEqual(len(app.cfg.temp_path), 4)
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_u_cancels_the_first_corner(self):
        app = _rect_app()
        app.process_keypress(ord('i'))
        _click(app, 60, 50)
        app.process_keypress(ord('u'))
        self.assertEqual(app.cfg.temp_nodes, [])
        _click(app, 160, 120)
        self.assertEqual(app.cfg.saved_ignores, [], "the next click starts a new rectangle")

    def test_empty_rectangle_is_not_saved(self):
        app = _rect_app()
        self._two_clicks(app, (60, 50), (60, 120))
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_click_inside_a_rectangle_deletes_it(self):
        app = _rect_app()
        self._two_clicks(app, (60, 50), (160, 120))
        _click(app, 100, 90)
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_y_does_nothing_in_rectangle_mode(self):
        app = _rect_app()
        app.process_keypress(ord('i'))
        _click(app, 60, 50)
        app.process_keypress(ord('y'))
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_cancelled_dialog_leaves_the_tool_unchanged(self):
        app = _rect_app()
        with mock.patch.object(app, "_prompt_ignore_options", return_value=None):
            app.process_keypress(ord('i'))
        self.assertEqual(app.cfg.current_tool, 'crack')

    def test_dialog_choice_is_remembered(self):
        app = _two_crack_app()
        choice = {"shape": "square", "edges": {"left": True}}
        with mock.patch.object(app, "_prompt_ignore_options", return_value=choice) as prompt:
            app.process_keypress(ord('i'))
        self.assertEqual(app.cfg.ignore_options["shape"], "square")
        self.assertEqual(app.cfg.ignore_options["edges"],
                         {"left": True, "right": False, "top": False, "bottom": False})
        with mock.patch.object(app, "_prompt_ignore_options", return_value=None) as prompt:
            app.process_keypress(ord('i'))
        self.assertEqual(prompt.call_args[0][0]["shape"], "square", "the dialog opens on the last choice")

    def test_rectangle_renders_closed_while_drawing(self):
        app = _rect_app()
        app.process_keypress(ord('i'))
        _click(app, 60, 50)
        wx, wy = app.transform_real_to_window_coords(160, 120)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_ignore_regions(win, 22)
        lx, ly = app.transform_real_to_window_coords(60, 85)  # middle of the left side
        self.assertTrue(win[ly, lx - 1:lx + 2].any(), "the closing (left) side is drawn")


def _drag(app, p_from, p_to):
    for event, (x, y) in ((cv2.EVENT_LBUTTONDOWN, p_from), (cv2.EVENT_MOUSEMOVE, p_to), (cv2.EVENT_LBUTTONUP, p_to)):
        wx, wy = app.transform_real_to_window_coords(x, y)
        app.mouse_callback(event, wx, wy, 0, None)


class TestIgnoreReshape(unittest.TestCase):

    def _bounds(self, app, idx=0):
        path = app.cfg.saved_ignores[idx]['path']
        return (min(p[0] for p in path), min(p[1] for p in path), max(p[0] for p in path), max(p[1] for p in path))

    def _rect(self):
        app = _rect_app()
        app.process_keypress(ord('i'))
        _click(app, 60, 50)
        _click(app, 160, 120)
        return app

    def test_drag_a_polygon_corner(self):
        app = _poly_app()
        _draw_square(app)
        _drag(app, (140, 140), (180, 170))
        path = app.cfg.saved_ignores[0]['path']
        self.assertTrue(_close(path[2], (180, 170)), path)
        self.assertTrue(_close(path[0], SQUARE[0]), "the other corners stay put")
        self.assertEqual(len(app.cfg.saved_ignores), 1, "grabbing a corner never deletes")

    def test_drag_a_rectangle_corner_keeps_it_rectangular(self):
        app = self._rect()
        _drag(app, (160, 120), (200, 150))
        l, t, r, b = self._bounds(app)
        self.assertTrue(abs(l - 60) <= 3 and abs(t - 50) <= 3 and abs(r - 200) <= 3 and abs(b - 150) <= 3)
        self.assertTrue(app._is_axis_rectangle(app.cfg.saved_ignores[0]['path']))

    def test_drag_a_rectangle_side(self):
        app = self._rect()
        _drag(app, (160, 85), (230, 95))  # right side, halfway down: only x moves
        l, t, r, b = self._bounds(app)
        self.assertTrue(abs(r - 230) <= 3)
        self.assertTrue(abs(t - 50) <= 3 and abs(b - 120) <= 3, "top and bottom unchanged")

    def test_dragging_a_side_onto_the_photo_edge_snaps(self):
        app = self._rect()
        _drag(app, (160, 85), (W - 3, 85))
        self.assertEqual(self._bounds(app)[2], W - 1)

    def test_rectangle_can_be_flipped_past_the_opposite_side(self):
        app = self._rect()
        _drag(app, (160, 85), (20, 85))
        l, t, r, b = self._bounds(app)
        self.assertTrue(abs(l - 20) <= 3 and abs(r - 60) <= 3)

    def test_undo_and_redo_a_reshape(self):
        app = self._rect()
        before = list(app.cfg.saved_ignores[0]['path'])
        _drag(app, (160, 120), (200, 150))
        after = list(app.cfg.saved_ignores[0]['path'])
        app.process_keypress(ord('u'))
        self.assertEqual(app.cfg.saved_ignores[0]['path'], before)
        app.process_keypress(ord('r'))
        self.assertEqual(app.cfg.saved_ignores[0]['path'], after)

    def test_collapsing_a_rectangle_reverts(self):
        app = self._rect()
        before = list(app.cfg.saved_ignores[0]['path'])
        _drag(app, (160, 85), (60, 85))  # right side dragged onto the left one
        self.assertEqual(app.cfg.saved_ignores[0]['path'], before)
        self.assertNotIn('ignore_edit', [a[0] for a in app.cfg.action_history if isinstance(a, tuple)])

    def test_click_without_moving_changes_nothing(self):
        app = self._rect()
        history = list(app.cfg.action_history)
        _drag(app, (160, 120), (160, 120))
        self.assertEqual(app.cfg.action_history, history)
        self.assertEqual(len(app.cfg.saved_ignores), 1)

    def test_click_in_the_middle_still_deletes(self):
        app = self._rect()
        _click(app, 110, 85)
        self.assertEqual(app.cfg.saved_ignores, [])

    def test_regions_loaded_from_json_are_editable_rectangles(self):
        app = _rect_app()
        loaded = app._build_loaded_ignore([(160, 120), (60, 120), (60, 50), (160, 50)])
        app.cfg.saved_ignores = [loaded]
        app.process_keypress(ord('i'))
        _drag(app, (160, 85), (200, 85))
        self.assertTrue(abs(self._bounds(app)[2] - 200) <= 3)
        self.assertTrue(app._is_axis_rectangle(app.cfg.saved_ignores[0]['path']))

    def test_handles_drawn_only_with_the_tool_on(self):
        app = self._rect()
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_ignore_regions(win, 22)
        cx, cy = app.transform_real_to_window_coords(60, 50)
        self.assertTrue((win[cy, cx] > 0).any())
        app.process_keypress(ord('c'))
        win2 = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_ignore_regions(win2, 22)
        self.assertLess(int((win2 > 0).sum()), int((win > 0).sum()), "fewer marks without the handles")


class TestIgnoreExport(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_ignore_")
        self.app = _poly_app(self.tmpdir)
        self.app.cfg.modalita_scelta = "2"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _cleanup_real_module_dir_artifacts()

    def _export(self):
        with mock.patch("smart_segmentation.resolve_script_dir", return_value=self.tmpdir), \
             mock.patch.object(self.app, "_play_save_beep"), _MockedHighGui():
            self.app.export_labelme_format()

    def _mask_path(self):
        return os.path.join(self.tmpdir, "Binary files", "wall-ignore_mask.png")

    def test_json_mask_and_reload_round_trip(self):
        _draw_square(self.app)
        self._export()
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            shapes = json.load(fh)["shapes"]
        ignores = [s for s in shapes if s["label"].startswith("ignore")]
        self.assertEqual(len(ignores), 1)
        self.assertEqual(ignores[0]["label"], "ignore_wall.png")
        self.assertEqual(ignores[0]["shape_type"], "polygon")
        self.assertEqual(ignores[0]["flags"], {"ignore": True})

        mask = cv2.imread(self._mask_path(), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (H, W))
        self.assertEqual(mask[90, 90], 255)
        self.assertEqual(mask[170, 220], 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "Binary files", "wall-detachment_mask.png")))

        self.app.cfg.saved_ignores = []
        self.app.load_labelme_format()
        self.assertEqual(len(self.app.cfg.saved_ignores), 1)
        self.assertEqual(self.app.cfg.saved_detachments, [], "an ignore polygon is never reloaded as a detachment")
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_stale_ignore_mask_is_removed_when_no_region_is_left(self):
        _draw_square(self.app)
        self._export()
        self.assertTrue(os.path.exists(self._mask_path()))
        self.app.cfg.saved_ignores = []
        self._export()
        self.assertFalse(os.path.exists(self._mask_path()))

    def test_mask_export_can_be_switched_off(self):
        self.app.cfg.IGNORE_EXPORT_MASK = False
        _draw_square(self.app)
        self._export()
        self.assertFalse(os.path.exists(self._mask_path()))

    def test_ignore_mask_follows_building_group_rename(self):
        folder = os.path.join(self.tmpdir, "Binary files")
        os.makedirs(folder, exist_ok=True)
        open(os.path.join(folder, "wall-ignore_mask.png"), "wb").close()
        self.app._rename_photo_outputs("wall.png", "wall__BLDG002.png")
        self.assertTrue(os.path.exists(os.path.join(folder, "wall__BLDG002-ignore_mask.png")))


class TestIgnoreElsewhere(unittest.TestCase):

    def test_import_projection_skips_ignore_shapes(self):
        app = _poly_app()
        shapes = [{"label": "ignore_old.png", "points": [list(p) for p in SQUARE],
                   "shape_type": "polygon", "flags": {"ignore": True}}]
        projected, rejected = app._project_all_shapes(shapes, np.eye(3), W, H)
        self.assertEqual((projected, rejected), ([], 0))

    def test_import_keeps_the_current_photo_ignores(self):
        app = _poly_app()
        _draw_square(app)
        kept = [s for s in app._collect_current_session_shapes() if s["label"] == "ignore"]
        self.assertEqual(len(kept), 1)

    def test_validation_does_not_count_ignores_as_detachments(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_ignore_tv_")
        try:
            path = os.path.join(tmpdir, "wall.json")
            shapes = [{"label": "ignore_wall.png", "points": [list(p) for p in SQUARE],
                       "shape_type": "polygon", "flags": {"ignore": True}},
                      {"label": "detachment_wall.png", "points": [list(p) for p in SQUARE],
                       "shape_type": "polygon", "flags": {}}]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"shapes": shapes}, fh)
            self.assertEqual(tv.photo_quality_from_json(path)["detachments"], 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_validation_moves_the_ignore_mask_with_its_photo(self):
        self.assertIn("-ignore_mask", tv.BINARY_EXPORT_SUFFIXES)


class TestIgnoreGui(unittest.TestCase):

    def test_sidebar_and_menu_have_the_ignore_button(self):
        from test_gui_panel_wiring import gm
        sidebar = {code: (label, key) for _, label, key, code in gm._SIDEBAR_SHORTCUTS}
        self.assertEqual(sidebar[ord('i')], ("Ignore region", "I"))
        menu_codes = [code for _, commands in gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(ord('i'), menu_codes)

    def test_options_dialog_returns_the_choices(self):
        from test_gui_panel_wiring import gm, make_app
        app = make_app(self)
        current = {"shape": "square", "edges": {"left": True, "right": False, "top": False, "bottom": True}}
        dialog = gm._IgnoreOptionsDialog(app._tk_root, current)
        self.assertEqual(dialog.shape_var.get(), "square", "opens on the current choice")
        self.assertTrue(dialog.edge_vars["bottom"].get())
        dialog.shape_var.set("rectangle")
        dialog.edge_vars["right"].set(True)
        dialog._ok()
        self.assertEqual(dialog.result, {"shape": "rectangle",
                                         "edges": {"left": True, "right": True, "top": False, "bottom": True}})
        dialog = gm._IgnoreOptionsDialog(app._tk_root, current)
        dialog._cancel()
        self.assertIsNone(dialog.result)

    def test_gui_i_key_goes_through_the_dialog(self):
        from test_gui_panel_wiring import make_app
        app = make_app(self)
        # The fake wait_window() returns at once, so the dialog closes with no choice: the tool stays unchanged.
        app.process_keypress(ord('i'))
        self.assertEqual(app.cfg.current_tool, 'crack')
        choice = {"shape": "rectangle", "edges": {"bottom": True}}
        with mock.patch.object(app, "_prompt_ignore_options", return_value=choice):
            app.process_keypress(ord('i'))
        self.assertEqual(app.cfg.current_tool, 'ignore')

    def test_sidebar_button_looks_pressed_while_the_tool_is_on(self):
        from test_gui_panel_wiring import gm, make_app
        app = make_app(self)
        app._prompt_ignore_options = lambda current: current
        app.process_keypress(ord('i'))
        app._sidebar.refresh_states(app.cfg, False)
        self.assertEqual(app._sidebar.buttons_by_code[ord('i')].kw.get("relief"), "sunken")
        app.process_keypress(ord('c'))
        app._sidebar.refresh_states(app.cfg, False)
        self.assertEqual(app._sidebar.buttons_by_code[ord('i')].kw.get("relief"), "raised")


if __name__ == "__main__":
    unittest.main()
