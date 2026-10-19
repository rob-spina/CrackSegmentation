"""test_v120_features.py

Headless tests for 1.2.0:
  A. Interrupted crack (interrupted_trace.py): a Crack-tool trace whose route crosses a stretch with no crack
     is stored as separate cracks (one per continuous piece, without the gap), an on-screen banner says so,
     a single [U] undoes it, and [S] writes the pieces as separate JSON shapes with the gap black in the PNG mask.
  B. Hideable sidebars (crack_segmentation_gui.py): the rectangle button of each sidebar hides it (a thin strip
     with the button is left to bring it back), and the photo grows into the freed space.

Run with:  python3 -m unittest test_v120_features -v
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

import uncertain_trace as ut
from test_v105_link import _two_crack_app


def _good_quality(path):
    return 90.0, 0.9, False


def _set_crack_masks(app, segments):
    """Photo, skeleton and binary mask holding only the given horizontal crack segments [(x0, x1)] on y=100."""
    mask = np.zeros((app.cfg.H_img, app.cfg.W_img), dtype=np.uint8)
    img = np.full((app.cfg.H_img, app.cfg.W_img, 3), 200, dtype=np.uint8)
    for x0, x1 in segments:
        cv2.line(mask, (x0, 100), (x1, 100), 255, 1)
        cv2.line(img, (x0, 100), (x1, 100), (30, 30, 30), 1)
    app.cfg.skeleton_mask = mask
    app.cfg.binary_mask = mask.copy()
    app.cfg.img_original = img


def _wide_crack_photo(app, segments, width=15):
    """A photo with WIDE dark cracks (+ noise); masks computed by the app itself, as on a real photo."""
    rng = np.random.default_rng(0)
    img = np.full((app.cfg.H_img, app.cfg.W_img), 190, dtype=np.uint8)
    for x0, x1 in segments:
        cv2.rectangle(img, (x0, 100 - width // 2), (x1, 100 + width // 2), 60, -1)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    app.cfg.img_original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    app._compute_binary_and_skeleton_masks()


class _TraceCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v120_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        self.app.cfg.saved_cracks = []
        self.app.cfg.current_tool = 'crack'
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trace(self, start=(20, 100), end=(280, 100)):
        self.app.cfg.temp_start = start
        self.app.cfg.action_history.append('crack_start')
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._complete_new_crack_trace(end)
        return self.app.cfg.saved_cracks

    def _key(self, code):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(code)

    def _json_cracks(self):
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            return [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]

    def _mask(self, suffix):
        return cv2.imread(os.path.join(self.tmpdir, "Binary files", f"wall{suffix}.png"), cv2.IMREAD_GRAYSCALE)


class TestInterruptedCrackSplit(_TraceCase):

    def test_trace_across_a_gap_becomes_two_cracks(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        cracks = self._trace()
        self.assertEqual(len(cracks), 2)
        first, second = cracks
        self.assertEqual(first['path'][0], (20, 100))
        self.assertLessEqual(first['path'][-1][0], 121)
        self.assertGreaterEqual(second['path'][0][0], 159)
        self.assertEqual(second['path'][-1], (280, 100))
        for c in cracks:
            self.assertEqual((c['start'], c['end']), (tuple(c['path'][0]), tuple(c['path'][-1])))
            self.assertTrue(c['active'])
        self.assertNotEqual(first['session_id'], second['session_id'])
        self.assertIs(self.app.cfg.last_traced_crack, second)
        self.assertIsNone(self.app.cfg.temp_start)

    def test_on_screen_message(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._trace()
        self.assertTrue(self.app.cfg.interrupted_crack_triggered)
        self.assertIn("INTERRUPTED CRACK", self.app.cfg.interrupted_crack_message)
        self.assertIn("2 separate cracks", self.app.cfg.interrupted_crack_message)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_interrupted_crack_banner(win, self.app._compute_hud_font_metrics())
        self.assertTrue(np.any(win[280]), "the banner is drawn on the window")

    def test_banner_hides_itself_and_with_dot_key(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._trace()
        self.app._dismiss_all_banners()
        self.assertFalse(self.app.cfg.interrupted_crack_triggered)
        self._trace()
        hm = self.app._compute_hud_font_metrics()
        self.app.cfg.banner_first_seen['interrupted_crack_triggered'] = 0.0   # long ago
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_interrupted_crack_banner(win, hm)
        self.assertFalse(self.app.cfg.interrupted_crack_triggered)
        self.assertFalse(np.any(win))

    def test_continuous_crack_is_unchanged(self):
        _set_crack_masks(self.app, [(20, 280)])
        cracks = self._trace()
        self.assertEqual(len(cracks), 1)
        self.assertEqual(cracks[0]['start'], (20, 100))
        self.assertEqual(self.app.cfg.action_history[-1], 'crack')
        self.assertFalse(self.app.cfg.interrupted_crack_triggered)

    def test_route_through_the_middle_of_a_wide_crack_is_continuous(self):
        # The crack mask is hollow in the middle of a wide crack: darkness against the wall must count too.
        _wide_crack_photo(self.app, [(20, 280)])
        self.assertLess(self.app.cfg.binary_mask[100, 50:250].mean(), 128, "precondition: hollow mask")
        path = [(x, 100) for x in range(20, 281)]
        self.assertEqual(self.app._find_trace_gaps(path), [])

    def test_gap_between_two_wide_cracks_is_found(self):
        _wide_crack_photo(self.app, [(20, 120), (160, 280)])
        path = [(x, 100) for x in range(20, 281)]
        gaps = self.app._find_trace_gaps(path)
        self.assertEqual(len(gaps), 1)
        first, last = gaps[0]
        self.assertTrue(115 <= path[first][0] <= 130 and 150 <= path[last][0] <= 165, (path[first], path[last]))

    def test_tiny_break_is_not_an_interruption(self):
        _set_crack_masks(self.app, [(20, 140), (145, 280)])   # 4 px hole in the mask
        self.assertEqual(len(self._trace()), 1)

    def test_texture_pixels_inside_the_gap_do_not_hide_it(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self.app.cfg.binary_mask[100, 140] = 255   # an isolated dark speck inside the gap
        self.app.cfg.img_original[100, 140] = (30, 30, 30)
        self.assertEqual(len(self._trace()), 2)

    def test_three_pieces_with_two_gaps(self):
        _set_crack_masks(self.app, [(20, 90), (120, 190), (220, 280)])
        cracks = self._trace()
        self.assertEqual(len(cracks), 3)
        self.assertIn("3 separate cracks", self.app.cfg.interrupted_crack_message)

    def test_off_crack_stretch_at_an_end_is_not_a_gap(self):
        _set_crack_masks(self.app, [(40, 280)])
        app = self.app
        path = [(x, 100) for x in range(20, 281)]   # starts 20 px before the crack
        self.assertEqual(app._find_trace_gaps(path), [])

    def test_can_be_switched_off(self):
        self.app.cfg.SPLIT_INTERRUPTED_CRACKS = False
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self.assertEqual(len(self._trace()), 1)

    def test_no_crack_mask_no_split(self):
        self.app.cfg.binary_mask = None
        self.assertEqual(self.app._interrupted_trace_pieces([(x, 100) for x in range(20, 281)]),
                         [[(x, 100) for x in range(20, 281)]])

    def test_one_undo_removes_both_pieces_and_redo_restores_them(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._trace()
        self._key(ord('u'))
        self.assertEqual(self.app.cfg.saved_cracks, [])
        self._key(ord('r'))
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_not_saved_before_save_button(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._trace()
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH), "written by [S], as any crack")

    def test_saved_as_separate_json_shapes_and_gap_left_out_of_png(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._trace()
        self.app._save_annotations_now("TEST", None)
        shapes = self._json_cracks()
        self.assertEqual(len(shapes), 2)
        xs = sorted((min(p[0] for p in s["points"]), max(p[0] for p in s["points"])) for s in shapes)
        self.assertLessEqual(xs[0][1], 121)
        self.assertGreaterEqual(xs[1][0], 159)
        mask = self._mask("-crack_mask")
        self.assertEqual(mask[100, 70], 255)
        self.assertEqual(mask[100, 220], 255)
        self.assertEqual(mask[100, 140], 0, "no crack drawn across the gap")

    def test_uncertain_mode_marks_and_saves_both_pieces(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        self._key(ut.UNCERTAIN_TRACE_CODE)
        cracks = self._trace()
        self.assertEqual(len(cracks), 2)
        self.assertTrue(all(c.get('uncertain_manual') for c in cracks))
        shapes = self._json_cracks()
        self.assertEqual(len(shapes), 2, "saved right away, as every uncertain trace")
        self.assertTrue(all(s["flags"].get("uncertain_manual") for s in shapes))
        self.assertIn("saved now", self.app.cfg.interrupted_crack_message)

    def test_guided_trace_through_waypoint_is_split_too(self):
        _set_crack_masks(self.app, [(20, 120), (160, 280)])
        route = [(x, 100) for x in range(20, 281)]
        self.app.cfg.trace_waypoints = [(200, 100)]
        with mock.patch.object(self.app, "_guided_route", return_value=route):
            cracks = self._trace()
        self.assertEqual(len(cracks), 2)
        self.assertEqual(self.app.cfg.trace_waypoints, [])


class TestSidebarCollapse(unittest.TestCase):

    def setUp(self):
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)
        self.right, self.left = self.app._sidebar.right, self.app._sidebar.left

    def _shapes(self, panel, kind):
        return [v for v in panel.collapse_button.canvas._items.values() if v.get("kind") == kind]

    def test_each_sidebar_has_a_flat_icon_button(self):
        for panel in (self.right, self.left):
            self.assertIsInstance(panel.collapse_button, self.gm.SidebarToggle)
            self.assertEqual(panel.collapse_button.side, panel.side)
            self.assertFalse(panel.collapsed)
            self.assertEqual(panel.frame.kw.get("width"), self.gm.SIDEBAR_WIDTH)
            self.assertEqual(panel.collapse_button.canvas.kw.get("highlightthickness"), 0, "flat, no 3D border")

    def test_click_on_the_icon_toggles(self):
        self.left.collapse_button.canvas.fire("<ButtonRelease-1>")
        self.assertTrue(self.left.collapsed)

    def test_icon_shows_the_panel_filled_when_open_and_an_arrow_when_hidden(self):
        panel = self.right
        self.assertEqual(len(self._shapes(panel, "rectangle")), 1, "side panel filled while shown")
        panel.toggle_collapsed()
        self.assertEqual(self._shapes(panel, "rectangle"), [], "empty once hidden")
        lines = self._shapes(panel, "line")
        self.assertEqual(len(lines), 2, "divider + arrow")
        # The fill sits on the sidebar's side: right of the divider for the right panel, left for the left one.
        [fill_l] = self._shapes(self.left, "rectangle")
        divider_l = self._shapes(self.left, "line")[0]["coords"][0]
        self.assertLessEqual(fill_l["coords"][2], divider_l)

    def test_hover_highlights(self):
        btn = self.left.collapse_button
        n_at_rest = len(btn.canvas._items)
        btn._on_enter()
        self.assertEqual(len(btn.canvas._items), n_at_rest + 1, "light pill behind the icon")
        [fill] = self._shapes(self.left, "rectangle")
        self.assertEqual(fill["fill"], btn.ACCENT)
        btn._on_leave()
        self.assertEqual(len(btn.canvas._items), n_at_rest)

    def test_left_button_hides_only_the_left_sidebar(self):
        self.left.collapse_button.command()
        self.assertTrue(self.left.collapsed)
        self.assertFalse(self.right.collapsed)
        self.assertEqual(self.left.frame.kw.get("width"), self.gm.SIDEBAR_COLLAPSED_WIDTH)
        self.assertEqual(self.right.frame.kw.get("width"), self.gm.SIDEBAR_WIDTH)
        self.assertEqual(self.app._sidebar.hidden_sides(), ("left",))

    def test_right_button_hides_only_the_right_sidebar(self):
        self.right.collapse_button.command()
        self.assertEqual(self.app._sidebar.hidden_sides(), ("right",))

    def test_both_buttons_hide_both(self):
        self.left.collapse_button.command()
        self.right.collapse_button.command()
        self.assertEqual(sorted(self.app._sidebar.hidden_sides()), ["left", "right"])

    def test_second_click_brings_the_sidebar_back(self):
        self.right.collapse_button.command()
        self.right.collapse_button.command()
        self.assertFalse(self.right.collapsed)
        self.assertEqual(self.right.frame.kw.get("width"), self.gm.SIDEBAR_WIDTH)
        self.assertFalse(self.right.collapse_button.collapsed)

    def test_hidden_panel_content_is_unpacked_and_restored(self):
        panel = self.right
        with mock.patch.object(panel._canvas, "pack_forget") as forget, \
                mock.patch.object(panel.title_label, "pack_forget") as forget_title:
            panel.toggle_collapsed()
        forget.assert_called_once()
        forget_title.assert_called_once()
        self.assertTrue(panel.collapse_button.collapsed)
        with mock.patch.object(panel._canvas, "pack") as pack:
            panel.toggle_collapsed()
        pack.assert_called_once_with(side="left", fill="both", expand=True)

    def test_button_stays_reachable_and_tip_explains(self):
        self.right.toggle_collapsed()
        self.assertIn("Show the right sidebar", self.right._collapse_tip_text())
        self.assertIn("Hide the left sidebar", self.left._collapse_tip_text())

    def test_photo_grows_into_the_freed_width(self):
        canvas = self.app._canvas
        canvas.canvas._test_width, canvas.canvas._test_height = 960, 900
        self.assertAlmostEqual(canvas._compute_display_scale(), 0.8)    # both sidebars on a laptop
        canvas.canvas._test_width = 1600
        canvas.canvas._test_height = 1200
        self.assertAlmostEqual(canvas._compute_display_scale(), 1600 / 1200.0)
        canvas.canvas._test_height = 700   # wider but short: no larger than the frame (height scrolls)
        self.assertEqual(canvas._compute_display_scale(), 1.0)
        canvas.canvas._test_width, canvas.canvas._test_height = 5000, 5000
        self.assertEqual(canvas._compute_display_scale(), canvas.MAX_DISPLAY_SCALE)

    def test_clicks_map_back_when_enlarged(self):
        canvas = self.app._canvas
        canvas.canvas._test_width, canvas.canvas._test_height = 1800, 1350   # scale 1.5
        canvas.update_frame(np.zeros((900, 1200, 3), dtype=np.uint8))
        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            canvas.canvas.fire("<Button-1>", x=900, y=675)
        self.assertEqual(calls[0][1:3], (600, 450))


if __name__ == "__main__":
    unittest.main()
