"""test_v200_sam_accept.py

Headless tests for the 2.0.0 [SAM click to accept] mode (sam_accept.py, code 138) and the crack-map cache:
  - the button computes the crack map once and adds nothing; a click near a crack adds the connected part of the
    map ("source": "sam_click", never uncertain, faint stretches too), JSON and masks written after every click;
  - a click far from any crack, or on a crack already there, adds nothing; [U] undoes the last click;
  - another command or a new photo ends the mode; [SAM auto-segment] keeps the accepted cracks;
  - the crack map is cached per model and photo, reused by both buttons and written by sam_batch.py --prob-only,
    whose report ranks the photos by uncertainty.
The model is the stand-in script of test_v200_sam.py (no PyTorch needed).

Run with:  python3 -m unittest test_v200_sam_accept -v
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
import fake_tkinter  # noqa: F401

import crack_segmentation_gui as gm
import sam_accept
import sam_autoseg
import sam_batch
from sam_accept import SAM_ACCEPT_CODE
from test_v200_sam import _sam_app, _press, H, W


def _json_cracks(app):
    with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
        return [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = _sam_app(self.tmp)
        self.app.cfg.SAM_ACCEPT_AUTO = False          # the click tests start from an empty photo
        self.app.cfg.SAM_ACCEPT_SHOW_CANDIDATES = True
        self.app.cfg.SAM_ACCEPT_SAVE_EACH_CHANGE = True   # these tests read the JSON after every click
        self._pump = mock.patch.object(self.app, "_pump_extra_events")
        self._pump.start()

    def tearDown(self):
        self._pump.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _on(self):
        _press(self.app, SAM_ACCEPT_CODE)
        self.assertTrue(self.app.cfg.sam_accept_state["active"])

    def _click(self, x, y):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            return self.app._accept_sam_crack_at(x, y)


class TestAcceptRegion(unittest.TestCase):

    def test_component_under_click(self):
        prob = np.zeros((100, 100), np.uint8)
        cv2.line(prob, (10, 50), (90, 50), 200, 2)
        cv2.line(prob, (10, 20), (90, 20), 200, 2)
        region, seed = sam_accept.accept_region(prob, 40, 55, 0.25, 8)
        self.assertIn(seed[1], (49, 50, 51))
        self.assertTrue(region[50, 80])
        self.assertFalse(region[20, 50])   # the other crack is not taken

    def test_nothing_near(self):
        prob = np.zeros((100, 100), np.uint8)
        cv2.line(prob, (10, 50), (90, 50), 200, 2)
        self.assertEqual(sam_accept.accept_region(prob, 40, 80, 0.25, 8), (None, "nothing"))

    def test_faint_stretch_joined(self):
        prob = np.zeros((100, 100), np.uint8)
        cv2.line(prob, (10, 50), (50, 50), 220, 2)
        cv2.line(prob, (50, 50), (90, 50), 80, 2)    # faint (0.31) but connected
        region, _ = sam_accept.accept_region(prob, 20, 50, 0.25, 8)
        self.assertTrue(region[50, 85])

    def test_taken(self):
        prob = np.zeros((100, 100), np.uint8)
        cv2.line(prob, (10, 50), (90, 50), 200, 2)
        blocked = np.zeros_like(prob)
        blocked[40:60, :] = 255
        self.assertEqual(sam_accept.accept_region(prob, 40, 50, 0.25, 8, blocked), (None, "taken"))


class TestClickToAccept(_Base):

    def test_button_adds_nothing(self):
        self._on()
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_click_accepts_crack_and_branch(self):
        self._on()
        self.assertTrue(self._click(150, 104))
        cracks = self.app.cfg.saved_cracks
        self.assertEqual(len(cracks), 2)   # main line + branch: one connected region
        self.assertTrue(all(c["source"] == "sam_click" and not c.get("uncertain_manual") for c in cracks))
        shapes = _json_cracks(self.app)
        self.assertEqual([s["source"] for s in shapes], ["sam_click", "sam_click"])
        mask = cv2.imread(os.path.join(self.tmp, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        expected = np.zeros((H, W), np.uint8)
        cv2.line(expected, (20, 100), (280, 100), 255, 3)
        cv2.line(expected, (150, 100), (150, 180), 255, 3)
        np.testing.assert_array_equal(mask > 0, expected > 0)   # the weak crack was not clicked: not there

    def test_weak_crack_accepted_and_not_uncertain(self):
        self._on()
        self.assertTrue(self._click(100, 43))
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertFalse(self.app.cfg.saved_cracks[0].get("uncertain_manual"))

    def test_click_on_wall_adds_nothing(self):
        self._on()
        self.assertFalse(self._click(80, 160))
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_same_crack_twice(self):
        self._on()
        self._click(60, 100)
        self.assertFalse(self._click(200, 101))
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_undo_last_click(self):
        self._on()
        self._click(60, 100)
        self._click(100, 40)
        self.assertEqual(len(self.app.cfg.saved_cracks), 3)
        _press(self.app, ord('u'))
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)
        self.assertTrue(self.app.cfg.sam_accept_state["active"])   # Undo keeps the mode on

    def test_huge_region_refused(self):
        self._on()
        self.app.cfg.SAM_ACCEPT_MAX_AREA_FRACTION = 0.001
        self.assertFalse(self._click(60, 100))

    def test_operator_crack_left_alone(self):
        own = [(x, 100) for x in range(20, 281)]
        self.app.cfg.saved_cracks = [{'start': own[0], 'end': own[-1], 'path': own, 'active': True, 'session_id': 'x'}]
        self._on()
        self.assertTrue(self._click(150, 150))   # the branch only: the main line is already the operator's
        branch = self.app.cfg.saved_cracks[1]
        self.assertTrue(all(p[1] > 102 for p in branch["path"]))

    def test_toggle_and_other_command_end_the_mode(self):
        self._on()
        _press(self.app, SAM_ACCEPT_CODE)
        self.assertFalse(self.app.cfg.sam_accept_state["active"])
        self._on()
        _press(self.app, ord('c'))
        self.assertFalse(self.app.cfg.sam_accept_state["active"])

    def test_new_photo_ends_the_mode(self):
        self._on()
        self.app.cfg.CURRENT_IMAGE_PATH = os.path.join(self.tmp, "Images", "other.png")
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(60, 100)):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 1, 1)
        self.assertFalse(self.app.cfg.sam_accept_state["active"])
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_mouse_click_goes_through_window_coordinates(self):
        self._on()
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(60, 101)) as tr, \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 5, 6)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, 5, 6)
        tr.assert_called_with(5, 6)
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_auto_segment_keeps_accepted_cracks(self):
        self._on()
        self._click(60, 100)
        _press(self.app, SAM_ACCEPT_CODE)
        _press(self.app, sam_autoseg.SAM_AUTOSEG_CODE)
        sources = sorted(c.get("source") for c in self.app.cfg.saved_cracks)
        self.assertEqual(sources, ["sam", "sam_click", "sam_click"])   # only the weak crack was new


class TestCandidates(_Base):

    def _scene(self):
        return np.full((H, W, 3), 190, np.uint8)

    def test_tint_where_the_model_sees_cracks(self):
        self._on()
        scene = self._scene()
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        self.assertFalse(np.array_equal(scene[100, 60], [190, 190, 190]))   # strong crack: tinted
        self.assertFalse(np.array_equal(scene[40, 100], [190, 190, 190]))   # weak crack: tinted too
        np.testing.assert_array_equal(scene[160, 80], [190, 190, 190])      # wall: untouched
        strong = np.abs(scene[100, 60].astype(int) - 190).sum()
        weak = np.abs(scene[40, 100].astype(int) - 190).sum()
        self.assertGreater(strong, weak)

    def test_zoomed_region(self):
        self._on()
        scene = self._scene()[90:110, 50:70].copy()
        self.app._paint_sam_candidates(scene, (90, 110, 50, 70))
        self.assertFalse(np.array_equal(scene[10, 10], [190, 190, 190]))   # photo (60, 100)

    def test_hidden_and_mode_off(self):
        self._on()
        _press(self.app, sam_accept.SAM_CANDIDATES_CODE)
        self.assertTrue(self.app.cfg.sam_accept_state["active"])   # the toggle keeps the mode on
        scene = self._scene()
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        self.assertTrue((scene == 190).all())
        _press(self.app, sam_accept.SAM_CANDIDATES_CODE)
        _press(self.app, SAM_ACCEPT_CODE)   # mode off
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        self.assertTrue((scene == 190).all())

    def test_shift_click_on_button(self):
        self.assertEqual(gm._SIDEBAR_SHIFT_CLICK_CODES[SAM_ACCEPT_CODE], sam_accept.SAM_CANDIDATES_CODE)


class TestColours(_Base):
    """v2.0.1: every crack the model sees is coloured at once by how sure the model is; H/G/A/L/D."""

    def _scene(self):
        scene = np.full((H, W, 3), 190, np.uint8)
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        return scene

    def _cls(self, x, y):
        state = self.app.cfg.sam_accept_state
        return int(state["classes"][state["labels"][y, x]])

    def _click(self, x, y):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            return self.app._accept_sam_crack_at(x, y)

    def _tinted(self, cls):
        return tuple(int(round(190 * 0.3 + c * 0.7)) for c in sam_accept.CLASS_COLORS[cls])

    def test_classes_and_colours_at_start(self):
        self._on()
        self.assertEqual(self._cls(60, 100), sam_accept.CLASS_SURE)    # 0.9: orange
        self.assertEqual(self._cls(100, 40), sam_accept.CLASS_DARK)    # 0.55: dark yellow
        scene = self._scene()
        self.assertEqual(tuple(int(v) for v in scene[100, 60]), self._tinted(sam_accept.CLASS_SURE))
        self.assertEqual(tuple(int(v) for v in scene[40, 100]), self._tinted(sam_accept.CLASS_DARK))
        self.assertEqual(self.app.cfg.saved_cracks, [])               # nothing saved
        self.app.cfg.SAM_ACCEPT_DARK_MIN_MEAN = 0.6
        self.app._classify_sam_regions()
        self.assertEqual(self._cls(100, 40), sam_accept.CLASS_LIGHT)

    def test_h_g_a_toggle_colours_and_keep_the_mode(self):
        self._on()
        hud = self.app.cfg.hud_large_size
        _press(self.app, ord('g'))                                     # yellow hidden
        scene = self._scene()
        self.assertEqual(tuple(scene[40, 100]), (190, 190, 190))
        self.assertNotEqual(tuple(scene[100, 60]), (190, 190, 190))
        _press(self.app, ord('A'))                                     # orange hidden too
        self.assertTrue((self._scene() == 190).all())
        _press(self.app, ord('h'))                                     # H: all back
        scene = self._scene()
        self.assertNotEqual(tuple(scene[40, 100]), (190, 190, 190))
        self.assertNotEqual(tuple(scene[100, 60]), (190, 190, 190))
        _press(self.app, ord('H'))                                     # H: all hidden
        self.assertTrue((self._scene() == 190).all())
        self.assertEqual(self.app.cfg.hud_large_size, hud)            # not the HUD size
        self.assertTrue(self.app.cfg.sam_accept_state["active"])
        self.assertTrue(self._click(60, 101))                          # clicks still accept
        _press(self.app, SAM_ACCEPT_CODE)                              # mode off: the letters are themselves again
        _press(self.app, ord('h'))
        self.assertNotEqual(self.app.cfg.hud_large_size, hud)

    def test_l_and_d_discard_without_saving(self):
        self.app.cfg.SAM_ACCEPT_DARK_MIN_MEAN = 0.6                    # the weak crack (0.55) is light yellow
        self.app.cfg.SAM_ACCEPT_AUTO_MIN_MEAN = 0.95                   # the strong one (0.9) dark yellow
        self._on()
        _press(self.app, ord('l'))
        scene = self._scene()
        self.assertEqual(tuple(scene[40, 100]), (190, 190, 190))       # light yellow gone
        self.assertNotEqual(tuple(scene[100, 60]), (190, 190, 190))
        self.assertFalse(self.app._update_sam_hover(100, 41) and self.app.cfg.sam_accept_state["hover"])
        _press(self.app, ord('D'))
        self.assertTrue((self._scene() == 190).all())
        self.assertEqual(self.app.cfg.saved_cracks, [])
        self.assertTrue(self.app.cfg.sam_accept_state["active"])
        self.assertEqual(self.app.cfg.current_tool, "crack")           # D did not choose the detachment tool

    def test_delete_by_number_in_the_mode(self):
        """v2.0.1: after Ctrl+click (accept every orange) and [Space], a click on a crack's number deletes it."""
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
        cfg = self.app.cfg
        self.assertEqual(len(cfg.saved_cracks), 2)
        self.assertEqual(tuple(self._scene()[100, 60]), (190, 190, 190))   # accepted: no longer coloured
        cfg.show_markers = True
        start = cfg.saved_cracks[0]["start"]
        # A 4K photo seen whole: 1 screen px = 4 photo px; the click lands 15 screen px off the number box centre.
        to_screen = lambda x, y: (int(x) // 4, int(y) // 4)
        with mock.patch.object(self.app, "transform_real_to_window_coords", side_effect=to_screen), \
                mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(5, 5)), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            sx, sy = to_screen(*start)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, sx + 15, sy - 10)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, sx + 15, sy - 10)
        self.assertEqual(len(cfg.saved_cracks), 1)
        self.assertEqual(len(_json_cracks(self.app)), 1)                    # saved at once
        self.assertTrue(cfg.sam_accept_state["active"])

    def test_numbers_hidden_click_accepts_instead_of_deleting(self):
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
        cfg = self.app.cfg
        cfg.show_markers = False
        start = cfg.saved_cracks[0]["start"]
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=start), \
                mock.patch.object(self.app, "transform_real_to_window_coords", side_effect=lambda x, y: (x, y)), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, *start)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, *start)
        self.assertEqual(len(cfg.saved_cracks), 2)

    def test_undo_colours_the_crack_again(self):
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
        self.assertEqual(tuple(self._scene()[100, 60]), (190, 190, 190))
        _press(self.app, ord('u'))
        self.assertEqual(tuple(int(v) for v in self._scene()[100, 60]), self._tinted(sam_accept.CLASS_SURE))

    def test_overlay_built_once_while_the_mouse_moves(self):
        self._on()
        self._scene()
        with mock.patch.object(self.app, "_refresh_sam_done_labels") as refresh:
            for x in range(40, 120, 5):
                self.app._update_sam_hover(x, 101)
                self._scene()
        refresh.assert_not_called()
        overlay = self.app.cfg.sam_accept_state["overlay"]
        self._scene()
        self.assertIs(self.app.cfg.sam_accept_state["overlay"], overlay)

    def test_saved_with_the_photo_not_after_every_click(self):
        """v2.0.1 default: no JSON / masks written per click; saved when the mode is turned off or on navigation."""
        cfg = self.app.cfg
        cfg.SAM_ACCEPT_SAVE_EACH_CHANGE = False
        self._on()
        with mock.patch.object(self.app, "_save_annotations_now") as save:
            self.assertTrue(self._click(60, 101))
            save.assert_not_called()
            self.assertTrue(cfg.sam_accept_state["unsaved"])
            _press(self.app, SAM_ACCEPT_CODE)                     # mode off: saved once
            save.assert_called_once()
        self._on()
        with mock.patch.object(self.app, "_save_annotations_now") as save:
            self._click(100, 41)
            self.app._handle_sam_accept_key(4)                    # Next without saving: kept anyway
            save.assert_called_once()
        self._on()
        with mock.patch.object(self.app, "_save_annotations_now") as save:
            _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
            self.app._handle_sam_accept_key(13)                   # Enter saves the photo itself
            save.assert_not_called()

    def test_accept_dark_light_and_every_colour(self):
        self.app.cfg.SAM_ACCEPT_DARK_MIN_MEAN = 0.6      # weak (0.55) -> light yellow
        self.app.cfg.SAM_ACCEPT_AUTO_MIN_MEAN = 0.95     # strong (0.9) -> dark yellow
        self._on()
        cfg = self.app.cfg
        _press(self.app, ord('3'))                       # light yellow only
        self.assertTrue(cfg.saved_cracks and all(abs(p[1] - 40) <= 3 for c in cfg.saved_cracks for p in c["path"]))
        n = len(cfg.saved_cracks)
        _press(self.app, sam_accept.SAM_ACCEPT_DARK_CODE)   # the sidebar button
        self.assertGreater(len(cfg.saved_cracks), n)
        _press(self.app, ord('u'))
        _press(self.app, ord('u'))
        self.assertEqual(cfg.saved_cracks, [])
        _press(self.app, ord('4'))                       # every colour
        self.assertEqual(len(cfg.saved_cracks), 3)
        self.assertTrue(all(c["source"] == "sam_click" for c in cfg.saved_cracks))
        self.assertTrue((self._scene() == 190).all())    # nothing coloured is left

    def test_exclude_removes_an_accepted_crack_from_json_and_png(self):
        cfg = self.app.cfg
        cfg.SAM_ACCEPT_SAVE_EACH_CHANGE = False
        self._on()
        _press(self.app, ord('4'))
        self.app._save_annotations_now("TEST", None)
        self.assertEqual(len(_json_cracks(self.app)), 3)
        _press(self.app, ord('x'))
        self.assertTrue(cfg.sam_accept_state["exclude"])
        to_screen = lambda x, y: (x, y)
        with mock.patch.object(self.app, "transform_window_to_real_coords", side_effect=to_screen), \
                mock.patch.object(self.app, "transform_real_to_window_coords", side_effect=to_screen), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 100, 43)   # on the weak crack (y=40)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, 101, 43)     # no window: a click
        self.assertEqual(len(cfg.saved_cracks), 2)
        self.assertEqual(len(_json_cracks(self.app)), 2)                       # saved at once
        mask = cv2.imread(os.path.join(self.tmp, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertFalse(mask[38:43, 60:190].any())                            # gone from the PNG too
        self.assertTrue(mask[98:103, 30:140].any())
        self.assertTrue(cfg.sam_accept_state["exclude"])                        # stays on for the next one
        _press(self.app, ord('X'))
        self.assertFalse(cfg.sam_accept_state["exclude"])

    def test_exclude_on_a_coloured_crack_drops_it(self):
        self._on()
        cfg = self.app.cfg
        _press(self.app, sam_accept.SAM_EXCLUDE_CODE)
        self.app._update_sam_hover(100, 41)
        with mock.patch.object(self.app, "transform_window_to_real_coords", side_effect=lambda x, y: (x, y)), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 100, 41)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, 100, 41)
        self.assertEqual(cfg.saved_cracks, [])
        self.assertEqual(tuple(self._scene()[40, 100]), (190, 190, 190))

    def _drag(self, x0, y0, x1, y1):
        same = lambda x, y: (x, y)
        with mock.patch.object(self.app, "transform_window_to_real_coords", side_effect=same), \
                mock.patch.object(self.app, "transform_real_to_window_coords", side_effect=same), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0)
            self.app._handle_sam_accept_mouse(cv2.EVENT_MOUSEMOVE, x1, y1)
            frame = np.zeros((H, W, 3), np.uint8)
            self.app._draw_exclude_window(frame)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, x1, y1)
        return frame

    def test_exclude_window_cuts_accepted_and_coloured_cracks(self):
        cfg = self.app.cfg
        self._on()
        _press(self.app, ord('1'))                       # the strong crack (y=100) accepted, the weak one coloured
        self.app._save_annotations_now("TEST", None)
        n_before = len(_json_cracks(self.app))
        _press(self.app, ord('x'))
        frame = self._drag(80, 30, 120, 110)             # a window over both, 40 x 80
        self.assertTrue((frame == np.array(sam_accept.EXCLUDE_OUTLINE, np.uint8)).all(axis=2).any())
        mask = cv2.imread(os.path.join(self.tmp, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertFalse(mask[95:106, 82:118].any())                       # the accepted crack cut there
        self.assertTrue(mask[95:106, 30:70].any())                         # the rest of it kept
        self.assertGreaterEqual(len(_json_cracks(self.app)), n_before)    # split in pieces, saved at once
        scene = self._scene()
        self.assertEqual(tuple(scene[40, 100]), (190, 190, 190))           # the coloured one blanked inside
        self.assertNotEqual(tuple(scene[40, 70]), (190, 190, 190))         # and kept outside
        self.assertFalse(self.app._accept_sam_crack_at(100, 41))           # nothing to accept there any more
        _press(self.app, ord('u'))                                         # Undo brings the accepted part back
        self.assertTrue(any(81 <= p[0] <= 119 and abs(p[1] - 100) <= 3 for c in cfg.saved_cracks
                            for p in c["path"]))

    def test_window_limited_to_12000_px(self):
        self._on()
        _press(self.app, ord('x'))
        self.app.cfg.sam_accept_state["window_drag"] = (0, 0, 0, 0)
        self.assertEqual(self.app._window_drag_rect(15000, -13000), (0, 0, 12000, -12000))
        self.app.cfg.SAM_WINDOW_MAX_BOX_PX = 50
        self.assertEqual(self.app._window_drag_rect(80, 20), (0, 0, 50, 20))
        self.app.cfg.sam_accept_state["window_drag"] = None

    def test_tiny_window_is_a_click(self):
        self._on()
        _press(self.app, ord('x'))
        with mock.patch.object(self.app, "_exclude_at") as click, \
                mock.patch.object(self.app, "_exclude_in_window") as window:
            self._drag(50, 50, 55, 70)                   # 5 px wide: below the 10 px minimum
        click.assert_called_once_with(50, 50)
        window.assert_not_called()

    def test_accept_window_takes_only_what_is_inside(self):
        """Without Exclude, a dragged window accepts the coloured cracks inside it (the part inside)."""
        cfg = self.app.cfg
        self._on()
        frame = self._drag(80, 30, 140, 50)              # over the weak crack only (y=40), x 80..140
        self.assertTrue((frame == np.array(sam_accept.HOVER_OUTLINE, np.uint8)).all(axis=2).any())   # cyan
        self.assertTrue(cfg.saved_cracks)
        xs = [p[0] for c in cfg.saved_cracks for p in c["path"]]
        self.assertTrue(all(abs(p[1] - 40) <= 4 for c in cfg.saved_cracks for p in c["path"]))
        self.assertGreaterEqual(min(xs), 79)
        self.assertLessEqual(max(xs), 141)
        self.assertTrue(all(c["source"] == "sam_click" for c in cfg.saved_cracks))
        scene = self._scene()
        self.assertEqual(tuple(scene[40, 110]), (190, 190, 190))           # accepted: no longer coloured
        self.assertNotEqual(tuple(scene[40, 70]), (190, 190, 190))         # outside the window: still coloured
        self.assertNotEqual(tuple(scene[100, 60]), (190, 190, 190))        # the strong crack untouched
        _press(self.app, ord('u'))
        self.assertEqual(cfg.saved_cracks, [])

    def test_accept_window_skips_hidden_colours(self):
        self._on()
        _press(self.app, ord('g'))                       # yellow hidden: the weak (dark yellow) crack is not taken
        self._drag(80, 30, 140, 50)
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_small_accept_window_is_a_click(self):
        self._on()
        with mock.patch.object(self.app, "_accept_sam_crack_at", return_value=True) as accept:
            self._drag(60, 100, 63, 104)
        accept.assert_called_once_with(60, 100)

    def test_threshold_recuts_the_map_and_keeps_accepted_cracks(self):
        cfg = self.app.cfg
        self._on()
        self.assertAlmostEqual(cfg.SAM_ACCEPT_LOW_THRESHOLD, 0.25)
        _press(self.app, ord('1'))                       # the strong crack accepted
        accepted = [list(map(tuple, c["path"])) for c in cfg.saved_cracks]
        before = self._sam_counts()
        _press(self.app, ord(']'))                       # raise
        self.assertAlmostEqual(cfg.SAM_ACCEPT_LOW_THRESHOLD, 0.25 + cfg.SAM_THRESHOLD_STEP)
        self.app._set_sam_threshold(0.6)                 # above the weak crack (0.55): it disappears
        self.assertEqual(self._sam_counts(), (0, 0, 0))
        self.assertEqual(tuple(self._scene()[40, 100]), (190, 190, 190))
        self.app._set_sam_threshold(0.25)
        self.assertEqual(self._sam_counts(), before)     # back: the weak crack again, the accepted one not
        self.assertEqual([list(map(tuple, c["path"])) for c in cfg.saved_cracks], accepted)
        self.assertEqual(self.app._set_sam_threshold(5), cfg.SAM_THRESHOLD_MAX)
        self.assertEqual(self.app._set_sam_threshold(-1), cfg.SAM_THRESHOLD_MIN)
        _press(self.app, ord('['))
        self.assertAlmostEqual(cfg.SAM_ACCEPT_LOW_THRESHOLD, cfg.SAM_THRESHOLD_MIN)   # stays at the minimum
        self.assertTrue(cfg.sam_accept_state["active"])

    def test_discarded_cracks_stay_discarded_across_thresholds(self):
        self._on()
        _press(self.app, ord('d'))                       # the weak (dark-yellow) crack discarded
        self.assertEqual(tuple(self._scene()[40, 100]), (190, 190, 190))
        self.app._set_sam_threshold(0.2)
        self.assertEqual(tuple(self._scene()[40, 100]), (190, 190, 190))
        self.assertNotEqual(tuple(self._scene()[100, 60]), (190, 190, 190))

    def _plaster_map(self):
        """A thin crack (3 px) running into a wide patch (missing plaster, 60 x 40 px)."""
        prob = np.zeros((H, W), np.uint8)
        cv2.line(prob, (10, 150), (200, 150), 230, 3)
        cv2.rectangle(prob, (200, 130), (260, 170), 230, -1)
        state = self.app.cfg.sam_accept_state
        state["prob"], state["prob_private"] = prob, True
        self.app.cfg.SAM_ACCEPT_MAX_AREA_FRACTION = 0.2   # the test photo is tiny: the patch is ~5% of it
        return prob

    def test_wide_parts_never_coloured_nor_accepted(self):
        cfg = self.app.cfg
        cfg.SAM_WIDE_AS_CENTERLINE = False               # wide parts hidden (not as a centre line)
        self._on()
        self._plaster_map()
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = 0
        self.app._set_sam_max_width(20)                  # limit on: the patch goes, the thin crack stays
        state = cfg.sam_accept_state
        self.assertEqual(int(state["labels"][150, 230]), 0)              # inside the patch: not a crack region
        self.assertEqual(self._cls(100, 150), sam_accept.CLASS_SURE)    # the thin crack, orange
        scene = self._scene()
        self.assertEqual(tuple(scene[150, 230]), (190, 190, 190))
        self.assertNotEqual(tuple(scene[150, 100]), (190, 190, 190))
        self.assertFalse(self._click(230, 150))                          # a click on the patch takes nothing
        self.assertTrue(self._click(100, 151))
        xs = [p[0] for c in cfg.saved_cracks for p in c["path"]]
        self.assertLess(max(xs), 205)                                    # the crack stops at the patch
        self.app._set_sam_max_width(0)                   # no limit: the patch is a coloured region again
        self.assertNotEqual(int(state["labels"][150, 230]), 0)
        self.assertEqual(self.app._set_sam_max_width(10 ** 6), cfg.SAM_MAX_WIDTH_LIMIT_PX)

    def test_wide_parts_as_a_5_px_orange_centre_line(self):
        cfg = self.app.cfg
        self._on()
        self._plaster_map()
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = 0
        self.app._set_sam_max_width(20)
        state = cfg.sam_accept_state
        column = state["labels"][128:173, 230] > 0       # across the 40 px high patch
        self.assertTrue(column.any())
        self.assertLessEqual(int(column.sum()), cfg.SAM_WIDE_CENTERLINE_PX + 1)   # only the centre line
        self.assertEqual(self._cls(230, 150), sam_accept.CLASS_SURE)             # orange
        self.assertEqual(self._cls(100, 150), sam_accept.CLASS_SURE)             # joined to the thin crack
        self.assertTrue(self._click(230, 150))
        xs = [p[0] for c in cfg.saved_cracks for p in c["path"]]
        self.assertGreater(max(xs), 240)
        crack_mask = self.app._build_variable_width_crack_mask(cfg.saved_cracks, H, W, 0)
        across = int((crack_mask[125:176, 235] > 0).sum())
        self.assertGreaterEqual(across, 38)              # the whole patch, at its REAL width (40 px), in the PNG

    def test_centre_line_only_when_asked(self):
        cfg = self.app.cfg
        cfg.SAM_WIDE_ACCEPT_FULL = False
        self._on()
        self._plaster_map()
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = 0
        self.app._set_sam_max_width(20)
        self.assertTrue(self._click(230, 150))
        crack_mask = self.app._build_variable_width_crack_mask(cfg.saved_cracks, H, W, 0)
        self.assertLessEqual(int((crack_mask[125:176, 235] > 0).sum()), cfg.SAM_WIDE_CENTERLINE_PX + 2)

    def test_window_on_a_centre_line_takes_the_patch_inside_it(self):
        cfg = self.app.cfg
        self._on()
        self._plaster_map()
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = 0
        self.app._set_sam_max_width(20)
        self._drag(225, 100, 245, 199)                   # a window over part of the patch, x 225..245
        crack_mask = self.app._build_variable_width_crack_mask(cfg.saved_cracks, H, W, 0)
        self.assertGreaterEqual(int((crack_mask[125:176, 235] > 0).sum()), 38)  # full width inside the window
        self.assertFalse((crack_mask[130:170, 252:260] > 0).any())               # nothing outside it

    def test_width_limit_toggle_keeps_the_value(self):
        cfg = self.app.cfg
        self._on()
        self._plaster_map()
        self.app._set_sam_max_width(20)
        _press(self.app, ord('0'))                       # limit off: the whole patch is a region
        self.assertFalse(cfg.SAM_WIDTH_LIMIT_ON)
        self.assertEqual(cfg.SAM_ACCEPT_MAX_WIDTH_PX, 20)
        self.assertTrue((cfg.sam_accept_state["labels"][132:168, 230] > 0).all())
        _press(self.app, ord('0'))
        self.assertTrue(cfg.SAM_WIDTH_LIMIT_ON)

    def test_class_thresholds_from_the_dial(self):
        cfg = self.app.cfg
        self._on()
        self.assertEqual(self._cls(100, 40), sam_accept.CLASS_DARK)              # weak crack, mean 0.55
        self.app._set_sam_class_threshold("dark", 0.6)
        self.assertEqual(self._cls(100, 40), sam_accept.CLASS_LIGHT)
        self.app._set_sam_class_threshold("orange", 0.5)                         # orange >= dark: 0.6
        self.assertEqual(cfg.SAM_ACCEPT_AUTO_MIN_MEAN, 0.6)
        self.app._set_sam_class_threshold("dark", 0.5)
        self.app._set_sam_class_threshold("orange", 0.5)
        self.assertEqual(self._cls(100, 40), sam_accept.CLASS_SURE)

    def test_recompute_map_here_replaces_the_cache(self):
        cfg = self.app.cfg
        self._on()
        settings = self.app._load_sam_settings()
        cache = self.app._sam_probability_cache_path(settings)
        new = np.zeros((H, W), np.uint8)
        cv2.line(new, (20, 60), (250, 60), 240, 3)
        with mock.patch.object(self.app, "_sam_run_model_on_photo", return_value=new) as run, \
                mock.patch.object(self.app, "_usable_sam_settings", return_value=settings):
            _press(self.app, sam_accept.SAM_RECOMPUTE_MAP_CODE)
        run.assert_called_once()
        self.assertTrue((cv2.imread(cache, cv2.IMREAD_GRAYSCALE) == new).all())
        self.assertTrue(cfg.sam_accept_state["active"])                  # reopened on the new map
        self.assertTrue((cfg.sam_accept_state["prob"] == new).all())
        with mock.patch.object(self.app, "_sam_run_model_on_photo", return_value=None), \
                mock.patch.object(self.app, "_usable_sam_settings", return_value=settings):
            self.assertFalse(self.app.recompute_sam_map_here())         # failed: the map stays
        self.assertTrue((cv2.imread(cache, cv2.IMREAD_GRAYSCALE) == new).all())

    def test_threshold_feedback(self):
        self._on()
        value, counts, share = self.app.sam_threshold_feedback()
        self.assertAlmostEqual(value, 0.25)
        self.assertEqual(counts[sam_accept.CLASS_SURE], 1)
        self.assertGreater(share, 0)

    def _sam_counts(self):
        c = self.app._sam_class_counts()
        return c[sam_accept.CLASS_SURE], c[sam_accept.CLASS_DARK], c[sam_accept.CLASS_LIGHT]

    def test_hover_outlined_in_cyan(self):
        self._on()
        self.app._update_sam_hover(60, 103)
        scene = self._scene()
        self.assertTrue((scene == np.array(sam_accept.HOVER_OUTLINE, np.uint8)).all(axis=2).any())

    def test_threshold_dial(self):
        import test_gui_panel_wiring as wiring
        app = wiring.make_app(self)
        cfg = app.cfg
        app._open_sam_threshold_dial()
        dial = app._sam_dial
        self.assertIsNotNone(dial)
        for v in (cfg.SAM_THRESHOLD_MIN, 0.3, 0.5, cfg.SAM_THRESHOLD_MAX):
            self.assertAlmostEqual(dial.angle_to_value(dial.value_to_angle(v)), v, places=2)
        self.assertAlmostEqual(dial.angle_to_value(-80), cfg.SAM_THRESHOLD_MAX)   # gap at the bottom: nearer end
        self.assertAlmostEqual(dial.angle_to_value(-100), cfg.SAM_THRESHOLD_MIN)
        with mock.patch.object(app, "_set_sam_threshold") as set_thr:
            dial.step(-1)
            set_thr.assert_called_once()
            self.assertAlmostEqual(set_thr.call_args[0][0], cfg.SAM_ACCEPT_LOW_THRESHOLD - cfg.SAM_THRESHOLD_STEP)
        self.assertIn("Turn SAM click to accept on", dial.counts_label.kw["text"])
        self.assertEqual(dial.limit_button.kw["text"], "limit ON")
        dial.toggle_limit()
        self.assertFalse(cfg.SAM_WIDTH_LIMIT_ON)
        self.assertEqual(dial.limit_button.kw["text"], "limit OFF")
        dial.toggle_centre()
        self.assertEqual(dial.centre_button.kw["text"], "Wider parts: hidden")
        dial.class_step("dark", -1)
        self.assertAlmostEqual(cfg.SAM_ACCEPT_DARK_MIN_MEAN, 0.40)
        self.assertEqual(dial.class_labels["dark"].kw["text"], "0.40")
        app._open_sam_threshold_dial()                  # a second press: the same window
        self.assertIs(app._sam_dial, dial)
        dial.close()
        self.assertIsNone(app._sam_dial)

    def test_gui_translates_only_the_keyboard_letters(self):
        """In the GUI the sidebar's 'd' (Detachment tool) keeps its meaning; the keyboard's D is the mode's."""
        self.assertFalse(gm.GuiCrackSegmentation._SAM_LETTERS_IN_PROCESS)
        import test_gui_panel_wiring as wiring
        app = wiring.make_app(self)
        app.cfg.sam_accept_state["active"] = True
        app._on_keypress(fake_tkinter._Event(keysym="d", char="d"))
        app._on_keypress(fake_tkinter._Event(keysym="h", char="h"))
        app._send(ord('d'))                                            # the sidebar's Detachment tool
        self.assertEqual(app._pending_keys[-3:], [sam_accept.SAM_DISCARD_DARK_CODE, sam_accept.SAM_OVERLAY_CODE,
                                                  ord('d')])
        app.cfg.sam_accept_state["active"] = False
        app._on_keypress(fake_tkinter._Event(keysym="d", char="d"))
        self.assertEqual(app._pending_keys[-1], ord('d'))
        codes = [code for _, code in gm._TOOL_COMMANDS]
        for code in (sam_accept.SAM_OVERLAY_CODE, sam_accept.SAM_YELLOW_CODE, sam_accept.SAM_ORANGE_CODE,
                     sam_accept.SAM_DISCARD_LIGHT_CODE, sam_accept.SAM_DISCARD_DARK_CODE):
            self.assertIn(code, codes)


class TestHoverAndStayOn(_Base):

    def test_hover_picks_the_crack_under_the_mouse(self):
        self._on()
        self.assertTrue(self.app._update_sam_hover(60, 103))
        hover = self.app.cfg.sam_accept_state["hover"]
        self.assertGreater(hover, 0)
        self.assertFalse(self.app._update_sam_hover(70, 102))   # same crack: no redraw
        self.assertTrue(self.app._update_sam_hover(80, 160))    # wall: nothing under the mouse
        self.assertEqual(self.app.cfg.sam_accept_state["hover"], 0)

    def test_mouse_move_updates_hover(self):
        self._on()
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(60, 101)), \
                mock.patch.object(self.app, "refresh_zoom_viewport") as redraw:
            self.app._handle_sam_accept_mouse(cv2.EVENT_MOUSEMOVE, 3, 3)
        redraw.assert_called_once()
        self.assertEqual(self.app.cfg.saved_cracks, [])   # moving never accepts

    def test_stays_on_for_the_next_photo(self):
        self._on()
        self.app._handle_sam_accept_key(13)   # Enter: next photo
        state = self.app.cfg.sam_accept_state
        self.assertFalse(state["active"])
        self.assertTrue(state["sticky"])
        self.app._resume_sam_accept_for_new_photo()
        self.assertTrue(state["active"])

    def test_other_tool_pauses_it_until_the_next_photo(self):
        self._on()
        self.app._handle_sam_accept_key(ord('c'))   # Crack tool, to trace one by hand
        self.assertFalse(self.app.cfg.sam_accept_state["active"])
        self.app._resume_sam_accept_for_new_photo()
        self.assertTrue(self.app.cfg.sam_accept_state["active"])

    def test_button_stops_it(self):
        self._on()
        _press(self.app, SAM_ACCEPT_CODE)   # off by the button
        self.app._resume_sam_accept_for_new_photo()
        self.assertFalse(self.app.cfg.sam_accept_state["active"])

    def test_stay_on_can_be_switched_off(self):
        self.app.cfg.SAM_ACCEPT_STAY_ON = False
        self._on()
        self.app._handle_sam_accept_key(13)
        self.app._resume_sam_accept_for_new_photo()
        self.assertFalse(self.app.cfg.sam_accept_state["active"])


class TestProposals(_Base):
    """v2.0.1: the cracks the model is sure of are only proposed (magenta); only clicked cracks are saved."""

    def setUp(self):
        super().setUp()
        self.app.cfg.SAM_ACCEPT_AUTO = True
        self.app.cfg.SAM_ACCEPT_SHOW_CANDIDATES = False

    def _mask(self):
        path = os.path.join(self.tmp, "Binary files", "wall-crack_mask.png")
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None

    def test_sure_cracks_proposed_but_nothing_saved(self):
        self._on()
        state = self.app.cfg.sam_accept_state
        self.assertEqual(self.app.cfg.saved_cracks, [])
        self.assertEqual(len(state["proposed"]), 1)          # the strong crack + branch; the weak one is not
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH))   # nothing written at all
        mask = self._mask()
        self.assertTrue(mask is None or not mask.any())

    def test_click_on_proposal_saves_only_it(self):
        self._on()
        self.app._update_sam_hover(60, 101)
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(60, 101)), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 1, 1)
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, 1, 1)
        cracks = self.app.cfg.saved_cracks
        self.assertEqual(len(cracks), 2)
        self.assertTrue(all(c["source"] == "sam_click" for c in cracks))
        self.assertEqual(len(_json_cracks(self.app)), 2)
        scene = np.full((H, W, 3), 190, np.uint8)
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        self.assertEqual(tuple(scene[100, 60]), (190, 190, 190))   # no longer proposed

    def test_accept_all(self):
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
        cracks = self.app.cfg.saved_cracks
        self.assertEqual(len(cracks), 2)
        self.assertTrue(all(c["source"] == "sam_click" and not c.get("uncertain_manual") for c in cracks))
        self.assertTrue(all(abs(p[1] - 40) > 3 for c in cracks for p in c["path"]))   # the weak one not taken
        self.assertTrue(self.app.cfg.sam_accept_state["active"])
        _press(self.app, ord('u'))
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_already_traced_crack_not_proposed(self):
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)
        _press(self.app, SAM_ACCEPT_CODE)
        self._on()
        self.assertEqual(self.app.cfg.sam_accept_state["proposed"], set())

    def test_threshold(self):
        self.app.cfg.SAM_ACCEPT_AUTO_MIN_MEAN = 0.95
        self._on()
        self.assertEqual(self.app.cfg.sam_accept_state["proposed"], set())

    def test_old_automatic_cracks_removed_on_request(self):
        cfg = self.app.cfg
        cfg.SAM_ACCEPT_AUTO = False
        self._on()
        _press(self.app, sam_accept.SAM_ACCEPT_ALL_CODE)    # nothing proposed: nothing
        self._click(60, 100)
        for c in cfg.saved_cracks:
            c["source"] = "sam"                              # as v2.0.0's automatic part saved them
        self.app._save_annotations_now("TEST", None)
        _press(self.app, SAM_ACCEPT_CODE)
        with mock.patch.object(self.app, "_confirm_remove_automatic_sam_cracks", return_value=True) as ask:
            self._on()
        ask.assert_called_once_with(2)
        self.assertEqual(cfg.saved_cracks, [])
        self.assertEqual(_json_cracks(self.app), [])
        _press(self.app, SAM_ACCEPT_CODE)
        with mock.patch.object(self.app, "_confirm_remove_automatic_sam_cracks") as ask:
            self._on()                                       # asked once per photo
        ask.assert_not_called()

    def test_old_automatic_cracks_kept_when_refused(self):
        cfg = self.app.cfg
        cfg.SAM_ACCEPT_AUTO = False
        self._on()
        self._click(60, 100)
        for c in cfg.saved_cracks:
            c["source"] = "sam"
        _press(self.app, SAM_ACCEPT_CODE)
        with mock.patch.object(self.app, "_confirm_remove_automatic_sam_cracks", return_value=False):
            self._on()
        self.assertEqual(len(cfg.saved_cracks), 2)


class TestCache(_Base):

    def test_map_computed_once(self):
        self._on()
        _press(self.app, SAM_ACCEPT_CODE)
        with mock.patch.object(self.app, "_sam_run_model_on_photo") as run:
            self._on()
            _press(self.app, SAM_ACCEPT_CODE)
            _press(self.app, sam_autoseg.SAM_AUTOSEG_CODE)
        run.assert_not_called()
        self.assertEqual(len(self.app.cfg.saved_cracks), 3)

    def test_map_found_after_building_tag_rename(self):
        """v2.0.1: a map computed before the photo got __BLDG001 (e.g. on Leonardo) is still used."""
        self._on()
        _press(self.app, SAM_ACCEPT_CODE)
        cfg = self.app.cfg
        settings = self.app._load_sam_settings()
        old_cache = self.app._sam_probability_cache_path(settings)
        base, ext = os.path.splitext(cfg.CURRENT_IMAGE_PATH)
        tagged = base + "__BLDG001" + ext
        os.rename(cfg.CURRENT_IMAGE_PATH, tagged)
        cfg.CURRENT_IMAGE_PATH = tagged
        with mock.patch.object(self.app, "_sam_run_model_on_photo") as run:
            self.assertIsNotNone(self.app._sam_probability_map(settings))
        run.assert_not_called()
        self.assertTrue(os.path.exists(self.app._sam_probability_cache_path(settings)))
        self.assertFalse(os.path.exists(old_cache))

    def test_stale_or_other_size_map_ignored(self):
        path = os.path.join(self.tmp, "m.png")
        cv2.imwrite(path, np.zeros((10, 10), np.uint8))
        photo = self.app.cfg.CURRENT_IMAGE_PATH
        self.assertIsNone(sam_autoseg.read_cached_probability(path, photo, (H, W)))
        os.utime(path, (1, 1))
        self.assertIsNone(sam_autoseg.read_cached_probability(path, photo, (10, 10)))   # older than the photo

    def test_tag_follows_weights_content(self):
        a, b = os.path.join(self.tmp, "best_lora.pth"), os.path.join(self.tmp, "cracksam_vit_h_lora_r4.pth")
        for p in (a, b):
            with open(p, "wb") as f:
                f.write(b"same weights")
        self.assertEqual(sam_autoseg.model_tag(a), sam_autoseg.model_tag(b))
        self.assertEqual(sam_autoseg.model_tag("", "/x/model.pt"), "model")

    def test_uncertainty(self):
        prob = np.zeros((10, 10), np.uint8)
        prob[0, :] = 250
        self.assertEqual(sam_autoseg.probability_uncertainty(prob), 0.0)
        prob[1, :] = 128
        self.assertEqual(sam_autoseg.probability_uncertainty(prob), 0.5)


class TestBatchProbOnly(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "Images"))
        for name in ("a.png", "b.png"):
            cv2.imwrite(os.path.join(self.tmp, "Images", name), np.full((H, W, 3), 190, np.uint8))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_maps_and_ranking(self):
        def factory(args):
            def predict(path):
                prob = np.zeros((H, W), np.float32)
                cv2.line(prob, (20, 100), (280, 100), 0.9 if "a." in path else 0.5, 3)
                return prob
            return predict
        rows = sam_batch.run(sam_batch.parse_args(["--data-dir", self.tmp, "--prob-only", "--checkpoint", "m.pt"]),
                             predictor_factory=factory)
        self.assertEqual([r["status"] for r in rows], ["map", "map"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "JSON files", "a.json")))
        self.assertGreater(rows[1]["uncertainty"], rows[0]["uncertainty"])   # b hesitates more
        maps = os.path.join(self.tmp, "SAM model", "probability maps", "m")
        self.assertEqual(sorted(os.listdir(maps)), ["a.png", "b.png"])


class TestGui(unittest.TestCase):

    def test_sidebar_and_menu(self):
        ai = dict(gm._RIGHT_SIDEBAR_SECTIONS)["AI"]
        self.assertIn(SAM_ACCEPT_CODE, [code for *_, code in ai])
        self.assertIn(SAM_ACCEPT_CODE, [code for _, code in gm._TOOL_COMMANDS])
        from config import Config
        cfg = Config()
        self.assertFalse(gm._SIDEBAR_TOGGLE_STATES[SAM_ACCEPT_CODE](cfg))
        cfg.sam_accept_state["active"] = True
        self.assertTrue(gm._SIDEBAR_TOGGLE_STATES[SAM_ACCEPT_CODE](cfg))


if __name__ == "__main__":
    unittest.main()
