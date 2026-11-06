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
        tr.assert_called_once_with(5, 6)
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


class TestHoverAndStayOn(_Base):

    def test_hover_picks_the_crack_under_the_mouse(self):
        self._on()
        self.assertTrue(self.app._update_sam_hover(60, 103))
        hover = self.app.cfg.sam_accept_state["hover"]
        self.assertGreater(hover, 0)
        self.assertFalse(self.app._update_sam_hover(70, 102))   # same crack: no redraw
        self.assertTrue(self.app._update_sam_hover(80, 160))    # wall: nothing under the mouse
        self.assertEqual(self.app.cfg.sam_accept_state["hover"], 0)

    def test_hover_painted_yellow_even_with_tint_hidden(self):
        self._on()
        self.app.cfg.SAM_ACCEPT_SHOW_CANDIDATES = False
        self.app._update_sam_hover(60, 103)
        scene = np.full((H, W, 3), 190, np.uint8)
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        self.assertGreater(int(scene[100, 60, 2]), 200)              # yellow (BGR 0,255,255) on the crack
        np.testing.assert_array_equal(scene[40, 100], [190, 190, 190])   # the other crack: not hovered, tint off

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


class TestAutoAccept(_Base):

    def setUp(self):
        super().setUp()
        self.app.cfg.SAM_ACCEPT_AUTO = True
        self.app.cfg.SAM_ACCEPT_SHOW_CANDIDATES = False

    def test_sure_cracks_added_at_once_the_others_left(self):
        self._on()
        cracks = self.app.cfg.saved_cracks
        self.assertEqual(len(cracks), 2)   # the strong crack (0.9) and its branch; the weak one (0.55) is left
        self.assertTrue(all(c["source"] == "sam" and not c.get("uncertain_manual") for c in cracks))
        self.assertTrue(all(abs(p[1] - 40) > 3 for c in cracks for p in c["path"]))
        self.assertEqual(len(_json_cracks(self.app)), 2)

    def test_left_crack_hovered_and_clicked(self):
        self._on()
        self.assertFalse(self.app._update_sam_hover(60, 101) and self.app.cfg.sam_accept_state["hover"])
        self.assertTrue(self.app._update_sam_hover(100, 41))   # the weak crack, left to the operator
        with mock.patch.object(self.app, "transform_window_to_real_coords", return_value=(100, 41)), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 1, 1)
        self.assertEqual(sorted(c["source"] for c in self.app.cfg.saved_cracks), ["sam", "sam", "sam_click"])
        self.assertFalse(self.app._update_sam_hover(100, 41) and self.app.cfg.sam_accept_state["hover"])

    def test_hover_shades_by_probability(self):
        self._on()
        self.app.cfg.sam_accept_state["done_labels"].clear()
        scene = np.full((H, W, 3), 190, np.uint8)
        self.app._update_sam_hover(60, 101)
        self.app._paint_sam_candidates(scene, (0, H, 0, W))
        strong = scene[100, 60].astype(int)
        scene2 = np.full((H, W, 3), 190, np.uint8)
        self.app._update_sam_hover(100, 41)
        self.app._paint_sam_candidates(scene2, (0, H, 0, W))
        weak = scene2[40, 100].astype(int)
        self.assertLess(strong[0], weak[0])   # darker yellow (less blue) where the model is surer

    def test_one_undo_removes_the_automatic_part(self):
        self._on()
        _press(self.app, ord('u'))
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_not_repeated_on_a_photo_already_done(self):
        self._on()
        _press(self.app, SAM_ACCEPT_CODE)
        self._on()
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_threshold(self):
        self.app.cfg.SAM_ACCEPT_AUTO_MIN_MEAN = 0.95
        self._on()
        self.assertEqual(self.app.cfg.saved_cracks, [])


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
