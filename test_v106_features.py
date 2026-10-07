"""test_v106_features.py

Headless tests for the v1.0.6 additions:
  1. HWAV -- highlight windows already viewed while zoomed (code 14)
  2. Zoom window -- drag a rectangle to zoom onto a crack (code 15)
  3. Crack report -- written reliability report of one crack in the [J] glass (code 16)
  4. Mode 2 resumes from the last segmented photo
  5. Validation -- moves training-ready photos to 'Suitable for training' (code 17)

Run with:  python3 -m unittest test_v106_features -v
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from config import Config
from smart_segmentation import CrackSegmentation
import training_validation as tv
from test_v105_link import _two_crack_app, H, W


def _zoomed_app():
    app = _two_crack_app()
    app.cfg.zoom_factor = 1.0
    app.cfg.zoom_center = [W // 2, H // 2]
    app.refresh_zoom_viewport()
    return app


def _window_xy(app, x, y):
    return app.transform_real_to_window_coords(x, y)


def _render(app):
    app.cfg.img_background = np.full((H, W, 3), 120, dtype=np.uint8)
    app.cfg.CURRENT_IMAGE_PATH = "wall.png"
    with mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1200, 900)):
        win = app._build_window_frame()
        app._draw_hud_stack(win, 1, 1, False, 0.0)
    return win


# ---------------------------------------------------------------------------
# 1. HWAV
# ---------------------------------------------------------------------------

class TestHighlightViewedWindows(unittest.TestCase):

    def test_full_photo_view_is_not_recorded(self):
        app = _zoomed_app()
        app.cfg.zoom_factor = 1.0
        app.refresh_zoom_viewport()
        self.assertEqual(app.viewed_fraction(), 0.0)

    def test_panning_a_zoomed_view_marks_the_window_left_behind(self):
        app = _zoomed_app()
        app.cfg.zoom_factor = 2.0
        app.cfg.zoom_center = [75, 50]
        app.refresh_zoom_viewport()
        first_box = list(app.cfg.zoom_box)
        self.assertEqual(app.viewed_fraction(), 0.0, "the window being looked at is not 'already viewed' yet")
        app.pan_view_by_fraction(1.0, 0.0)
        self.assertNotEqual(app.cfg.zoom_box, first_box)
        mask = app.cfg.viewed_mask
        x0, y0, x1, y1 = first_box
        self.assertTrue(mask[y0 + 1:y1 - 1, x0 + 1:x1 - 1].all(), "the first window is recorded as seen")
        self.assertAlmostEqual(app.viewed_fraction(), 0.25, delta=0.02)

    def test_highlight_paints_green_glass_only_on_viewed_area(self):
        app = _zoomed_app()
        app.cfg.zoom_factor = 2.0
        app.cfg.zoom_center = [75, 50]
        app.refresh_zoom_viewport()
        app.cfg.zoom_factor = 1.0
        app.refresh_zoom_viewport()       # back to the whole photo: the top-left quarter was seen
        app.process_keypress(14)
        self.assertTrue(app.cfg.show_viewed_windows)
        win = _render(app)
        seen_px = win[300, 300].astype(int)    # top-left quarter of the 1200x900 window
        unseen_px = win[700, 900].astype(int)
        self.assertGreater(seen_px[1] - seen_px[0], 20, "viewed area is tinted green")
        self.assertLess(abs(int(unseen_px[1]) - int(unseen_px[0])), 5, "unviewed area keeps its colors")

    def test_second_press_removes_the_colors_and_clears_the_record(self):
        app = _zoomed_app()
        app._record_viewed_window([0, 0, 50, 50])
        app.process_keypress(14)
        app.process_keypress(14)
        self.assertFalse(app.cfg.show_viewed_windows)
        self.assertIsNone(app.cfg.viewed_mask)
        self.assertEqual(app.viewed_fraction(), 0.0)

    def test_a_new_photo_starts_with_nothing_viewed(self):
        app = _zoomed_app()
        app._record_viewed_window([0, 0, 50, 50])
        app._reset_inspection_tools_for_new_image()
        self.assertEqual(app.viewed_fraction(), 0.0)

    def test_large_photos_use_a_bounded_low_resolution_map(self):
        app = _zoomed_app()
        app.cfg.H_img, app.cfg.W_img = 4000, 6000
        app._record_viewed_window([0, 0, 3000, 2000])
        self.assertLessEqual(max(app.cfg.viewed_mask.shape), app.cfg.VIEWED_MASK_MAX_DIM)
        self.assertAlmostEqual(app.viewed_fraction(), 0.25, delta=0.01)


# ---------------------------------------------------------------------------
# 2. Zoom window
# ---------------------------------------------------------------------------

class TestZoomWindow(unittest.TestCase):

    def _drag(self, app, p1, p2):
        a, b = _window_xy(app, *p1), _window_xy(app, *p2)
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, a[0], a[1], 0, None)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, (a[0] + b[0]) // 2, (a[1] + b[1]) // 2, 0, None)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, b[0], b[1], 0, None)
        app.mouse_callback(cv2.EVENT_LBUTTONUP, b[0], b[1], 0, None)

    def test_dragging_a_rectangle_zooms_onto_it(self):
        app = _zoomed_app()
        app.process_keypress(15)
        self.assertTrue(app.cfg.zoom_window_state["active"])
        self._drag(app, (100, 80), (160, 120))
        self.assertAlmostEqual(app.cfg.zoom_factor, W / 60.0, places=3)
        x0, y0, x1, y1 = app.cfg.zoom_box
        self.assertLessEqual(x0, 101)
        self.assertGreaterEqual(x1, 159)
        self.assertLessEqual(y0, 81)
        self.assertGreaterEqual(y1, 119)

    def test_narrow_rectangle_is_widened_to_the_photo_aspect(self):
        app = _zoomed_app()
        app.process_keypress(15)
        self._drag(app, (140, 40), (160, 160))   # tall and thin
        x0, y0, x1, y1 = app.cfg.zoom_box
        self.assertAlmostEqual((x1 - x0) / float(y1 - y0), W / float(H), delta=0.05)
        self.assertLessEqual(y0, 41)
        self.assertGreaterEqual(y1, 159)

    def test_dragging_never_traces_a_crack(self):
        app = _zoomed_app()
        n = len(app.cfg.saved_cracks)
        app.process_keypress(15)
        self._drag(app, (30, 30), (90, 90))
        self.assertEqual(len(app.cfg.saved_cracks), n)
        self.assertIsNone(app.cfg.temp_start)

    def test_a_click_without_dragging_leaves_the_zoom_alone(self):
        app = _zoomed_app()
        app.process_keypress(15)
        self._drag(app, (100, 100), (101, 101))
        self.assertEqual(app.cfg.zoom_factor, 1.0)

    def test_second_press_closes_the_tool_and_resets_the_zoom(self):
        app = _zoomed_app()
        app.process_keypress(15)
        self._drag(app, (100, 80), (160, 120))
        app.process_keypress(15)
        self.assertFalse(app.cfg.zoom_window_state["active"])
        self.assertEqual(app.cfg.zoom_factor, 1.0)
        self.assertEqual(app.cfg.zoom_box, [0, 0, W, H])

    def test_live_rectangle_is_drawn_while_dragging(self):
        app = _zoomed_app()
        app.process_keypress(15)
        a, b = _window_xy(app, 50, 50), _window_xy(app, 200, 150)
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, a[0], a[1], 0, None)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, b[0], b[1], 0, None)
        win = _render(app)
        self.assertTrue((win[a[1], a[0] + 3] == (255, 255, 0)).all(), "dashed cyan edge at the first corner")


# ---------------------------------------------------------------------------
# 3. Crack report
# ---------------------------------------------------------------------------

class TestCrackReport(unittest.TestCase):

    def _click(self, app, x, y):
        wx, wy = _window_xy(app, x, y)
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)

    def _app(self):
        app = _zoomed_app()
        app.cfg.img_original = np.full((H, W, 3), 200, dtype=np.uint8)
        cv2.line(app.cfg.img_original, (20, 100), (280, 100), (30, 30, 30), 2)
        return app

    def test_click_on_a_crack_opens_its_report_in_the_glass(self):
        app = self._app()
        app.process_keypress(16)
        self.assertFalse(app.cfg.show_info_overlay, "the glass opens only once a crack is picked")
        self._click(app, 200, 100)
        self.assertEqual(app.cfg.crack_report_state["idx"], 1)
        self.assertTrue(app.cfg.show_info_overlay, "the [J] glass is activated automatically")
        title, lines = app.build_crack_report()
        self.assertEqual(title, "CRACK 2 REPORT")
        text = "\n".join(lines)
        for word in ("Reliability", "Confidence", "Multi-view", "Uncertainty", "Overall reliability"):
            self.assertIn(word, text)

    def test_report_replaces_the_usual_panel_content(self):
        app = self._app()
        app.process_keypress(16)
        self._click(app, 200, 100)
        with mock.patch.object(app, "_build_crack_reliability_lines") as usual, \
                mock.patch.object(app, "_draw_hud_text_with_shadow") as building:
            _render(app)
        usual.assert_not_called()
        building.assert_not_called()

    def test_click_away_from_cracks_selects_nothing(self):
        app = self._app()
        app.process_keypress(16)
        self._click(app, 150, 30)
        self.assertIsNone(app.cfg.crack_report_state["idx"])
        self.assertEqual(app.build_crack_report(), (None, []))

    def test_second_press_hides_the_glass(self):
        app = self._app()
        app.process_keypress(16)
        self._click(app, 50, 100)
        app.process_keypress(16)
        self.assertFalse(app.cfg.crack_report_state["active"])
        self.assertFalse(app.cfg.show_info_overlay)

    def test_report_of_a_deleted_crack_says_so(self):
        app = self._app()
        app.process_keypress(16)
        self._click(app, 50, 100)
        app.cfg.saved_cracks[0]['active'] = False
        title, lines = app.build_crack_report()
        self.assertEqual(title, "CRACK 1 REPORT")
        self.assertIn("no longer exists", lines[0])

    def test_verdict_levels(self):
        app = self._app()
        self.assertEqual(app._crack_report_verdict(90.0, 0.9, False)[0], "HIGH")
        self.assertEqual(app._crack_report_verdict(55.0, 0.9, False)[0], "MEDIUM")
        self.assertEqual(app._crack_report_verdict(20.0, 0.3, True)[0], "LOW")

    def test_report_panel_renders(self):
        app = self._app()
        app.process_keypress(16)
        self._click(app, 200, 100)
        _render(app)   # must not raise

    def test_zoom_window_and_crack_report_exclude_each_other(self):
        app = self._app()
        app.process_keypress(16)
        app.process_keypress(15)
        self.assertFalse(app.cfg.crack_report_state["active"])
        app.process_keypress(16)
        self.assertFalse(app.cfg.zoom_window_state["active"])


# ---------------------------------------------------------------------------
# 4. Mode 2 resumes from the last segmented photo
# ---------------------------------------------------------------------------

def _make_segmented(tmpdir, names, shapes_by_name=None, mtimes=None, image_maker=None):
    images = os.path.join(tmpdir, "Images")
    jsons = os.path.join(tmpdir, "JSON files")
    binary = os.path.join(tmpdir, "Binary files")
    for d in (images, jsons, binary):
        os.makedirs(d, exist_ok=True)
    for i, name in enumerate(names):
        img = image_maker(i) if image_maker else np.full((120, 160, 3), 40 + 20 * i, dtype=np.uint8)
        cv2.imwrite(os.path.join(images, f"{name}.png"), img)
        shapes = (shapes_by_name or {}).get(name, [])
        json_path = os.path.join(jsons, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"shapes": shapes, "imagePath": f"{name}.png"}, f)
        if mtimes:
            os.utime(json_path, (mtimes[i], mtimes[i]))
        for suffix in ("-seg.png", "-crack_mask.png"):
            cv2.imwrite(os.path.join(binary, f"{name}{suffix}"), np.zeros((4, 4), np.uint8))
    return images, jsons, binary


class TestMode2ResumesFromLastSegmented(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v106_mode2_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        now = time.time()
        _make_segmented(self.tmpdir, ["a", "b", "c", "d"], mtimes=[now - 400, now - 300, now - 10, now - 200])
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        self.app = CrackSegmentation(cfg)

    def test_switch_mode_lands_on_the_most_recently_saved_photo(self):
        self.app.cfg.modalita_scelta = "1"
        self.app.cfg.switch_mode_requested = True
        pos, total = self.app._advance_or_switch_mode(0, 0)
        self.assertEqual(self.app.cfg.modalita_scelta, "2")
        self.assertEqual(total, 4)
        self.assertEqual(os.path.basename(self.app.cfg.image_queue[pos]), "c.png")

    def test_switch_back_to_mode_1_still_starts_from_the_first(self):
        os.remove(os.path.join(self.tmpdir, "JSON files", "a.json"))
        self.app.cfg.modalita_scelta = "2"
        self.app.cfg.switch_mode_requested = True
        pos, _ = self.app._advance_or_switch_mode(2, 4)
        self.assertEqual(self.app.cfg.modalita_scelta, "1")
        self.assertEqual(pos, 0)


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

def _crack(rel, conf, uncertain=False):
    return {"label": "crack_incerta_x.png" if uncertain else "crack_x.png", "points": [[0, 0], [10, 10]],
            "shape_type": "linestrip", "flags": {"uncertain": uncertain},
            "reliability_pct": rel, "confidence": conf}


def _texture(seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(small, (160, 120), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


class TestValidationRules(unittest.TestCase):

    def test_quality_summary_and_thresholds(self):
        cfg = Config()
        tmp = tempfile.mkdtemp(prefix="crackseg_v106_q_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "p.json")
        with open(path, "w") as f:
            json.dump({"shapes": [_crack(80, 0.9), _crack(70, 0.7, True),
                                  {"label": "detachment_x", "shape_type": "polygon", "points": []}]}, f)
        q = tv.photo_quality_from_json(path)
        self.assertEqual((q["cracks"], q["detachments"], q["uncertain_cracks"]), (2, 1, 1))
        self.assertAlmostEqual(q["mean_reliability_pct"], 75.0)
        ok, reason = tv.evaluate_photo_quality(q, cfg)
        self.assertFalse(ok, "half the cracks are uncertain")
        self.assertIn("uncertain", reason)
        cfg.VALIDATION_MAX_UNCERTAIN_RATIO = None
        self.assertTrue(tv.evaluate_photo_quality(q, cfg)[0])

    def test_rejections(self):
        cfg = Config()
        base = {"cracks": 1, "detachments": 0, "mean_reliability_pct": 80.0, "mean_confidence": 0.9,
                "uncertain_cracks": 0, "uncertain_ratio": 0.0}
        self.assertTrue(tv.evaluate_photo_quality(base, cfg)[0])
        self.assertFalse(tv.evaluate_photo_quality(dict(base, cracks=0), cfg)[0])
        self.assertFalse(tv.evaluate_photo_quality(dict(base, mean_reliability_pct=40.0), cfg)[0])
        self.assertFalse(tv.evaluate_photo_quality(dict(base, mean_confidence=0.5), cfg)[0])
        self.assertIn("Mode 2", tv.evaluate_photo_quality(dict(base, mean_confidence=None), cfg)[1])

    def test_difference_hash_spots_near_duplicates(self):
        tmp = tempfile.mkdtemp(prefix="crackseg_v106_h_")
        self.addCleanup(shutil.rmtree, tmp, True)
        a, b, c = (os.path.join(tmp, n) for n in ("a.png", "b.png", "c.png"))
        cv2.imwrite(a, _texture(1))
        cv2.imwrite(b, cv2.convertScaleAbs(_texture(1), alpha=1.1, beta=8))   # same scene, brighter
        cv2.imwrite(c, _texture(2))
        ha, hb, hc = (tv.difference_hash(p) for p in (a, b, c))
        self.assertLessEqual(tv.hash_distance(ha, hb), 20)
        self.assertGreater(tv.hash_distance(ha, hc), 60)

    def test_binary_exports_match_only_their_own_photo(self):
        names = ["IMG_1-seg.png", "IMG_1-crack_mask.png", "IMG_10-crack_mask.png", "IMG_1-other.png"]
        self.assertEqual(tv.binary_exports_of("IMG_1", names), ["IMG_1-crack_mask.png", "IMG_1-seg.png"])


class TestValidationRun(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v106_val_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        textures = {0: _texture(1), 1: cv2.convertScaleAbs(_texture(1), alpha=1.1, beta=8),
                    2: _texture(2), 3: _texture(3)}
        shapes = {
            "good": [_crack(85, 0.9)],
            "good_twin": [_crack(75, 0.8)],          # same scene as "good", lower scores
            "weak": [_crack(30, 0.55, True)],
            "other": [_crack(90, 0.92)],
        }
        self.images, self.jsons, self.binary = _make_segmented(
            self.tmpdir, ["good", "good_twin", "weak", "other"], shapes, image_maker=lambda i: textures[i])
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        self.app = CrackSegmentation(cfg)
        self.app.cfg.modalita_scelta = "2"
        self.app.cfg.OUTPUT_FOLDER = self.jsons
        self.app.cfg.image_queue = [os.path.join(self.images, f"{n}.png") for n in ("good", "good_twin", "other", "weak")]
        self.app.cfg.CURRENT_IMAGE_PATH = self.app.cfg.image_queue[-1]

    def test_plan_keeps_good_photos_and_drops_weak_and_duplicates(self):
        plan = self.app.plan_training_validation()
        self.assertEqual([c["photo"] for c in plan["selected"]], ["good.png", "other.png"])
        self.assertEqual([c["photo"] for c, _ in plan["rejected"]], ["weak.png"])
        self.assertEqual([(c["photo"], twin) for c, twin in plan["duplicates"]], [("good_twin.png", "good.png")])

    def test_validation_moves_photo_json_and_binary_files(self):
        with mock.patch.object(self.app, "_confirm_training_validation", return_value=True), \
                mock.patch.object(self.app, "_notify_training_validation_result"):
            advanced = self.app.process_keypress(17)
        self.assertTrue(advanced, "the image loop exits so the queue is reloaded")
        self.assertTrue(self.app.cfg.reload_queue_requested)
        root = os.path.join(self.tmpdir, "Suitable for training")
        self.assertEqual(sorted(os.listdir(os.path.join(root, "Images"))), ["good.png", "other.png"])
        self.assertEqual(sorted(os.listdir(os.path.join(root, "JSON files"))), ["good.json", "other.json"])
        self.assertEqual(sorted(os.listdir(os.path.join(root, "Binary files"))),
                         ["good-crack_mask.png", "good-seg.png", "other-crack_mask.png", "other-seg.png"])
        self.assertEqual(sorted(os.listdir(self.images)), ["good_twin.png", "weak.png"], "the others stay")
        reports = [f for f in os.listdir(root) if f.startswith("validation_report_")]
        self.assertEqual(len(reports), 1)
        with open(os.path.join(root, reports[0]), newline="", encoding="utf-8") as f:
            rows = {r["photo"]: r["decision"] for r in csv.DictReader(f)}
        self.assertEqual(rows, {"good.png": "suitable", "other.png": "suitable",
                                "weak.png": "below thresholds", "good_twin.png": "too similar"})

    def test_queue_reload_after_validation(self):
        with mock.patch.object(self.app, "_confirm_training_validation", return_value=True), \
                mock.patch.object(self.app, "_notify_training_validation_result"):
            self.app.process_keypress(17)
        pos, total = self.app._advance_or_switch_mode(3, 4)
        self.assertFalse(self.app.cfg.reload_queue_requested)
        self.assertEqual(sorted(os.path.basename(p) for p in self.app.cfg.image_queue), ["good_twin.png", "weak.png"])
        self.assertEqual(total, 2)
        self.assertTrue(0 <= pos < 2)

    def test_photos_already_moved_win_against_new_twins(self):
        with mock.patch.object(self.app, "_confirm_training_validation", return_value=True), \
                mock.patch.object(self.app, "_notify_training_validation_result"):
            self.app.process_keypress(17)
        with open(os.path.join(self.jsons, "good_twin.json"), "w") as f:
            json.dump({"shapes": [_crack(99, 0.99)]}, f)   # now better than the moved "good"
        plan = self.app.plan_training_validation()
        self.assertEqual([(c["photo"], t) for c, t in plan["duplicates"]], [("good_twin.png", "good.png")])
        self.assertEqual(plan["selected"], [])

    def test_cancel_moves_nothing(self):
        with mock.patch.object(self.app, "_confirm_training_validation", return_value=False):
            self.assertFalse(self.app.process_keypress(17))
        self.assertEqual(len(os.listdir(self.images)), 4)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "Suitable for training")))

    def test_not_available_before_the_last_image(self):
        self.app.cfg.CURRENT_IMAGE_PATH = self.app.cfg.image_queue[0]
        self.assertFalse(self.app._validation_available())
        with mock.patch.object(self.app, "_confirm_training_validation") as confirm:
            self.assertFalse(self.app.process_keypress(17))
        confirm.assert_not_called()
        self.assertEqual(len(os.listdir(self.images)), 4)

    def test_available_after_saving_the_last_mode1_photo_with_s(self):
        # [S] in Mode 1 removes the photo from the queue but stays on it.
        self.app.cfg.image_queue = self.app.cfg.image_queue[:3]
        self.assertTrue(self.app._validation_available())
        self.app.cfg.CURRENT_IMAGE_PATH = os.path.join(self.images, "a_first.png")
        self.assertFalse(self.app._validation_available(), "photos still follow it in the queue")

    def test_existing_destination_files_are_never_overwritten(self):
        dest = os.path.join(self.tmpdir, "Suitable for training", "JSON files")
        os.makedirs(dest)
        with open(os.path.join(dest, "other.json"), "w") as f:
            f.write("{}")
        self.assertEqual(os.path.basename(tv.unique_destination(dest, "other.json")), "other_1.json")


# ---------------------------------------------------------------------------
# GUI sidebar
# ---------------------------------------------------------------------------

class TestSidebarButtons(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def _button(self, label):
        return next(b for b in self.app._sidebar.buttons if label in b.text)

    def test_new_buttons_send_their_codes(self):
        for label, code in (("HWAV", 14), ("Zoom window", 15), ("Crack report", 16), ("Validation", 17)):
            btn = self._button(label)
            self.assertNotIn("[]", btn.text)
            self.app._pending_keys.clear()
            btn.command()
            self.assertEqual(self.app._pending_keys, [code])

    def test_toggles_look_pressed_and_validation_follows_the_queue_end(self):
        cfg = self.app.cfg
        cfg.image_queue = ["a.png", "b.png"]
        cfg.CURRENT_IMAGE_PATH = "a.png"
        cfg.zoom_window_state["active"] = True
        self.app._sidebar.refresh_states(cfg, self.app._validation_available())
        self.assertEqual(self._button("Zoom window").kw.get("relief"), "sunken")
        self.assertEqual(self._button("HWAV").kw.get("relief"), "raised")
        self.assertEqual(self._button("Validation").kw.get("state"), "disabled")
        cfg.CURRENT_IMAGE_PATH = "b.png"
        self.app._sidebar.refresh_states(cfg, self.app._validation_available())
        self.assertEqual(self._button("Validation").kw.get("state"), "normal")


if __name__ == "__main__":
    unittest.main()
