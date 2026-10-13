"""test_v108_features.py

Headless tests for 1.0.8:
  - The sidebar title reads "More Features".
  - Eraser (sidebar, code 21): a window follows the mouse; dragging it over a crack erases that part
    (a crack crossed in the middle is split in two) and updates the JSON and the binary crack mask.
  - Remove uncertain (sidebar, code 22): deletes every crack on screen marked [?] (cracks hidden by
    Show cracks by number are kept) and updates the JSON and masks.
  - Training / Test / Validation set (sidebar, codes 23-25, Mode 2): the photo on screen moves to its set
    folder; the training set also takes its JSON and binary exports. The next photo is shown.
  (Validation available once every photo is segmented: see test_v106_features.TestValidationRun.)

Run with:  python3 -m unittest test_v108_features -v
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
import dataset_split as cc_split
from test_v105_link import _two_crack_app


def _mouse(app, event, x, y):
    wx, wy = app.transform_real_to_window_coords(x, y)
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.mouse_callback(event, wx, wy, 0, None)


def _drag(app, points):
    _mouse(app, cv2.EVENT_LBUTTONDOWN, *points[0])
    for p in points[1:]:
        _mouse(app, cv2.EVENT_MOUSEMOVE, *p)
    _mouse(app, cv2.EVENT_LBUTTONUP, *points[-1])


def _xs(app):
    return sorted(int(p[0]) for c in app.cfg.saved_cracks for p in c['path'])


class TestHelpers(unittest.TestCase):

    def test_kept_runs(self):
        self.assertEqual(cc.kept_runs([False, False, True, True, False]), [(0, 1), (4, 4)])
        self.assertEqual(cc.kept_runs([True, True]), [])
        self.assertEqual(cc.kept_runs([False, False]), [(0, 1)])

    def test_erased_runs(self):
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        self.assertEqual(cc.erased_runs(path, [True, False, True, True, False]), [[(0, 0)], [(2, 0), (3, 0)]])

    def test_sweep_centres_leave_no_gap(self):
        centres = cc.sweep_centres((0, 0), (100, 0), 10)
        self.assertEqual(centres[-1], (100, 0))
        steps = [centres[0][0]] + [b[0] - a[0] for a, b in zip(centres, centres[1:])]
        self.assertTrue(all(s <= 10 for s in steps))
        self.assertEqual(cc.sweep_centres(None, (5, 5), 10), [(5, 5)])


class TestEraser(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_eraser_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)

    def test_button_toggles_the_tool(self):
        self.assertTrue(self.app.cfg.eraser_state["active"])
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)
        self.assertFalse(self.app.cfg.eraser_state["active"])

    def test_drag_through_the_middle_splits_the_crack(self):
        _drag(self.app, [(60, 100), (65, 100), (70, 100), (80, 100)])
        xs = _xs(self.app)
        self.assertFalse(any(60 <= x <= 80 for x in xs), "the dragged stretch is gone")
        self.assertTrue(all(x in xs for x in list(range(20, 50)) + list(range(90, 121)) + list(range(160, 281))))
        self.assertEqual(len(self.app.cfg.saved_cracks), 3, "crack 1 split in two, crack 2 untouched")
        for c in self.app.cfg.saved_cracks:
            self.assertEqual((tuple(c['start']), tuple(c['end'])), (tuple(c['path'][0]), tuple(c['path'][-1])))

    def test_fast_drag_leaves_no_gap(self):
        # Only two mouse events far apart: everything between them is erased too.
        _drag(self.app, [(40, 100), (110, 100)])
        self.assertFalse(any(40 <= x <= 110 for x in _xs(self.app)))

    def test_release_updates_json_and_binary_mask(self):
        _drag(self.app, [(60, 100), (80, 100)])
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(len(shapes), 3)
        json_xs = {int(p[0]) for s in shapes for p in s["points"]}
        self.assertFalse(any(62 <= x <= 78 for x in json_xs))
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask[100, 70], 0, "erased stretch is background in the binary mask")
        self.assertEqual(mask[100, 30], 255)
        self.assertEqual(mask[100, 200], 255)

    def test_erasing_a_whole_crack_removes_it(self):
        _drag(self.app, [(155, 100), (200, 100), (240, 100), (285, 100)])
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertFalse(any(x >= 160 for x in _xs(self.app)))

    def test_undo_restores_the_crack(self):
        before = _xs(self.app)
        _drag(self.app, [(60, 100), (80, 100)])
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('u'))
        self.assertEqual(_xs(self.app), before)

    def test_click_away_from_cracks_changes_nothing(self):
        before = _xs(self.app)
        _drag(self.app, [(60, 30), (90, 30)])
        self.assertEqual(_xs(self.app), before)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH), "nothing to save")
        self.assertEqual(self.app.cfg.action_history, [])

    def test_hover_moves_the_window_without_erasing(self):
        before = _xs(self.app)
        _mouse(self.app, cv2.EVENT_MOUSEMOVE, 70, 100)
        self.assertEqual(_xs(self.app), before)
        self.assertIsNotNone(self.app.cfg.eraser_state["pos"])

    def test_brackets_resize_the_window(self):
        size = self.app.cfg.ERASER_SIZE_WIN_PX
        self.app.process_keypress(ord(']'))
        self.assertEqual(self.app.cfg.ERASER_SIZE_WIN_PX, size + self.app.cfg.ERASER_SIZE_STEP_WIN_PX)
        for _ in range(100):
            self.app.process_keypress(ord('['))
        self.assertEqual(self.app.cfg.ERASER_SIZE_WIN_PX, self.app.cfg.ERASER_MIN_SIZE_WIN_PX)

    def test_erases_while_zoom_window_is_on(self):
        self.app.cfg.zoom_window_state["active"] = True
        _drag(self.app, [(60, 100), (80, 100)])
        self.assertFalse(any(65 <= x <= 75 for x in _xs(self.app)))

    def test_other_tools_close_the_eraser(self):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('1'))
        self.assertFalse(self.app.cfg.eraser_state["active"])
        self.assertEqual(self.app.cfg.crack_cut_join_state["mode"], "cut")
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.ERASER_CODE)
        self.assertTrue(self.app.cfg.eraser_state["active"])
        self.assertIsNone(self.app.cfg.crack_cut_join_state["mode"], "the eraser closes CUT")
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('c'))
        self.assertFalse(self.app.cfg.eraser_state["active"])

    def test_window_is_drawn(self):
        _mouse(self.app, cv2.EVENT_MOUSEMOVE, 70, 100)
        win = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._draw_eraser(win)
        self.assertTrue(win.any())


class TestRemoveUncertain(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_unc_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _two_crack_app(self.tmpdir)
        self.uncertain_path = self.app.cfg.saved_cracks[1]['path']
        quality = lambda path: (30.0, 0.4, True) if path == self.uncertain_path else (90.0, 0.9, False)
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_removes_only_uncertain_cracks_and_updates_files(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=True) as confirm, \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(cc.REMOVE_UNCERTAIN_CODE)
        confirm.assert_called_once_with(1)
        self.assertEqual(len(self.app.cfg.saved_cracks), 1)
        self.assertFalse(any(x >= 160 for x in _xs(self.app)))
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(len(shapes), 1)
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask[100, 200], 0)
        self.assertEqual(mask[100, 50], 255)

    def test_cracks_hidden_by_number_filter_are_kept(self):
        # Only crack 1 shown: the uncertain crack 2 is hidden, so nothing is removed.
        self.app.cfg.crack_filter = {"numbers": [1],
                                     "hidden_keys": {self.app._crack_path_key(self.uncertain_path)}}
        with mock.patch.object(self.app, "_confirm_remove_uncertain") as confirm:
            self.assertFalse(self.app._remove_uncertain_cracks())
        confirm.assert_not_called()
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)

    def test_cancel_removes_nothing(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=False):
            self.assertFalse(self.app._remove_uncertain_cracks())
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH))

    def test_no_uncertain_crack_asks_nothing(self):
        self.app.cfg.saved_cracks.pop(1)
        with mock.patch.object(self.app, "_confirm_remove_uncertain") as confirm:
            self.assertFalse(self.app._remove_uncertain_cracks())
        confirm.assert_not_called()

    def test_undo_brings_them_back(self):
        with mock.patch.object(self.app, "_confirm_remove_uncertain", return_value=True), \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app._remove_uncertain_cracks()
            self.app.process_keypress(ord('u'))
        self.assertEqual(len(self.app.cfg.saved_cracks), 2)


class TestDatasetSplit(unittest.TestCase):

    def setUp(self):
        from test_gui_shell_parity import make_app
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_split_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        d = lambda *p: os.path.join(self.tmpdir, *p)
        for sub in ("Images", "JSON files", "Binary files"):
            os.makedirs(d(sub))
        for n in ("a", "b", "c"):
            open(d("Images", f"{n}.png"), "w").close()
            open(d("JSON files", f"{n}.json"), "w").close()
            for suffix in ("-crack_mask.png", "-seg.png", "-crack_uncertain_mask.png"):
                open(d("Binary files", f"{n}{suffix}"), "w").close()
        self.d = d
        app = make_app()
        app.cfg.SCRIPT_DIR = self.tmpdir
        app.cfg.modalita_scelta = "2"
        app.cfg.OUTPUT_FOLDER = d("JSON files")
        app.cfg.image_queue = [d("Images", f"{n}.png") for n in ("a", "b", "c")]
        app.cfg.CURRENT_IMAGE_PATH = d("Images", "b.png")
        app.cfg.JSON_OUTPUT_PATH = d("JSON files", "b.json")
        self.app = app

    def _ls(self, *p):
        path = self.d(*p)
        return sorted(os.listdir(path)) if os.path.isdir(path) else []

    def test_training_set_moves_photo_json_and_binaries(self):
        self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE), "the image loop moves on")
        self.assertEqual(self._ls("Training set", "Images"), ["b.png"])
        self.assertEqual(self._ls("Training set", "JSON files"), ["b.json"])
        self.assertEqual(self._ls("Training set", "Binary files"),
                         ["b-crack_mask.png", "b-crack_uncertain_mask.png", "b-seg.png"])
        self.assertEqual(self._ls("Images"), ["a.png", "c.png"])
        self.assertEqual(self._ls("JSON files"), ["a.json", "c.json"])
        self.assertFalse(any(n.startswith("b") for n in self._ls("Binary files")))
        self.assertEqual(len(self._ls("Binary files")), 6, "the other photos' files stay")

    def test_test_and_validation_sets_move_only_the_photo(self):
        for code, folder, photo in ((cc_split.TEST_SET_CODE, "Test set", "b"),
                                    (cc_split.VALIDATION_SET_CODE, "Validation set", "c")):
            self.app.cfg.CURRENT_IMAGE_PATH = self.d("Images", f"{photo}.png")
            self.app.cfg.JSON_OUTPUT_PATH = self.d("JSON files", f"{photo}.json")
            self.assertTrue(self.app.process_keypress(code))
            self.assertEqual(self._ls(folder), ["Images"])
            self.assertEqual(self._ls(folder, "Images"), [f"{photo}.png"])
            self.assertIn(f"{photo}.json", self._ls("JSON files"), "the JSON stays")
            self.assertIn(f"{photo}-crack_mask.png", self._ls("Binary files"), "the masks stay")
        self.assertEqual(self._ls("Images"), ["a.png"])

    def test_next_photo_takes_its_place_in_the_queue(self):
        self.app.process_keypress(cc_split.TRAINING_SET_CODE)
        self.assertEqual([os.path.basename(p) for p in self.app.cfg.image_queue], ["a.png", "c.png"])
        pos, _ = self.app._advance_or_switch_mode(1, 3)
        self.assertEqual(os.path.basename(self.app.cfg.image_queue[pos]), "c.png")

    def test_last_photo_ends_the_queue(self):
        self.app.cfg.CURRENT_IMAGE_PATH = self.d("Images", "c.png")
        self.app.cfg.JSON_OUTPUT_PATH = self.d("JSON files", "c.json")
        self.app.process_keypress(cc_split.TEST_SET_CODE)
        pos, _ = self.app._advance_or_switch_mode(2, 3)
        self.assertGreaterEqual(pos, len(self.app.cfg.image_queue))

    def test_mode1_moves_nothing(self):
        self.app.cfg.modalita_scelta = "1"
        for code in cc_split.SPLIT_BY_CODE:
            self.assertFalse(self.app.process_keypress(code))
        self.assertEqual(len(self._ls("Images")), 3)
        self.assertFalse(os.path.exists(self.d("Training set")))

    def test_training_set_needs_a_saved_json(self):
        os.remove(self.d("JSON files", "b.json"))
        self.assertFalse(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        self.assertIn("b.png", self._ls("Images"))

    def _group_photo(self, name):
        """Puts a photo named name in 'Images' with its JSON, and shows it."""
        open(self.d("Images", f"{name}.png"), "w").close()
        open(self.d("JSON files", f"{name}.json"), "w").close()
        self._show_photo(name)

    def _show_photo(self, name):
        self.app.cfg.CURRENT_IMAGE_PATH = self.d("Images", f"{name}.png")
        self.app.cfg.JSON_OUTPUT_PATH = self.d("JSON files", f"{name}.json")

    def test_first_photo_of_a_group_moves_without_asking(self):
        self._group_photo("x__BLDG007")
        with mock.patch.object(self.app, "_confirm_same_group_split") as confirm:
            self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        confirm.assert_not_called()

    def test_same_group_in_the_set_is_listed_and_waits_for_confirmation(self):
        self._group_photo("x__BLDG007")
        self.app.process_keypress(cc_split.TRAINING_SET_CODE)
        self._group_photo("y__BLDG007")
        with mock.patch.object(self.app, "_confirm_same_group_split", return_value=False) as confirm:
            self.assertFalse(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        message = confirm.call_args[0][0]
        self.assertIn("BLDG007", message)
        self.assertIn("x__BLDG007.png", message)
        self.assertIn("'Training set' already holds", message)
        self.assertIn("y__BLDG007.png", self._ls("Images"), "No = nothing moved")
        with mock.patch.object(self.app, "_confirm_same_group_split", return_value=True):
            self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        self.assertEqual(self._ls("Training set", "Images"), ["x__BLDG007.png", "y__BLDG007.png"])

    def test_same_group_in_another_set_is_reported_too(self):
        self._group_photo("x__BLDG007")
        self.app.process_keypress(cc_split.TRAINING_SET_CODE)
        self._group_photo("y__bldg007")
        with mock.patch.object(self.app, "_confirm_same_group_split", return_value=False) as confirm:
            self.assertFalse(self.app.process_keypress(cc_split.TEST_SET_CODE))
        self.assertIn("'Training set' holds (the same building in two sets)", confirm.call_args[0][0])
        self.assertFalse(os.path.exists(self.d("Test set")))

    def test_other_group_or_no_group_moves_without_asking(self):
        self._group_photo("x__BLDG007")
        self.app.process_keypress(cc_split.TRAINING_SET_CODE)
        self._group_photo("z__BLDG009")
        with mock.patch.object(self.app, "_confirm_same_group_split") as confirm:
            self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
            self._show_photo("a")   # no __BLDG tag at all
            self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        confirm.assert_not_called()

    def test_group_tag(self):
        self.assertEqual(cc_split.building_group_tag("DJI_20250307092515_0349_D__BLDG028.JPG"), "BLDG028")
        self.assertIsNone(cc_split.building_group_tag("IMG_0001.png"))

    def test_existing_files_are_never_overwritten(self):
        os.makedirs(self.d("Test set", "Images"))
        open(self.d("Test set", "Images", "b.png"), "w").close()
        self.app.process_keypress(cc_split.TEST_SET_CODE)
        self.assertEqual(self._ls("Test set", "Images"), ["b.png", "b_1.png"])

    def test_unsaved_changes_cancel_save_or_keep(self):
        self.app.cfg.action_history.append("crack_start")
        with mock.patch.object(self.app, "_ask_save_before_split", return_value=None), \
                mock.patch.object(self.app, "_save_annotations_now") as save:
            self.assertFalse(self.app.process_keypress(cc_split.TEST_SET_CODE))
        save.assert_not_called()
        self.assertIn("b.png", self._ls("Images"), "cancel moves nothing")
        with mock.patch.object(self.app, "_ask_save_before_split", return_value=True), \
                mock.patch.object(self.app, "_save_annotations_now") as save:
            self.assertTrue(self.app.process_keypress(cc_split.TRAINING_SET_CODE))
        save.assert_called_once()
        self.assertEqual(self._ls("Training set", "Images"), ["b.png"])

    def test_unsaved_changes_moved_as_last_saved_on_no(self):
        self.app.cfg.action_history.append("crack_start")
        with mock.patch.object(self.app, "_ask_save_before_split", return_value=False), \
                mock.patch.object(self.app, "_save_annotations_now") as save:
            self.assertTrue(self.app.process_keypress(cc_split.VALIDATION_SET_CODE))
        save.assert_not_called()
        self.assertEqual(self._ls("Validation set", "Images"), ["b.png"])


class TestSidebar(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_title_reads_more_features(self):
        title = self.app._sidebar.right.frame.children[0]
        self.assertEqual(title.kw.get("text"), "More Features")
        self.assertEqual(self.app._sidebar.left.frame.children[0].kw.get("text"), "View & Dataset")

    def test_new_buttons_send_their_codes(self):
        for label, code in (("Eraser", cc.ERASER_CODE), ("Remove uncertain", cc.REMOVE_UNCERTAIN_CODE)):
            btn = next(b for b in self.app._sidebar.buttons if label in b.text)
            self.app._pending_keys.clear()
            btn.command()
            self.assertEqual(self.app._pending_keys, [code])

    def test_eraser_button_looks_pressed_while_on(self):
        cfg = self.app.cfg
        btn = next(b for b in self.app._sidebar.buttons if "Eraser" in b.text)
        cfg.eraser_state["active"] = True
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "sunken")
        cfg.eraser_state["active"] = False
        self.app._sidebar.refresh_states(cfg, False)
        self.assertEqual(btn.kw.get("relief"), "raised")

    def test_new_commands_are_in_the_menu(self):
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        for code in (cc.ERASER_CODE, cc.REMOVE_UNCERTAIN_CODE) + tuple(cc_split.SPLIT_BY_CODE):
            self.assertIn(code, menu_codes)

    def test_split_buttons_send_their_codes_and_follow_the_mode(self):
        cfg = self.app.cfg
        buttons = {code: next(b for b in self.app._sidebar.buttons if label in b.text)
                   for code, label in ((23, "Training set"), (24, "Test set"), (25, "Validation set"))}
        for code, btn in buttons.items():
            self.app._pending_keys.clear()
            btn.command()
            self.assertEqual(self.app._pending_keys, [code])
        cfg.modalita_scelta, cfg.queue_idle = "1", False
        self.app._sidebar.refresh_states(cfg, False)
        self.assertTrue(all(b.kw.get("state") == "disabled" for b in buttons.values()), "Mode 1: greyed out")
        cfg.modalita_scelta = "2"
        self.app._sidebar.refresh_states(cfg, False)
        self.assertTrue(all(b.kw.get("state") == "normal" for b in buttons.values()))


if __name__ == "__main__":
    unittest.main()


class TestSplitCountsOnHover(unittest.TestCase):
    """Hovering Training / Test / Validation set shows how many photos each set holds."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_hover_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app.cfg.SCRIPT_DIR = self.tmpdir
        for folder, n in (("Training set", 3), ("Validation set", 1)):
            os.makedirs(os.path.join(self.tmpdir, folder, "Images"))
            for i in range(n):
                open(os.path.join(self.tmpdir, folder, "Images", f"p{i}.JPG"), "w").close()
        open(os.path.join(self.tmpdir, "Training set", "Images", "notes.txt"), "w").close()

    def _tip_text(self, tip):
        label = tip.tip.children[0]
        return label.kw["text"]

    def test_counts_per_set(self):
        self.assertEqual(self.app.dataset_split_counts(),
                         [("Training set", 3), ("Validation set", 1), ("Test set", 0)])

    def test_hover_shows_and_leave_hides(self):
        btn = self.app._sidebar.buttons_by_code[cc_split.TEST_SET_CODE]
        btn.fire("<Enter>")
        tip = self.app._sidebar.hover_tips[cc_split.TEST_SET_CODE]
        text = self._tip_text(tip)
        for line in ("Training set: 3", "Validation set: 1", "\u25B6 Test set: 0", "Total: 4"):
            self.assertIn(line, text)
        shown = tip.tip
        btn.fire("<Leave>")
        self.assertIsNone(tip.tip)
        self.assertTrue(shown.destroyed)

    def test_counts_are_read_at_each_hover(self):
        btn = self.app._sidebar.buttons_by_code[cc_split.TRAINING_SET_CODE]
        tip = self.app._sidebar.hover_tips[cc_split.TRAINING_SET_CODE]
        btn.fire("<Enter>")
        self.assertIn("\u25B6 Training set: 3", self._tip_text(tip))
        btn.fire("<ButtonPress>")
        self.assertIsNone(tip.tip, "a click hides it")
        open(os.path.join(self.tmpdir, "Training set", "Images", "p9.png"), "w").close()
        btn.fire("<Enter>")
        self.assertIn("Training set: 4", self._tip_text(tip))

    def test_every_set_button_has_its_tip(self):
        self.assertEqual(sorted(self.app._sidebar.hover_tips), [23, 24, 25])


class TestTwoSidebars(unittest.TestCase):
    """v1.0.8: a sidebar on each side of the canvas, each grouped into sections of similar commands."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)
        self.sb = self.app._sidebar

    def _texts(self, panel):
        return [b.text for b in panel.buttons]

    def test_one_panel_each_side_of_the_canvas(self):
        self.assertEqual(self.sb.right.side, "right")
        self.assertEqual(self.sb.left.side, "left")
        for panel in (self.sb.right, self.sb.left):
            self.assertIn(panel.frame, self.app._middle_row.children)

    def test_buttons_split_by_purpose(self):
        right, left = " ".join(self._texts(self.sb.right)), " ".join(self._texts(self.sb.left))
        for label in ("Crack tool", "Eraser", "Cut crack part", "Crack report"):
            self.assertIn(label, right)
            self.assertNotIn(label, left)
        for label in ("Training set", "CV folds", "Switch Mode", "Assign building", "Zoom window"):
            self.assertIn(label, left)
            self.assertNotIn(label, right)

    def test_every_command_once_and_both_panels_balanced(self):
        codes = [code for *_, code in self.gm._SIDEBAR_SHORTCUTS]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(sorted(self.sb.buttons_by_code), sorted(codes))
        # v1.0.9 added Reset pipeline training and Dataset browser to the Dataset section (left).
        self.assertLessEqual(abs(len(self.sb.right.buttons) - len(self.sb.left.buttons)), 3)

    def test_section_headings(self):
        heads = [h.kw["text"] for h in self.sb.right.headings + self.sb.left.headings]
        self.assertEqual(heads, ["TOOLS", "EDIT CRACKS", "INSPECT CRACKS", "VIEW", "BUILDING & SCALE", "DATASET"])

    def test_states_refreshed_on_both_panels(self):
        cfg = self.app.cfg
        cfg.eraser_state["active"] = True          # right panel
        cfg.zoom_window_state["active"] = True     # left panel
        cfg.modalita_scelta, cfg.queue_idle = "1", False
        self.sb.refresh_states(cfg, False)
        self.assertEqual(self.sb.buttons_by_code[21].kw.get("relief"), "sunken")
        self.assertEqual(self.sb.buttons_by_code[15].kw.get("relief"), "sunken")
        self.assertEqual(self.sb.buttons_by_code[23].kw.get("state"), "disabled")
        self.assertEqual(self.sb.buttons_by_code[17].kw.get("state"), "disabled")

    def test_set_counts_tip_opens_towards_the_canvas(self):
        self.assertEqual(self.sb.hover_tips[23].side, "left")
