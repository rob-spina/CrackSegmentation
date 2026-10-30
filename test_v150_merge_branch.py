"""test_v150_merge_branch.py

Headless tests for the 1.5.0 [Merge crack branches] command (crack_branch_merge.py):
  - the sidebar button / Tools menu (code 134) turns the mode on; the first crack clicked is the main crack,
    the next ones become its branches (one LabelMe group_id, flags branch_main / branch);
  - a branch that stops short of the group is bridged to it (recorded in 'links'), a far one is refused;
  - a main crack with branches brings them along; markers read N and N.k;
  - Shift+C, code 135 and Shift+click on the button exit; another command exits; Undo keeps the mode;
  - JSON round trip and binary masks.

Run with:  python3 -m unittest test_v150_merge_branch -v
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

import crack_branch_merge as cbm
from test_gui_shell_parity import make_app

H, W = 300, 300


def _good_quality(path):
    return 90.0, 0.9, False


def _crack(points):
    return {'start': points[0], 'end': points[-1], 'path': list(points), 'active': True, 'session_id': 'current'}


def _branch_app(tmpdir):
    """Main crack on y=100 (x 20..280); crack 2 touches it at x=100 (y 101..180); crack 3 stops 20 px short of
    it at x=200 (y 120..180); crack 4 is far away (y 270, x 20..80)."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    img = np.full((H, W, 3), 200, dtype=np.uint8)
    skel = np.zeros((H, W), dtype=np.uint8)
    for a, b in (((20, 100), (280, 100)), ((100, 101), (100, 180)), ((200, 100), (200, 180)), ((20, 270), (80, 270))):
        cv2.line(skel, a, b, 255, 1)
        cv2.line(img, a, b, (30, 30, 30), 2)
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    app.cfg.saved_cracks = [
        _crack([(x, 100) for x in range(20, 281)]),
        _crack([(100, y) for y in range(101, 181)]),
        dict(_crack([(200, y) for y in range(120, 181)]), width_segments=[{'i0': 0, 'i1': 10, 'width_px': 3}]),
        _crack([(x, 270) for x in range(20, 81)]),
    ]
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    images = os.path.join(tmpdir, "Images")
    os.makedirs(images)
    img_path = os.path.join(images, "wall.png")
    cv2.imwrite(img_path, img)
    app.cfg.SCRIPT_DIR = tmpdir
    app.cfg.OUTPUT_FOLDER = os.path.join(tmpdir, "JSON files")
    app.cfg.CURRENT_IMAGE_PATH = img_path
    app.cfg.JSON_OUTPUT_PATH = os.path.join(app.cfg.OUTPUT_FOLDER, "wall.json")
    app.cfg.img_original = img
    app.cfg.current_tool = 'crack'
    return app


class _Case(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v150_merge_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.app = _branch_app(self.tmpdir)
        patcher = mock.patch.object(self.app, "_crack_quality", side_effect=_good_quality)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _press(self, code):
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            return self.app.process_keypress(code)

    def _click(self, x, y):
        wx, wy = self.app.transform_real_to_window_coords(x, y)
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)

    def _state(self):
        return self.app.cfg.crack_branch_merge_state

    def _cracks(self):
        return self.app.cfg.saved_cracks

    def _json_lines(self):
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            return [s for s in json.load(f)["shapes"] if s["shape_type"] == "linestrip"]


class TestMode(_Case):

    def test_codes_are_never_typed_characters(self):
        for code in (cbm.MERGE_BRANCH_CODE, cbm.MERGE_BRANCH_EXIT_CODE):
            self.assertTrue(128 <= code < 160)

    def test_on_and_plain_button_again_keeps_it_on(self):
        self.assertFalse(self._state()["active"])
        self._press(cbm.MERGE_BRANCH_CODE)
        self.assertTrue(self._state()["active"])
        self._press(cbm.MERGE_BRANCH_CODE)
        self.assertTrue(self._state()["active"], "only Shift+click / Shift+C exit")

    def test_shift_c_exits_and_does_not_select_a_tool(self):
        self.app.cfg.current_tool = 'detachment'
        self._press(cbm.MERGE_BRANCH_CODE)
        self._press(ord('C'))
        self.assertFalse(self._state()["active"])
        self.assertEqual(self.app.cfg.current_tool, 'detachment', "Shift+C only exits the mode")

    def test_shift_c_without_the_mode_still_selects_the_crack_tool(self):
        self.app.cfg.current_tool = 'detachment'
        self._press(ord('C'))
        self.assertEqual(self.app.cfg.current_tool, 'crack')

    def test_exit_code_exits(self):
        self._press(cbm.MERGE_BRANCH_CODE)
        self._press(cbm.MERGE_BRANCH_EXIT_CODE)
        self.assertFalse(self._state()["active"])

    def test_another_command_exits_view_keys_do_not(self):
        self._press(cbm.MERGE_BRANCH_CODE)
        self._press(ord(' '))
        self.assertTrue(self._state()["active"])
        self._press(ord('1'))
        self.assertFalse(self._state()["active"])

    def test_mode_line(self):
        self._press(cbm.MERGE_BRANCH_CODE)
        texts = []
        with mock.patch.object(self.app, "_put_hud_text", side_effect=lambda w, t, *a, **k: texts.append(t)):
            self.app._draw_crack_branch_merge_label(None)
        self.assertTrue(texts and texts[0].startswith("MERGE CRACK BRANCHES"))

    def test_main_alone_is_plain_again_after_exit(self):
        self._press(cbm.MERGE_BRANCH_CODE)
        self._click(150, 100)
        self.assertTrue(self._cracks()[0].get('branch_main'))
        self._press(ord('C'))
        self.assertNotIn('crack_group', self._cracks()[0])
        self.assertNotIn('branch_main', self._cracks()[0])


class TestMerge(_Case):

    def _merge(self, *points):
        self._press(cbm.MERGE_BRANCH_CODE)
        for p in points:
            self._click(*p)

    def test_touching_branch_joins_the_group(self):
        self._merge((150, 100), (100, 150))
        main, br = self._cracks()[0], self._cracks()[1]
        self.assertTrue(main['branch_main'])
        self.assertTrue(br['branch'])
        self.assertEqual(main['crack_group'], br['crack_group'])
        self.assertEqual(br['path'][0], (100, 101), "touching branch: path unchanged")
        self.assertNotIn('links', br)

    def test_short_branch_is_bridged_to_the_main_crack(self):
        self._merge((150, 100), (200, 150))
        br = self._cracks()[2]
        self.assertTrue(br['branch'])
        self.assertEqual(br['path'][0], (200, 100), "bridged up to the main crack")
        self.assertEqual(br['start'], (200, 100))
        self.assertEqual(br['links'][0]['from'], [200, 120])
        self.assertEqual(br['links'][0]['to'], [200, 100])
        steps = np.abs(np.diff(np.asarray(br['path']), axis=0)).max()
        self.assertLessEqual(steps, 1, "continuous path")
        seg = br['width_segments'][0]
        self.assertEqual((seg['i0'], seg['i1']), (20, 30), "width tract re-indexed after the bridge")

    def test_far_crack_is_refused(self):
        self._merge((150, 100), (50, 270))
        self.assertNotIn('crack_group', self._cracks()[3])
        self.assertFalse(os.path.exists(self.app.cfg.JSON_OUTPUT_PATH))

    def test_several_branches_and_marker_labels(self):
        self._merge((150, 100), (100, 150), (200, 150))
        labels = [self.app._crack_marker_label(i) for i in range(4)]
        self.assertEqual(labels, [1, "1.1", "1.2", 4])

    def test_clicking_a_branch_first_selects_its_main(self):
        self._merge((150, 100), (100, 150))
        self._press(ord('C'))
        self._merge((100, 150), (200, 150))
        gid = self._cracks()[0]['crack_group']
        self.assertEqual(self._cracks()[2]['crack_group'], gid)
        self.assertTrue(self._cracks()[0]['branch_main'])

    def test_a_main_crack_brings_its_branches(self):
        self.app.cfg.BRANCH_MERGE_MAX_GAP_PX = 200
        self._merge((200, 150), (150, 100))    # crack 3 main, crack 1 its branch
        self._press(ord('C'))
        self._merge((50, 270), (200, 160))     # crack 4 main: crack 3 brings crack 1 along
        gid = self._cracks()[3]['crack_group']
        self.assertTrue(self._cracks()[3]['branch_main'])
        for i in (0, 2):
            self.assertEqual(self._cracks()[i]['crack_group'], gid)
            self.assertTrue(self._cracks()[i]['branch'])
            self.assertNotIn('branch_main', self._cracks()[i])

    def test_undo_removes_the_last_branch_and_keeps_the_mode(self):
        self._merge((150, 100), (100, 150), (200, 150))
        self._press(ord('u'))
        self.assertTrue(self._state()["active"])
        self.assertNotIn('crack_group', self._cracks()[2])
        self.assertEqual(self._cracks()[2]['path'][0], (200, 120))
        self.assertTrue(self._cracks()[1]['branch'])

    def test_json_group_and_flags_and_round_trip(self):
        self._merge((150, 100), (100, 150), (200, 150))
        lines = self._json_lines()
        gids = {s["group_id"] for s in lines[:3]}
        self.assertEqual(len(gids), 1)
        self.assertTrue(lines[0]["flags"]["branch_main"])
        self.assertTrue(lines[1]["flags"]["branch"] and lines[2]["flags"]["branch"])
        self.assertFalse(lines[3]["flags"].get("branch") or lines[3]["flags"].get("branch_main"))
        self.assertNotEqual(lines[3]["group_id"], lines[0]["group_id"])
        cracks, dets, ign = [], [], []
        for s in lines:
            self.app._load_one_shape(s, cracks, dets, ign)
        self.assertTrue(cracks[0]['branch_main'])
        self.assertTrue(cracks[1]['branch'] and cracks[2]['branch'])
        self.assertEqual(cracks[0]['crack_group'], cracks[2]['crack_group'])
        self.assertNotIn('crack_group', cracks[3])

    def test_mask_covers_the_bridge(self):
        self._merge((150, 100), (200, 150))
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(mask)
        self.assertGreater(mask[110, 200], 0, "the bridged gap is part of the crack mask")


class TestGui(unittest.TestCase):

    def setUp(self):
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_sidebar_button_and_shift_click(self):
        buttons = self.app._sidebar.buttons_by_code
        self.assertIn(cbm.MERGE_BRANCH_CODE, buttons)
        self.app._pending_keys.clear()
        buttons[cbm.MERGE_BRANCH_CODE].command()
        self.assertEqual(self.app._pending_keys, [cbm.MERGE_BRANCH_CODE])
        self.app._pending_keys.clear()
        result = buttons[cbm.MERGE_BRANCH_CODE].fire("<Shift-Button-1>")
        self.assertEqual(result, "break")
        self.assertEqual(self.app._pending_keys, [cbm.MERGE_BRANCH_EXIT_CODE])

    def test_shift_c_key_code(self):
        key = types.SimpleNamespace(keysym="C", char="C", state=0x1)
        self.assertEqual(self.gm.tk_key_event_to_code(key), ord('C'))

    def test_tools_menu_entries(self):
        codes = [code for _, code in self.gm._TOOL_COMMANDS]
        self.assertIn(cbm.MERGE_BRANCH_CODE, codes)
        self.assertIn(cbm.MERGE_BRANCH_EXIT_CODE, codes)


if __name__ == "__main__":
    unittest.main()
