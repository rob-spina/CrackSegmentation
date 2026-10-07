"""test_v105_link.py

Headless tests for the LINK [3] tool added in 1.0.5: click the end of one crack and the
start of another (or vice versa) to merge them into a single crack, saved right away
to the JSON (with a "links" record) and to the binary masks.

Run with:  python3 -m unittest test_v105_link -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from test_gui_shell_parity import make_app
from test_v104_fixes import _max_step

H, W = 200, 300


def _two_crack_app(tmpdir=None):
    """Two cracks on the same straight edge y=100, with a gap between x=120 and x=160."""
    app = make_app()
    app.cfg.H_img, app.cfg.W_img = H, W
    app.cfg.zoom_box = [0, 0, W, H]
    skel = np.zeros((H, W), dtype=np.uint8)
    cv2.line(skel, (20, 100), (280, 100), 255, 1)
    app.cfg.skeleton_mask = skel
    app.cfg.binary_mask = skel.copy()
    a = [(x, 100) for x in range(20, 121)]
    b = [(x, 100) for x in range(160, 281)]
    app.cfg.saved_cracks = [
        {'start': a[0], 'end': a[-1], 'path': list(a), 'active': True, 'session_id': 'current',
         'width_segments': [{'i0': 0, 'i1': 10, 'width_px': 3}]},
        {'start': b[0], 'end': b[-1], 'path': list(b), 'active': True, 'session_id': 'current',
         'width_segments': [{'i0': 100, 'i1': 110, 'width_px': 4}]},
    ]
    app.cfg.blue_visual_mask = np.zeros((H, W), dtype=np.uint8)
    app.cfg.green_visual_mask = np.zeros((H, W), dtype=np.uint8)
    if tmpdir:
        img = np.full((H, W, 3), 200, dtype=np.uint8)
        cv2.line(img, (20, 100), (280, 100), (30, 30, 30), 2)
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


class TestLinkMerge(unittest.TestCase):

    def _assert_merged(self, app):
        self.assertEqual(len(app.cfg.saved_cracks), 1, "the two cracks must become one")
        path = app.cfg.saved_cracks[0]['path']
        self.assertEqual(_max_step(path), 1, "the merged crack has no gaps")
        self.assertEqual({path[0], path[-1]}, {(20, 100), (280, 100)}, "both cracks are kept whole")
        self.assertIn((140, 100), path, "the gap is bridged")
        return app.cfg.saved_cracks[0]

    def test_end_of_first_then_start_of_second(self):
        app = _two_crack_app()
        self.assertTrue(app._link_cracks((0, 100), (1, 0)))
        crack = self._assert_merged(app)
        self.assertEqual(crack['path'][0], (20, 100))
        self.assertEqual(crack['links'], [{"from": [120, 100], "to": [160, 100], "method": "edge"}])

    def test_start_of_second_then_end_of_first(self):
        app = _two_crack_app()
        self.assertTrue(app._link_cracks((1, 0), (0, 100)))
        crack = self._assert_merged(app)
        self.assertEqual(crack['path'][0], (280, 100), "the first clicked crack leads, walked backwards")

    def test_width_tracts_follow_both_cracks(self):
        app = _two_crack_app()
        app._link_cracks((0, 100), (1, 0))
        crack = app.cfg.saved_cracks[0]
        path = crack['path']
        tracts = {(path[s['i0']], path[s['i1']], s['width_px']) for s in crack['width_segments']}
        self.assertEqual(tracts, {((20, 100), (30, 100), 3), ((260, 100), (270, 100), 4)})

    def test_crack_drawn_backwards_is_turned_around(self):
        app = _two_crack_app()
        b = app.cfg.saved_cracks[1]
        b['path'] = b['path'][::-1]   # drawn from x=280 to x=160: its END faces crack 1
        b['start'], b['end'] = b['path'][0], b['path'][-1]
        b['width_segments'] = [{'i0': 10, 'i1': 20, 'width_px': 4}]  # x=270..260
        self.assertTrue(app._link_cracks((0, 100), (1, 120)))   # end of 1, end of 2
        crack = self._assert_merged(app)
        path = crack['path']
        self.assertEqual(path[0], (20, 100))
        tracts = {(tuple(sorted((path[s['i0']], path[s['i1']]))), s['width_px']) for s in crack['width_segments']}
        self.assertEqual(tracts, {(((20, 100), (30, 100)), 3), (((260, 100), (270, 100)), 4)})

    def test_same_crack_is_refused(self):
        app = _two_crack_app()
        self.assertFalse(app._link_cracks((0, 0), (0, 100)))
        self.assertEqual(len(app.cfg.saved_cracks), 2)

    def test_no_edge_falls_back_to_a_straight_bridge(self):
        app = _two_crack_app()
        app.cfg.skeleton_mask = np.zeros((H, W), dtype=np.uint8)
        app.cfg.saved_cracks[1]['path'] = [(x, 140) for x in range(160, 281)]
        self.assertTrue(app._link_cracks((0, 100), (1, 0)))
        crack = app.cfg.saved_cracks[0]
        self.assertEqual(_max_step(crack['path']), 1)
        self.assertEqual(crack['links'][0]['method'], 'straight')

    def test_undo_restores_both_cracks(self):
        app = _two_crack_app()
        app._link_cracks((0, 100), (1, 0))
        app._undo_last_action()
        self.assertEqual(len(app.cfg.saved_cracks), 2)

    def test_a_later_cut_through_the_bridge_drops_its_record(self):
        app = _two_crack_app()
        app._link_cracks((0, 100), (1, 0))
        path = app.cfg.saved_cracks[0]['path']
        app._cut_crack_section((0, path.index((110, 100))), (0, path.index((170, 100))))
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        self.assertTrue(all(not c.get('links') for c in app.cfg.saved_cracks))


class TestLinkToolAndFiles(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_link_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_two_clicks_merge_save_and_close_the_tool(self):
        app = _two_crack_app(self.tmpdir)
        app.handle_keyboard(ord('3'))
        self.assertEqual(app.cfg.crack_cut_join_state["mode"], 'link')
        _click(app, 118, 102)   # near the END of crack 1
        _click(app, 163, 99)    # near the START of crack 2
        self.assertEqual(len(app.cfg.saved_cracks), 1)
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "a completed LINK closes the tool")

        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shapes = json.load(f)["shapes"]
        self.assertEqual(len(shapes), 1, "the JSON holds a single merged crack")
        self.assertEqual(shapes[0]["links"], [{"from": [120, 100], "to": [160, 100], "method": "edge"}])
        mask = cv2.imread(os.path.join(self.tmpdir, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(mask, "the binary crack mask is written right away")
        self.assertTrue(mask[100, 140], "the bridged gap is part of the binary mask")

        app.cfg.saved_cracks = []
        app.load_labelme_format()
        self.assertEqual(app.cfg.saved_cracks[0]['links'][0]['to'], [160, 100], "the record survives a reload")

    def test_click_in_the_middle_of_a_crack_picks_that_point(self):
        # Since 1.0.7 LINK also accepts a point on the blue line (see test_v107_features).
        app = _two_crack_app(self.tmpdir)
        app.handle_keyboard(ord('3'))
        _click(app, 70, 100)
        self.assertEqual(app.cfg.crack_cut_join_state["first"], (0, 50))
        self.assertFalse(os.path.exists(app.cfg.JSON_OUTPUT_PATH), "nothing is saved before the second click")

    def test_click_far_from_any_crack_is_ignored(self):
        app = _two_crack_app(self.tmpdir)
        app.handle_keyboard(ord('3'))
        _click(app, 70, 160)
        self.assertIsNone(app.cfg.crack_cut_join_state["first"])

    def test_indicator_text(self):
        app = _two_crack_app()
        app.handle_keyboard(ord('3'))
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_cut_join_indicator(win_out)
        self.assertTrue(win_out.any())
        app.handle_keyboard(ord('3'))
        self.assertIsNone(app.cfg.crack_cut_join_state["mode"], "pressing [3] again leaves the tool")


if __name__ == "__main__":
    unittest.main(verbosity=2)
