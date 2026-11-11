"""test_v201_sam_width.py -- 2.0.1 [width under the mouse] in SAM click to accept: the crack width (photo px, at the
current threshold) is shown next to the mouse; above SAM_ACCEPT_MAX_WIDTH_PX it turns red with the hint to use an
[I] ignore region instead of accepting it as a crack.

Run with:  python3 -m unittest test_v201_sam_width -v
"""
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

from test_v200_sam import _sam_app


def _map():
    """400 x 600 map: a 5 px crack, a 21 px crack, a 120 px lesion with a 3 px crack leaving it."""
    prob = np.zeros((400, 600), np.uint8)
    prob[48:53, 20:580] = 230          # 5 px
    prob[110:131, 20:300] = 230        # 21 px
    prob[200:321, 300:421] = 230       # 121 px
    prob[259:262, 421:590] = 230       # 3 px, leaving the lesion
    return prob


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = _sam_app(self.tmp)
        cfg = self.app.cfg
        cfg.SAM_ACCEPT_LOW_THRESHOLD = 0.3
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = 40
        cfg.sam_accept_state.update(active=True, prob=_map(), photo=cfg.CURRENT_IMAGE_PATH, window_drag=None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def width(self, x, y):
        return self.app._sam_width_at(x, y)


class TestMeasure(_Base):

    def test_thin_crack(self):
        w, capped = self.width(300, 50)
        self.assertEqual(w, 5)
        self.assertFalse(capped)

    def test_mouse_a_few_px_off_the_crack(self):
        self.assertEqual(self.width(300, 58)[0], 5)

    def test_medium_crack_from_its_edge(self):
        self.assertEqual(self.width(150, 129)[0], 21)

    def test_lesion_from_its_edge_and_centre(self):
        for x, y in ((305, 260), (360, 260)):
            w, _ = self.width(x, y)
            self.assertEqual(w, 121, (x, y))

    def test_thin_crack_leaving_the_lesion(self):
        self.assertEqual(self.width(560, 260)[0], 3)

    def test_nothing_near(self):
        self.assertIsNone(self.width(300, 380))
        self.assertIsNone(self.width(-5, 10))

    def test_threshold_counts(self):
        prob = self.app.cfg.sam_accept_state["prob"]
        prob[40:61, 100:200] = np.maximum(prob[40:61, 100:200], 100)   # a weak halo (0.39) around the 5 px crack
        self.app.cfg.SAM_ACCEPT_LOW_THRESHOLD = 0.5                    # halo below the threshold
        self.assertEqual(self.width(150, 50)[0], 5)
        self.app.cfg.SAM_ACCEPT_LOW_THRESHOLD = 0.3                    # halo above it: 21 px
        self.assertEqual(self.width(150, 50)[0], 21)

    def test_width_limit_off_still_measures(self):
        self.app.cfg.SAM_WIDTH_LIMIT_ON = False
        self.assertGreater(self.width(360, 260)[0], 100)


class TestLabel(_Base):

    def test_texts(self):
        self.assertEqual(self.app.sam_width_text(12, False), ("width 12 px", False))
        text, wide = self.app.sam_width_text(118, False)
        self.assertTrue(wide)
        self.assertIn("> 40", text)
        self.assertIn("[I] ignore", text)
        self.assertIn(">", self.app.sam_width_text(400, True)[0])
        self.app.cfg.SAM_ACCEPT_MAX_WIDTH_PX = 0
        self.assertFalse(self.app.sam_width_text(400, False)[1])

    def test_probe_updates_and_redraws_only_when_needed(self):
        app = self.app
        self.assertTrue(app._update_sam_width_probe(100, 100, 300, 50))
        self.assertEqual(app.cfg.sam_accept_state["width_probe"], (100, 100, 5, False))
        self.assertFalse(app._update_sam_width_probe(110, 100, 302, 50))    # same width, mouse close by
        self.assertTrue(app._update_sam_width_probe(100, 100, 360, 260))    # the lesion
        self.assertTrue(app._update_sam_width_probe(100, 100, 300, 380))    # off every crack
        self.assertIsNone(app.cfg.sam_accept_state["width_probe"])
        self.assertFalse(app._update_sam_width_probe(100, 100, 300, 380))

    def test_switch_off(self):
        self.app.cfg.SAM_SHOW_WIDTH = False
        self.assertFalse(self.app._update_sam_width_probe(100, 100, 300, 50))
        self.assertIsNone(self.app.cfg.sam_accept_state["width_probe"])

    def test_drawn_next_to_the_mouse(self):
        app = self.app
        app._update_sam_width_probe(200, 150, 360, 260)
        canvas = np.zeros((300, 1200, 3), np.uint8)
        app._draw_sam_width_tag(canvas)
        red = (canvas[:, :, 2] > 150) & (canvas[:, :, 0] < 50)
        ys, xs = np.nonzero(red)
        self.assertTrue(len(xs))
        self.assertGreaterEqual(xs.min(), 200)
        self.assertLess(abs(int(ys.mean()) - 140), 30)
        narrow = np.zeros((300, 500, 3), np.uint8)       # near the right edge: moved left, still whole
        app._draw_sam_width_tag(narrow)
        self.assertLess(np.nonzero(narrow[:, :, 2] > 150)[1].max(), 500)

    def test_not_drawn_while_dragging_a_window(self):
        app = self.app
        app._update_sam_width_probe(200, 150, 360, 260)
        app.cfg.sam_accept_state["window_drag"] = (0, 0, 10, 10)
        canvas = np.zeros((300, 500, 3), np.uint8)
        app._draw_sam_width_tag(canvas)
        self.assertFalse(canvas.any())

    def test_mouse_move_wiring(self):
        app = self.app
        with mock.patch.object(app, "transform_window_to_real_coords", return_value=(360, 260)), \
                mock.patch.object(app, "_update_sam_hover", return_value=False), \
                mock.patch.object(app, "refresh_zoom_viewport") as refresh:
            app._handle_sam_accept_mouse(cv2.EVENT_MOUSEMOVE, 50, 60)
        refresh.assert_called_once()
        self.assertEqual(app.cfg.sam_accept_state["width_probe"][:2], (50, 60))

    def test_hud_draws_it(self):
        app = self.app
        app._update_sam_width_probe(200, 150, 360, 260)
        with mock.patch.object(app, "_draw_sam_width_tag") as tag, \
                mock.patch.object(app, "_put_hud_text"), mock.patch.object(app, "_sam_class_counts",
                                                                            return_value={0: 0, 1: 0, 2: 0, 3: 0}):
            app._draw_sam_accept_label(np.zeros((50, 50, 3), np.uint8))
        tag.assert_called_once()


if __name__ == "__main__":
    unittest.main()
