"""test_v201_sam_cut.py -- 2.0.1 [Cut map] (K) in SAM click to accept: a line dragged across the crack map splits it,
so a crack joined to something else (tree branches ...) becomes two regions and a click takes only the crack.

The stand-in map (test_v200_sam): a horizontal crack y=100 (x 20-280) with a branch going down at x=150 (y 100-180),
one connected region; a click on it takes both.

Run with:  python3 -m unittest test_v201_sam_cut -v
"""
import unittest
from unittest import mock

import cv2
import numpy as np

import sam_accept
from sam_accept import SAM_CUT_CODE, SAM_EXCLUDE_CODE, SAM_MODE_KEYS
from test_v200_sam import _press
from test_v200_sam_accept import _Base


class _CutBase(_Base):

    def setUp(self):
        super().setUp()
        self._on()
        self._ident = mock.patch.object(self.app, "transform_window_to_real_coords",
                                        side_effect=lambda x, y: (float(x), float(y)))
        self._ident.start()
        self._refresh = mock.patch.object(self.app, "refresh_zoom_viewport")
        self._refresh.start()

    def tearDown(self):
        self._refresh.stop()
        self._ident.stop()
        super().tearDown()

    @property
    def state(self):
        return self.app.cfg.sam_accept_state

    def drag(self, points):
        app = self.app
        app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, *points[0])
        for p in points[1:]:
            app._handle_sam_accept_mouse(cv2.EVENT_MOUSEMOVE, *p)
        app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONUP, *points[-1])

    def label_at(self, x, y):
        return int(self.state["labels"][y, x])


class TestToggle(_CutBase):

    def test_key_and_code(self):
        self.assertEqual(SAM_MODE_KEYS[ord('k')], SAM_CUT_CODE)
        self.assertIn(SAM_CUT_CODE, sam_accept.SAM_MODE_CODES)
        _press(self.app, SAM_CUT_CODE)
        self.assertTrue(self.state["cut"])
        self.assertTrue(self.state["active"])            # the mode stays on
        _press(self.app, SAM_CUT_CODE)
        self.assertFalse(self.state["cut"])

    def test_exclusive_with_exclude(self):
        _press(self.app, SAM_EXCLUDE_CODE)
        _press(self.app, SAM_CUT_CODE)
        self.assertTrue(self.state["cut"])
        self.assertFalse(self.state["exclude"])
        _press(self.app, SAM_EXCLUDE_CODE)
        self.assertFalse(self.state["cut"])

    def test_off_with_the_mode(self):
        _press(self.app, SAM_CUT_CODE)
        self.app._exit_sam_accept("test")
        self.assertFalse(self.state["cut"])
        self.assertIsNone(self.state["cut_mask"])


class TestCut(_CutBase):

    def test_before_one_region(self):
        self.assertEqual(self.label_at(60, 100), self.label_at(150, 160))

    def test_cut_splits_the_branch_off(self):
        _press(self.app, SAM_CUT_CODE)
        self.drag([(140, 110), (145, 110), (150, 110), (155, 110), (160, 110)])
        main, branch = self.label_at(60, 100), self.label_at(150, 160)
        self.assertTrue(main and branch)
        self.assertNotEqual(main, branch)
        self.assertEqual(int(self.state["prob"][110, 150]), 0)
        self.assertIsNotNone(self.state["cut_mask"])
        self.assertIsNone(self.state["cut_path"])

    def test_click_then_takes_only_the_crack(self):
        _press(self.app, SAM_CUT_CODE)
        self.drag([(140, 110), (160, 110)])
        _press(self.app, SAM_CUT_CODE)                    # back to accepting
        self.assertTrue(self._click(60, 100))
        ys = [p[1] for c in self.app.cfg.saved_cracks for p in c["path"]]
        self.assertTrue(ys)
        self.assertLess(max(ys), 110)                     # the branch below the cut is not in it (a stub may be)
        self.assertTrue(self._click(150, 160))            # the branch is still there, on its own
        self.assertGreater(max(p[1] for c in self.app.cfg.saved_cracks for p in c["path"]), 150)

    def test_freehand_curve(self):
        _press(self.app, SAM_CUT_CODE)
        self.drag([(135, 120), (142, 114), (150, 112), (158, 114), (165, 120)])
        self.assertNotEqual(self.label_at(60, 100), self.label_at(150, 160))

    def test_cut_survives_a_new_threshold(self):
        _press(self.app, SAM_CUT_CODE)
        self.drag([(140, 110), (160, 110)])
        self.app._set_sam_threshold(self.app.cfg.SAM_ACCEPT_LOW_THRESHOLD - 0.1)
        self.assertNotEqual(self.label_at(60, 100), self.label_at(150, 160))

    def test_the_cached_map_is_not_changed(self):
        cached = self.state["prob"]
        before = cached.copy()
        _press(self.app, SAM_CUT_CODE)
        self.drag([(140, 110), (160, 110)])
        self.assertTrue(np.array_equal(cached, before))  # a private copy was cut

    def test_a_click_does_not_cut_or_accept(self):
        _press(self.app, SAM_CUT_CODE)
        self.drag([(60, 100), (61, 100)])
        self.assertEqual(self.app.cfg.saved_cracks, [])
        self.assertEqual(self.label_at(60, 100), self.label_at(150, 160))

    def test_line_away_from_the_map(self):
        _press(self.app, SAM_CUT_CODE)
        self.assertEqual(self.app._apply_sam_cut([(10, 190), (60, 190)]), 0)

    def test_zoomed_out_cut_is_thicker(self):
        self.app.transform_window_to_real_coords.side_effect = lambda x, y: (4.0 * x, 4.0 * y)
        _press(self.app, SAM_CUT_CODE)
        self.drag([(35, 28), (40, 28)])                   # photo (140, 112) -> (160, 112)
        cleared = np.count_nonzero(self.state["cut_mask"][:, 150])
        self.assertGreaterEqual(cleared, 12)              # 3 screen px = 12 photo px

    def test_accepted_cracks_are_not_cut(self):
        self.assertTrue(self._click(60, 100))
        n = len(self.app.cfg.saved_cracks)
        _press(self.app, SAM_CUT_CODE)
        self.drag([(140, 110), (160, 110)])
        self.assertEqual(len(self.app.cfg.saved_cracks), n)

    def test_drawn_while_dragging(self):
        _press(self.app, SAM_CUT_CODE)
        self.app._handle_sam_accept_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
        self.app._handle_sam_accept_mouse(cv2.EVENT_MOUSEMOVE, 60, 40)
        canvas = np.zeros((100, 100, 3), np.uint8)
        self.app._draw_sam_cut_path(canvas)
        self.assertTrue(((canvas[:, :, 0] > 200) & (canvas[:, :, 2] > 200)).any())   # magenta


class TestGui(unittest.TestCase):

    def test_button_and_menu(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        import crack_segmentation_gui as gm
        self.assertTrue(any(code == SAM_CUT_CODE for _, code in gm._TOOL_COMMANDS))
        self.assertIn(SAM_CUT_CODE, gm._SIDEBAR_ENABLED_STATES)
        with open(gm.__file__, encoding="utf-8") as f:
            self.assertIn('"Cut map", "K", SAM_CUT_CODE', f.read())


if __name__ == "__main__":
    unittest.main()
