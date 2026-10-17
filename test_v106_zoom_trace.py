"""test_v106_zoom_trace.py

USER REPORT (v1.0.6 Zoom window): after zooming onto a crack with the Zoom window tool,
clicking did nothing -- the tool swallowed every mouse event, so no crack could be traced.
Now only a press-drag-release zooms; plain clicks and hover moves reach the current tool.

Run with:  python3 -m unittest test_v106_zoom_trace -v
"""
import unittest
from unittest import mock

import cv2

from test_v105_link import _two_crack_app, H, W


def _click(app, x, y):
    """A real click as Tk/OpenCV deliver it: press then release on the same spot."""
    wx, wy = app.transform_real_to_window_coords(x, y)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)
    app.mouse_callback(cv2.EVENT_LBUTTONUP, wx, wy, 0, None)


def _drag(app, p1, p2):
    a = app.transform_real_to_window_coords(*p1)
    b = app.transform_real_to_window_coords(*p2)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, a[0], a[1], 0, None)
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, b[0], b[1], 0, None)
    app.mouse_callback(cv2.EVENT_LBUTTONUP, b[0], b[1], 0, None)


def _zoomed_onto_second_crack():
    """Zoom window on, the second crack removed, view zoomed onto where it was."""
    app = _two_crack_app()
    app.cfg.saved_cracks.pop()
    app.recalculate_masks()
    app.process_keypress(15)
    _drag(app, (160, 70), (280, 130))
    return app


class TestTracingInsideZoomWindow(unittest.TestCase):

    def test_drag_zooms_onto_the_rectangle(self):
        app = _zoomed_onto_second_crack()
        self.assertTrue(app.cfg.zoom_window_state["active"])
        self.assertGreater(app.cfg.zoom_factor, 2.0)
        self.assertEqual(len(app.cfg.saved_cracks), 1, "zooming must not trace anything")

    def test_two_clicks_trace_a_crack_while_zoomed(self):
        app = _zoomed_onto_second_crack()
        zoom = app.cfg.zoom_factor
        with mock.patch.object(app, "refresh_zoom_viewport", wraps=app.refresh_zoom_viewport):
            _click(app, 170, 100)
            self.assertIsNotNone(app.cfg.temp_start, "first click starts the trace")
            _click(app, 270, 100)
        self.assertEqual(len(app.cfg.saved_cracks), 2)
        path = app.cfg.saved_cracks[-1]['path']
        self.assertTrue(all(abs(y - 100) <= 1 for _, y in path), "the trace follows the real edge")
        self.assertEqual(app.cfg.zoom_factor, zoom, "tracing leaves the zoom unchanged")
        self.assertTrue(app.cfg.zoom_window_state["active"], "the tool stays on: dragging again zooms further")

    def test_hover_reaches_the_live_trace_preview(self):
        app = _zoomed_onto_second_crack()
        _click(app, 170, 100)
        wx, wy = app.transform_real_to_window_coords(240, 100)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, wx, wy, 0, None)
        self.assertEqual(len(app.cfg.temp_path), 2)
        self.assertLessEqual(abs(app.cfg.temp_path[-1][0] - 240), 1)

    def test_dragging_again_zooms_further_without_tracing(self):
        app = _zoomed_onto_second_crack()
        zoom = app.cfg.zoom_factor
        _drag(app, (200, 90), (240, 110))
        self.assertGreater(app.cfg.zoom_factor, zoom)
        self.assertIsNone(app.cfg.temp_start)
        self.assertEqual(len(app.cfg.saved_cracks), 1)

    def test_other_tools_also_get_plain_clicks(self):
        app = _two_crack_app()
        app.process_keypress(15)
        _drag(app, (10, 60), (140, 140))
        app.handle_keyboard(ord('1'))  # Cut tool
        _click(app, 40, 100)
        _click(app, 100, 100)
        self.assertEqual(len(app.cfg.saved_cracks), 3, "Cut split the first crack in two")


if __name__ == "__main__":
    unittest.main(verbosity=2)
