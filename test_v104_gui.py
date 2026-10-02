"""test_v104_gui.py

GUI-side checks for v1.0.4 (uses tests_support/fake_tkinter.py, like test_gui_panel_wiring):
  - no bottom hint text in the GUI (menu, toolbar and status bar cover it);
  - the [J] panel follows the visible top of the scrolled canvas;
  - when zoomed, wheel / two-finger trackpad pans the photo (Shift = sideways);
  - a real <KeyPress> 'v' reaches the bulk retrace exactly like the toolbar button;
  - the new CUT [1] / JOIN [2] commands are in the menu and the sidebar.

Run with:  python3 -m unittest test_v104_gui -v
"""
import types
import unittest
from unittest import mock

import numpy as np

from test_gui_panel_wiring import gm, make_app


class TestGuiV104(unittest.TestCase):

    def test_bottom_hint_bar_is_not_drawn(self):
        app = make_app(self)
        win_out = np.zeros((900, 1200, 3), dtype=np.uint8)
        app._draw_bottom_hint_bar(win_out, app._compute_hud_font_metrics())
        self.assertFalse(win_out.any())

    def test_visible_top_follows_the_canvas_scroll(self):
        app = make_app(self)
        app._canvas._content_offset = (0, 0)
        with mock.patch.object(app._canvas.canvas, "canvasy", return_value=250):
            self.assertEqual(app._visible_canvas_top(), 250)
        with mock.patch.object(app._canvas.canvas, "canvasy", return_value=0):
            self.assertEqual(app._visible_canvas_top(), 0)

    def _zoomed(self, app):
        app.cfg.W_img, app.cfg.H_img = 4000, 3000
        app.cfg.zoom_factor = 4.0
        app.cfg.zoom_center = [2000, 1500]
        app.refresh_zoom_viewport()

    def test_wheel_pans_when_zoomed(self):
        app = make_app(self)
        self._zoomed(app)
        y0, x0 = app.cfg.zoom_box[1], app.cfg.zoom_box[0]
        result = app._canvas._on_wheel(types.SimpleNamespace(num=None, delta=-120, state=0))
        self.assertEqual(result, "break")
        self.assertGreater(app.cfg.zoom_box[1], y0, "wheel down moves the view down")
        app._canvas._on_wheel(types.SimpleNamespace(num=None, delta=-120, state=0x0001))
        self.assertGreater(app.cfg.zoom_box[0], x0, "Shift + wheel moves the view sideways")

    def test_wheel_scrolls_the_canvas_when_not_zoomed(self):
        app = make_app(self)
        app.cfg.zoom_factor = 1.0
        with mock.patch.object(app._canvas.canvas, "yview_scroll") as scroll:
            app._canvas._on_wheel(types.SimpleNamespace(num=None, delta=-120, state=0))
        scroll.assert_called_once()

    def test_v_keypress_runs_the_bulk_retrace_like_the_button(self):
        app = make_app(self)
        app._tk_root.fire("<KeyPress>", keysym="v", char="v")
        self.assertEqual(app._pending_keys, [ord('v')])
        with mock.patch.object(app, "_bulk_retrace_imported_cracks") as retrace:
            app.process_keypress(app._pending_keys.pop(0))
        retrace.assert_called_once()

    def test_cut_and_join_commands_are_in_menu_and_sidebar(self):
        menu_codes = [code for _, commands in gm._ALL_MENU_GROUPS for _, code in commands]
        sidebar_codes = [code for *_, code in gm._SIDEBAR_SHORTCUTS]
        for code in (ord('1'), ord('2')):
            self.assertIn(code, menu_codes)
            self.assertIn(code, sidebar_codes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
