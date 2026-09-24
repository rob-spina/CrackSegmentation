"""test_gui_panel_wiring.py

Verifies crack_segmentation_gui.py's embedded single-window app:

  A. Menu bar wiring: every one of the ~37 commands queues exactly its
     declared key code into _pending_keys (same principle as the previous
     side-panel delivery, adapted to the menu-only layout).
  B. Toolbar shortcut buttons: same guarantee, for the smaller "most
     common actions" row.
  C. tk_key_event_to_code(): the Tk-keypress -> cv2-style integer code
     translation, covering letters (both cases), digits, punctuation,
     space, backspace/delete, enter, escape, and arrows -- pure logic, no
     Tk/display needed at all.
  D. EmbeddedCanvas: mouse events (<Button-1>/<ButtonRelease-1>/<Motion>)
     call app.mouse_callback() with the exact cv2 event constants and raw
     (x, y) Tk gives it -- no coordinate scaling, matching the fixed
     1200x900 canvas built to exactly match transform_*_coords()'s own
     assumption. update_frame() exercises the REAL cv2.cvtColor()/
     PIL.Image.fromarray() conversion (only the final Tk PhotoImage step
     is stubbed -- see tests_support/fake_tkinter.py).
  E. GuiCrackSegmentation's hooks (_create_display_window, _display_frame,
     _prompt_*, _report_fatal_error, window-close -> ESC).

This machine's sandbox has no display and no Tk installed, which is why
this uses tests_support/fake_tkinter.py -- a stand-in that only records
what was called/fires the same bind() callbacks a real widget would. On a
real Mac or Windows machine (where Tk ships with the standard python.org
installer), crack_segmentation_gui.py itself needs no changes to run for
real; this test only needs the stub because THIS sandbox lacks Tk.

Run with:  python3 -m unittest test_gui_panel_wiring -v
"""
import os
import sys
import time
import unittest
from collections import Counter
from unittest import mock

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers sys.modules['tkinter']/['PIL.ImageTk'] before the real import
import fake_fitz

from config import Config
import crack_segmentation_gui as gm


def _all_declared_commands():
    declared = []
    for _, commands in gm._ALL_MENU_GROUPS:
        declared.extend(commands)
    declared.append(("\u2753 Show user manual\t?", ord('?')))
    return declared


def _walk_menu(menu, acc):
    for kind, label, target in menu.items:
        if kind == "command":
            acc.append((label, target))
        elif kind == "cascade":
            _walk_menu(target, acc)
    return acc


def make_app(testcase=None):
    """Creates a GuiCrackSegmentation for a test. GuiCrackSegmentation.
    __init__ replaces sys.stdout/sys.stderr globally (to feed the status
    bar, see _StatusBarStream) -- when `testcase` is given, the original
    streams are restored via addCleanup() so one test's app doesn't leak
    a wrapped stdout into the next test."""
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    app = gm.GuiCrackSegmentation(Config())
    if testcase is not None:
        def _restore():
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
        testcase.addCleanup(_restore)
    else:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return app


# ---------------------------------------------------------------------------
# A/B. Menu bar + toolbar wiring
# ---------------------------------------------------------------------------

class TestMenuAndToolbarWiring(unittest.TestCase):

    def setUp(self):
        self.app = make_app(self)
        self.declared = _all_declared_commands()

    def test_every_menu_group_and_item_has_an_icon_prefix(self):
        # Every label (menu-bar group names and each item inside them)
        # starts with a non-ASCII Unicode symbol acting as a lightweight
        # icon -- plain text, no bitmap image files needed. Guards against
        # someone accidentally stripping icons back out later.
        for group_name, commands in gm._ALL_MENU_GROUPS:
            self.assertGreater(ord(group_name[0]), 127, f"menu group {group_name!r} has no icon prefix")
            for label, _ in commands:
                self.assertGreater(ord(label[0]), 127, f"menu item {label!r} has no icon prefix")

    def test_no_duplicate_key_codes_among_the_grouped_commands(self):
        # Every grouped command must be distinct -- a collision would mean
        # two different actions firing the same code. The standalone
        # top-level "Aiuto" menu item is a DELIBERATE alias of the View
        # group's own "?" entry (same ord('?') on purpose) and is
        # intentionally excluded from this check.
        grouped_declared = []
        for _, commands in gm._ALL_MENU_GROUPS:
            grouped_declared.extend(commands)
        codes = [code for _, code in grouped_declared]
        dupes = {c: n for c, n in Counter(codes).items() if n > 1}
        self.assertEqual(dupes, {}, f"colliding synthetic key codes across grouped commands: {dupes}")

    def test_menu_bar_exposes_every_declared_command_exactly_once(self):
        menubar = self.app._tk_root.kw.get("menu")
        self.assertIsNotNone(menubar, "master.config(menu=...) was not called")
        menu_pairs = _walk_menu(menubar, [])
        self.assertEqual(len(menu_pairs), len(self.declared))
        for (label, callback), (declared_label, declared_code) in zip(menu_pairs, self.declared):
            self.assertEqual(label, declared_label)
            self.app._pending_keys.clear()
            callback()
            self.assertEqual(self.app._pending_keys, [declared_code],
                              f"menu item {label!r} queued {self.app._pending_keys}, expected [{declared_code}]")

    def test_toolbar_shortcuts_queue_their_declared_codes(self):
        toolbar = self.app._toolbar_frame
        buttons = [c for c in toolbar.children if isinstance(c, gm.ttk.Button)]
        self.assertGreaterEqual(len(buttons), 6, "expected several toolbar shortcut buttons")
        for btn in buttons:
            self.app._pending_keys.clear()
            btn.command()
            self.assertEqual(len(self.app._pending_keys), 1,
                              f"toolbar button {btn.text!r} should queue exactly one code")
            self.assertGreater(ord(btn.text[0]), 127, f"toolbar button {btn.text!r} has no icon prefix")

    def test_toolbar_is_horizontally_scrollable_and_holds_every_button(self):
        # USER REPORT: the last toolbar button (Manual) kept being
        # unreachable regardless of computed window width. Scrollable
        # means every button exists and is reachable no matter what.
        self.assertIsInstance(self.app._toolbar_canvas, gm.tk.Canvas)
        buttons = [c for c in self.app._toolbar_frame.children if isinstance(c, gm.ttk.Button)]
        expected = len(self.app._toolbar_shortcuts())
        self.assertEqual(len(buttons), expected)
        manual_btn = next((b for b in buttons if "Manual" in b.text), None)
        self.assertIsNotNone(manual_btn, "the Manual button must exist in the toolbar regardless of window width")

    def test_next_and_previous_image_toolbar_buttons_use_the_no_save_codes(self):
        # USER REQUEST: browse the queue without saving. Codes 4/5 are the
        # same synthetic, no-physical-key codes process_keypress() checks
        # for (see smart_segmentation.py) -- distinct from Enter/S, which
        # always save.
        toolbar = self.app._toolbar_frame
        buttons = {c.text: c for c in toolbar.children if isinstance(c, gm.ttk.Button)}
        next_btn = next(b for label, b in buttons.items() if "Next Image" in label)
        prev_btn = next(b for label, b in buttons.items() if "Previous Image" in label)

        self.app._pending_keys.clear()
        next_btn.command()
        self.assertEqual(self.app._pending_keys, [4])

        self.app._pending_keys.clear()
        prev_btn.command()
        self.assertEqual(self.app._pending_keys, [5])

    def test_navigation_buttons_disabled_at_queue_boundaries(self):
        # USER REQUEST: grey out Previous on the first image and Next on
        # the last, instead of letting the operator press a button with
        # nothing to do.
        self.app.cfg.image_queue = ["/a.png", "/b.png", "/c.png"]

        self.app.cfg.CURRENT_IMAGE_PATH = "/a.png"
        self.app._update_navigation_button_states()
        self.assertEqual(self.app._prev_image_button.kw.get("state"), "disabled")
        self.assertEqual(self.app._next_image_button.kw.get("state"), "normal")

        self.app.cfg.CURRENT_IMAGE_PATH = "/b.png"
        self.app._update_navigation_button_states()
        self.assertEqual(self.app._prev_image_button.kw.get("state"), "normal")
        self.assertEqual(self.app._next_image_button.kw.get("state"), "normal")

        self.app.cfg.CURRENT_IMAGE_PATH = "/c.png"
        self.app._update_navigation_button_states()
        self.assertEqual(self.app._prev_image_button.kw.get("state"), "normal")
        self.assertEqual(self.app._next_image_button.kw.get("state"), "disabled")

    def test_navigation_buttons_single_image_queue_disables_both(self):
        self.app.cfg.image_queue = ["/only.png"]
        self.app.cfg.CURRENT_IMAGE_PATH = "/only.png"
        self.app._update_navigation_button_states()
        self.assertEqual(self.app._prev_image_button.kw.get("state"), "disabled")
        self.assertEqual(self.app._next_image_button.kw.get("state"), "disabled")

    def test_navigation_buttons_left_alone_before_any_image_is_loaded(self):
        # CURRENT_IMAGE_PATH not yet in image_queue (very first frame,
        # before build_chronological_queue()/initialize_image_session()
        # have run) -- must not raise, must just leave the buttons as-is.
        self.app.cfg.image_queue = []
        self.app.cfg.CURRENT_IMAGE_PATH = None
        self.app._update_navigation_button_states()  # must not raise

    def test_send_appends_to_pending_keys_directly(self):
        self.app._pending_keys.clear()
        self.app._send(ord('x'))
        self.assertEqual(self.app._pending_keys, [ord('x')])

    def test_send_returns_keyboard_focus_to_the_canvas(self):
        # USER REPORT ("Enter doesn't work for Save and next"): clicking a
        # toolbar button gives IT keyboard focus in Tk, which can swallow
        # the next keypress before it reaches _on_keypress(). _send() is
        # the one place every button/menu click goes through, so
        # reclaiming focus there covers all of them at once.
        self.app._canvas.canvas.has_focus = False
        self.app._send(ord('s'))
        self.assertTrue(self.app._canvas.canvas.has_focus)

    def test_toolbar_buttons_do_not_take_keyboard_focus(self):
        # Defense in depth alongside the _send() fix above: buttons
        # shouldn't be able to grab focus via click or Tab in the first
        # place.
        toolbar = self.app._toolbar_frame
        buttons = [c for c in toolbar.children if isinstance(c, gm.ttk.Button)]
        self.assertTrue(buttons)
        for btn in buttons:
            self.assertEqual(btn.kw.get("takefocus"), 0, f"{btn.text!r} should have takefocus=0")

    def test_window_width_never_exceeds_the_screen_on_a_small_display(self):
        # USER REPORT: forcing the window wider than the screen (so a wide
        # toolbar never had to scroll) instead pushed the Shortcuts
        # sidebar's own right edge off-screen, with no way to reach it
        # (unlike the toolbar, which now scrolls -- see _build_toolbar).
        # The window must never exceed the screen, regardless of content.
        target_w, _, _ = self.app._compute_initial_window_geometry(screen_w=1280, screen_h=800)
        self.assertLessEqual(target_w, max(900, 1280 - 40))

    def test_window_width_matches_canvas_and_sidebar_when_the_screen_allows(self):
        # On a large screen, the window should size to exactly what the
        # canvas+sidebar need -- the toolbar no longer factors in at all,
        # since it scrolls horizontally on its own.
        target_w, _, _ = self.app._compute_initial_window_geometry(screen_w=2400, screen_h=1200)
        expected = gm.CANVAS_W + 24 + gm.SIDEBAR_WIDTH
        self.assertEqual(target_w, expected)

    def test_window_right_edge_never_lands_past_the_screen(self):
        # USER REPORT: the sidebar's right edge was pushed off-screen (no
        # scrollbar to reach it, unlike the toolbar) -- the geometric root
        # cause was x + target_w exceeding screen_w. Checked across a
        # range of screen sizes, not just one.
        for screen_w in (1024, 1280, 1440, 1920, 2560, 3440):
            target_w, _, x = self.app._compute_initial_window_geometry(screen_w=screen_w, screen_h=900)
            self.assertLessEqual(x + target_w, screen_w,
                                  f"window right edge exceeds the screen at screen_w={screen_w}")



# ---------------------------------------------------------------------------
# C. Keyboard translation (pure logic, no Tk needed)
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, keysym="", char=""):
        self.keysym = keysym
        self.char = char


class TestKeyEventTranslation(unittest.TestCase):

    def test_lowercase_letter(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="s", char="s")), ord('s'))

    def test_uppercase_letter_via_shift(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="S", char="S")), ord('S'))

    def test_digit(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="0", char="0")), ord('0'))

    def test_space(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="space", char=" ")), ord(' '))

    def test_punctuation_used_by_width_edit_keys(self):
        for ch in ["+", "-", "[", "]", "{", "}", "?"]:
            with self.subTest(ch=ch):
                self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym=ch, char=ch)), ord(ch))

    def test_backspace(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="BackSpace", char="\x08")), 8)

    def test_delete_matches_mac_delete_code(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Delete", char="\x7f")), 127)

    def test_enter_matches_the_13_10_check(self):
        code = gm.tk_key_event_to_code(_FakeEvent(keysym="Return", char="\r"))
        self.assertIn(code, (13, 10))

    def test_escape(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Escape", char="\x1b")), 27)

    def test_arrow_keys_use_the_alternate_codes(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Up")), 0)
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Down")), 1)
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Left")), 2)
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Right")), 3)

    def test_bare_modifier_key_is_ignored(self):
        # A lone Shift/Ctrl/Cmd press has no char and isn't an arrow key.
        self.assertIsNone(gm.tk_key_event_to_code(_FakeEvent(keysym="Shift_L", char="")))

    # --- USER REPORT: "Enter doesn't work for Save and next" -- still
    # true even after fixing button focus-stealing. Real root cause: this
    # function used to rely on event.char for Return/BackSpace/Delete/
    # Escape too, ASSUMING Tk would reliably populate it with the
    # matching ASCII control character -- not guaranteed for non-
    # printable keys the way it is for ordinary letters/digits. The tests
    # above all happened to pass BOTH keysym and a matching char, so they
    # never actually exercised the failure mode. These don't: char is
    # deliberately empty/absent, proving the code path taken is the
    # keysym lookup, not a char fallback that happened to also work. ---

    def test_enter_works_even_when_char_is_empty(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Return", char="")), 13)

    def test_keypad_enter_also_maps_to_13(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="KP_Enter", char="")), 13)

    def test_backspace_works_even_when_char_is_empty(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="BackSpace", char="")), 8)

    def test_delete_works_even_when_char_is_empty(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Delete", char="")), 127)

    def test_escape_works_even_when_char_is_empty(self):
        self.assertEqual(gm.tk_key_event_to_code(_FakeEvent(keysym="Escape", char="")), 27)


# ---------------------------------------------------------------------------
# D. Embedded canvas: mouse events + frame display
# ---------------------------------------------------------------------------

class TestEmbeddedCanvas(unittest.TestCase):

    def setUp(self):
        self.app = make_app(self)

    def test_canvas_logical_content_is_exactly_1200x900(self):
        # The canvas widget itself can now be smaller (scrollable
        # viewport, see EmbeddedCanvas's docstring) -- what must stay
        # exactly 1200x900 is its scrollregion, matching
        # transform_*_coords()'s baked-in assumption.
        self.assertEqual(self.app._canvas.canvas.kw.get("scrollregion"), (0, 0, gm.CANVAS_W, gm.CANVAS_H))
        self.assertEqual((gm.CANVAS_W, gm.CANVAS_H), (1200, 900))

    def test_no_horizontal_scrollbar(self):
        # USER REPORT ("the horizontal bar doesn't work by dragging"):
        # removed rather than fixed -- the window's minimum width is tied
        # to the toolbar's own required width (see
        # GuiCrackSegmentation.__init__), already wider than the 1200px
        # canvas content, so the viewport can never be narrower than the
        # content and a horizontal scrollbar could never have anything to
        # scroll to. Scoped to the PHOTO canvas specifically (not the
        # whole app tree): the PDF manual panel (see PdfManualPanel) has
        # its own horizontal scrollbar for a genuinely different reason
        # -- a rendered PDF page can be wider than the viewport, unlike
        # the fixed 1200x900 photo canvas.
        def collect_scrollbars(widget, acc):
            for child in getattr(widget, "children", []):
                if isinstance(child, gm.ttk.Scrollbar):
                    acc.append(child)
                collect_scrollbars(child, acc)
            return acc

        scrollbars = collect_scrollbars(self.app._canvas.container, [])
        horizontal = [s for s in scrollbars if s.orient == "horizontal"]
        self.assertEqual(horizontal, [], "no horizontal scrollbar should exist on the photo canvas")

    def test_click_calls_mouse_callback_with_lbuttondown_and_raw_xy(self):
        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            self.app._canvas.canvas.fire("<Button-1>", x=123, y=456)
        self.assertEqual(calls, [(cv2.EVENT_LBUTTONDOWN, 123, 456, 0, None)])

    def test_release_calls_mouse_callback_with_lbuttonup(self):
        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            self.app._canvas.canvas.fire("<ButtonRelease-1>", x=10, y=20)
        self.assertEqual(calls, [(cv2.EVENT_LBUTTONUP, 10, 20, 0, None)])

    def test_motion_calls_mouse_callback_with_mousemove(self):
        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            self.app._canvas.canvas.fire("<Motion>", x=5, y=7)
        self.assertEqual(calls, [(cv2.EVENT_MOUSEMOVE, 5, 7, 0, None)])

    def test_no_coordinate_scaling_is_applied_at_the_origin(self):
        # With no scroll offset (the stub's canvasx/canvasy are identity
        # functions), raw Tk event.x/event.y reach mouse_callback
        # unscaled -- transform_window_to_real_coords() (unchanged, see
        # smart_segmentation.py) already assumes a 1200x900 window, which
        # is exactly what the canvas's scrollregion is built as.
        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            self.app._canvas.canvas.fire("<Button-1>", x=999, y=888)
        self.assertEqual(calls[0][1], 999)
        self.assertEqual(calls[0][2], 888)

    def test_photo_is_centered_when_the_viewport_is_wider_than_the_content(self):
        # USER REQUEST: avoid an overly extended black border on one
        # side, same fix as PdfManualPanel's page centering -- a
        # viewport wider than the fixed 1200x900 content should split
        # the extra space evenly on both sides instead of leaving the
        # photo pinned to the top-left corner.
        self.app._canvas.canvas._test_width = 1600
        self.app._canvas.canvas._test_height = 900
        frame = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._canvas.update_frame(frame)
        [image_item] = [v for v in self.app._canvas.canvas._items.values() if "image" in v]
        self.assertEqual((image_item["x"], image_item["y"]), ((1600 - 1200) // 2, 0))

    def test_clicks_still_map_onto_the_correct_photo_pixel_once_centered(self):
        # The whole point of also updating _canvas_xy(): a click at the
        # SAME on-screen spot must land on the SAME photo pixel whether
        # or not the photo is currently centered.
        self.app._canvas.canvas._test_width = 1600
        self.app._canvas.canvas._test_height = 900
        frame = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._canvas.update_frame(frame)  # centers at x_offset=200

        calls = []
        with mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            # Widget-x=200 is the photo's own on-screen left edge, now
            # shifted right by the centering offset -- must map to
            # photo-relative x=0, not x=200.
            self.app._canvas.canvas.fire("<Button-1>", x=200, y=50)
        self.assertEqual(calls, [(cv2.EVENT_LBUTTONDOWN, 0, 50, 0, None)])

    def test_no_centering_when_the_viewport_matches_the_content_exactly(self):
        self.app._canvas.canvas._test_width = 1200
        self.app._canvas.canvas._test_height = 900
        frame = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._canvas.update_frame(frame)
        [image_item] = [v for v in self.app._canvas.canvas._items.values() if "image" in v]
        self.assertEqual((image_item["x"], image_item["y"]), (0, 0))

    def test_content_offset_never_goes_negative_in_a_narrower_viewport(self):
        self.app._canvas.canvas._test_width = 400  # narrower than 1200
        self.app._canvas.canvas._test_height = 900
        frame = np.zeros((900, 1200, 3), dtype=np.uint8)
        self.app._canvas.update_frame(frame)
        [image_item] = [v for v in self.app._canvas.canvas._items.values() if "image" in v]
        self.assertEqual((image_item["x"], image_item["y"]), (0, 0))

    def test_coordinates_go_through_canvasx_canvasy_for_scroll_correctness(self):
        # USER REQUEST (scrollbars on the photo): once the canvas can be
        # scrolled, a raw widget-relative event.x/event.y is WRONG the
        # moment the view isn't at the origin -- canvasx()/canvasy() are
        # what convert it into the canvas's own logical coordinate space.
        # This proves that conversion is actually being used, not just
        # bypassed, by making it return something different from the
        # untranslated widget coordinate and checking mouse_callback gets
        # the TRANSLATED value.
        calls = []
        with mock.patch.object(self.app._canvas.canvas, "canvasx", return_value=321), \
             mock.patch.object(self.app._canvas.canvas, "canvasy", return_value=654), \
             mock.patch.object(self.app, "mouse_callback", side_effect=lambda *a: calls.append(a)):
            self.app._canvas.canvas.fire("<Button-1>", x=10, y=20)
        self.assertEqual(calls[0][1], 321)
        self.assertEqual(calls[0][2], 654)

    def test_update_frame_runs_real_bgr_to_rgb_conversion(self):
        # Build a small BGR frame with a known blue pixel (255, 0, 0 in
        # BGR -- see render_scene()'s crack-overlay paint color) and
        # confirm the REAL cv2.cvtColor()/PIL.Image.fromarray() pipeline
        # produces the correctly-converted RGB pixel (0, 0, 255).
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[0, 0] = (255, 0, 0)  # BGR blue
        fake_tkinter._FakePhotoImage._created.clear()

        self.app._canvas.update_frame(frame)

        self.assertEqual(len(fake_tkinter._FakePhotoImage._created), 1)
        pil_image = fake_tkinter._FakePhotoImage._created[0]
        self.assertEqual(pil_image.size, (4, 4))
        self.assertEqual(pil_image.getpixel((0, 0)), (0, 0, 255))  # RGB blue

    def test_update_frame_reuses_the_same_canvas_item(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.app._canvas.update_frame(frame)
        first_id = self.app._canvas._image_id
        self.app._canvas.update_frame(frame)
        second_id = self.app._canvas._image_id
        self.assertEqual(first_id, second_id, "update_frame() should reconfigure the existing "
                                               "canvas image, not create a new one every call")

    def test_click_gives_canvas_keyboard_focus(self):
        self.app._canvas.canvas.fire("<Button-1>", x=1, y=1)
        self.assertTrue(getattr(self.app._canvas.canvas, "has_focus", False))

    def test_wheel_scrolls_the_photo_view_when_help_is_closed(self):
        self.app.cfg.show_help_menu = False
        self.app._canvas.canvas.fire("<Enter>")  # establishes the bind_all wheel binding
        with mock.patch.object(self.app._canvas.canvas, "yview_scroll") as myv:
            self.app._canvas.canvas.fire("<MouseWheel>", delta=120)
        myv.assert_called_once()
        self.assertEqual(self.app._pending_keys, [], "must not touch the key queue while help is closed")

    def test_wheel_scrolls_the_help_list_when_help_is_open(self):
        self.app.cfg.show_help_menu = True
        self.app._canvas.canvas.fire("<Enter>")
        with mock.patch.object(self.app._canvas.canvas, "yview_scroll") as myv:
            self.app._canvas.canvas.fire("<MouseWheel>", delta=120)   # scroll up
            self.app._canvas.canvas.fire("<MouseWheel>", delta=-120)  # scroll down
        myv.assert_not_called()
        self.assertEqual(self.app._pending_keys, [0, 1], "delta>0 should queue the Up code, delta<0 the Down code")

    def test_wheel_linux_button_4_and_5_also_respect_help_state(self):
        self.app.cfg.show_help_menu = True
        self.app._canvas.canvas.fire("<Enter>")
        self.app._canvas.canvas.fire("<Button-4>", num=4)  # Linux scroll up
        self.app._canvas.canvas.fire("<Button-5>", num=5)  # Linux scroll down
        self.assertEqual(self.app._pending_keys, [0, 1])

    def test_wheel_binding_uses_bind_all_not_a_plain_widget_bind(self):
        # USER REPORT ("scrolling only works with the keyboard, not the
        # mouse"): a plain per-widget <MouseWheel> bind is known to be
        # unreliable across platforms -- the fix is bind_all(), active
        # only while the mouse is actually over the canvas (<Enter>/
        # <Leave>). This proves the wheel handler is genuinely gone after
        # <Leave> (not just "reset"), and comes back after the next <Enter>.
        canvas = self.app._canvas.canvas
        canvas.fire("<Enter>")
        self.assertIn("<MouseWheel>", canvas._bindings)
        canvas.fire("<Leave>")
        self.assertNotIn("<MouseWheel>", canvas._bindings, "wheel binding must be released on <Leave>")
        canvas.fire("<Enter>")
        self.assertIn("<MouseWheel>", canvas._bindings, "wheel binding must come back on the next <Enter>")


# D2. PDF user manual panel ([?] key -- see PdfManualPanel)
# ---------------------------------------------------------------------------

class TestPdfManualPanel(unittest.TestCase):
    """The real rendering pipeline (PIL.Image.frombytes(), the stubbed
    ImageTk.PhotoImage(), canvas placement, scrollregion sizing) runs
    for real against tests_support/fake_fitz.py's fake PDF pages; only
    the PDF decoding step itself is faked (see that module's own
    docstring for why)."""

    def _make_panel(self, fake_fitz_module=None):
        root = fake_tkinter.Tk()
        panel = gm.PdfManualPanel(root, "/fake/path/manual.pdf")
        if fake_fitz_module is not None:
            patcher = mock.patch.dict(sys.modules, {"fitz": fake_fitz_module})
            patcher.start()
            self.addCleanup(patcher.stop)
        return panel

    def _collect_scrollbars(self, widget, acc=None):
        acc = [] if acc is None else acc
        for child in getattr(widget, "children", []):
            if isinstance(child, gm.ttk.Scrollbar):
                acc.append(child)
            self._collect_scrollbars(child, acc)
        return acc

    def test_panel_has_both_a_vertical_and_a_horizontal_scrollbar(self):
        # Unlike the photo canvas (vertical-only, see EmbeddedCanvas's
        # own test above), a rendered PDF page can be wider than the
        # viewport, so this one genuinely needs both.
        panel = self._make_panel()
        orients = sorted(s.orient for s in self._collect_scrollbars(panel.frame))
        self.assertEqual(orients, ["horizontal", "vertical"])

    def test_show_renders_every_page_exactly_once(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=3)
        panel = self._make_panel(fake_module)
        panel.show()
        self.assertEqual(fake_module.opened_paths, ["/fake/path/manual.pdf"])
        self.assertEqual(len(panel._photo_refs), 3, "one PhotoImage per page")
        self.assertTrue(fake_module.last_document.closed, "the document must be closed after rendering")

    def test_show_only_renders_on_the_first_call(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=2)
        panel = self._make_panel(fake_module)
        panel.show()
        panel.hide()
        panel.show()
        self.assertEqual(len(fake_module.opened_paths), 1, "a second show() must not re-render")

    def test_scrollregion_covers_the_full_stacked_page_height(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=2)
        panel = self._make_panel(fake_module)
        panel.show()
        scrollregion = panel.canvas.kw.get("scrollregion")
        self.assertIsNotNone(scrollregion, "scrollregion must be set after rendering")
        # Two 60px-tall pages plus at least one gap between them.
        self.assertGreaterEqual(scrollregion[3], 60 * 2 + 16)

    def test_missing_fitz_shows_a_fallback_message_instead_of_crashing(self):
        panel = self._make_panel()
        with mock.patch.dict(sys.modules, {"fitz": None}):
            panel.show()  # must not raise
        self.assertEqual(panel._photo_refs, [], "no pages rendered when fitz is unavailable")

    def test_prefers_the_modern_pymupdf_import_name_over_fitz(self):
        # Recent PyMuPDF versions print a deprecation warning on
        # `import fitz` ("Use `import pymupdf` instead") -- injecting
        # the fake module ONLY as 'pymupdf' (not 'fitz') confirms that
        # name is tried, and used, first.
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        panel = self._make_panel()
        with mock.patch.dict(sys.modules, {"pymupdf": fake_module}):
            panel.show()
        self.assertEqual(fake_module.opened_paths, ["/fake/path/manual.pdf"])

    def test_pages_are_horizontally_centered_in_a_wide_viewport(self):
        # USER REQUEST: avoid an overly extended dark border on one
        # side -- a page narrower than the viewport should have the
        # empty space split evenly on both sides, not dumped all on
        # the right (which flush-left placement, the previous
        # revision, produced on any window wider than one page).
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)  # FakePage defaults to 40x60
        panel = self._make_panel(fake_module)
        panel.canvas._test_width = 200
        panel.show()
        [item] = panel.canvas._items.values()
        self.assertEqual(item["x"], (200 - 40) // 2)

    def test_a_page_wider_than_the_viewport_is_not_pushed_off_screen(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        panel = self._make_panel(fake_module)
        panel.canvas._test_width = 10  # narrower than the 40px-wide fake page
        panel.show()
        [item] = panel.canvas._items.values()
        self.assertEqual(item["x"], 0, "a page wider than the viewport must stay flush left, not go negative")

    def test_scrollregion_width_is_never_narrower_than_the_viewport(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        panel = self._make_panel(fake_module)
        panel.canvas._test_width = 500
        panel.show()
        scrollregion = panel.canvas.kw.get("scrollregion")
        self.assertEqual(scrollregion[2], 500, "scrollregion must match the (wider) viewport, not just the page")

    def test_open_failure_shows_a_fallback_message_instead_of_crashing(self):
        fake_module = fake_fitz.FakeFitzModule(open_should_raise=RuntimeError("corrupt PDF"))
        panel = self._make_panel(fake_module)
        panel.show()  # must not raise
        self.assertEqual(panel._photo_refs, [])


# E. GuiCrackSegmentation hooks
# ---------------------------------------------------------------------------

class TestGuiCrackSegmentationHooks(unittest.TestCase):

    def setUp(self):
        self.app = make_app(self)
        # _create_display_window() permanently mutates the live cv2
        # module (cv2.getWindowImageRect + a _crackseg_...patched flag)
        # as a real side effect, by design (see its docstring) -- save
        # and restore both around every test in this class so tests can't
        # leak state into each other regardless of pass/fail/order.
        self._real_get_window_image_rect = cv2.getWindowImageRect
        self._real_patched_flag = getattr(cv2, "_crackseg_get_window_image_rect_patched", False)
        self.addCleanup(self._restore_cv2_get_window_image_rect)

    def _restore_cv2_get_window_image_rect(self):
        cv2.getWindowImageRect = self._real_get_window_image_rect
        cv2._crackseg_get_window_image_rect_patched = self._real_patched_flag

    def test_window_geometry_is_set_explicitly_with_width_and_height(self):
        # Regression test for two real-Mac reports where the status strip
        # ended up clipped below the visible window. The underlying fix is
        # now architectural (see EmbeddedCanvas's docstring: toolbar/status
        # bar packed with a fixed side before the canvas's own scrollable
        # container, which absorbs whatever room is left) rather than a
        # perfectly precise size calculation -- but the window is still
        # given an explicit, sane initial "WxH+X+Y" geometry (not just a
        # position) so it doesn't rely on Tk's own implicit sizing either.
        calls = []
        original_geometry = fake_tkinter.Tk.geometry

        def recording_geometry(self, spec=None, **kw):
            if spec is not None:
                calls.append(spec)
            return original_geometry(self, spec, **kw)

        fake_tkinter.Tk.geometry = recording_geometry
        try:
            app = make_app(self)
        finally:
            fake_tkinter.Tk.geometry = original_geometry

        explicit_size_calls = [c for c in calls if "x" in c and c.split("x")[0].isdigit()]
        self.assertTrue(explicit_size_calls, f"no explicit WxH geometry call found, only: {calls}")
        width_str, rest = explicit_size_calls[-1].split("x", 1)
        height_str = rest.split("+", 1)[0]
        self.assertGreater(int(width_str), 0)
        self.assertGreater(int(height_str), 0)

    def test_window_width_ignores_the_toolbars_own_required_width(self):
        # The toolbar scrolls horizontally now (see _build_toolbar), so its
        # own required width must no longer inflate the window at all --
        # unlike before, when an unusually wide toolbar forced the window
        # wider too (which is what pushed the sidebar off-screen).
        calls = []
        original_geometry = fake_tkinter.Tk.geometry
        original_frame_reqwidth = fake_tkinter.Frame.winfo_reqwidth

        def recording_geometry(self, spec=None, **kw):
            if spec is not None:
                calls.append(spec)
            return original_geometry(self, spec, **kw)

        fake_tkinter.Tk.geometry = recording_geometry
        fake_tkinter.Frame.winfo_reqwidth = lambda self: 1350
        try:
            app = make_app(self)
        finally:
            fake_tkinter.Tk.geometry = original_geometry
            fake_tkinter.Frame.winfo_reqwidth = original_frame_reqwidth

        width_str = calls[-1].split("x", 1)[0]
        screen_w, screen_h = app._get_screen_dimensions()
        expected, _, _ = app._compute_initial_window_geometry(screen_w, screen_h)
        self.assertEqual(int(width_str), expected,
                          "a wide toolbar must not change the window width anymore")


    def test_window_is_resizable_with_a_minimum_size(self):
        # USER REPORT: the window used to be locked non-resizable at a
        # fixed size, which is what made it possible for the total height
        # to exceed the screen in the first place. Now resizable, with a
        # floor so it can't be shrunk to something unusable.
        calls = {}
        original_resizable = fake_tkinter.Tk.resizable
        original_minsize = fake_tkinter.Tk.minsize

        def recording_resizable(self, *a, **kw):
            calls["resizable"] = a
            return original_resizable(self, *a, **kw)

        def recording_minsize(self, *a, **kw):
            calls["minsize"] = a
            return original_minsize(self, *a, **kw)

        fake_tkinter.Tk.resizable = recording_resizable
        fake_tkinter.Tk.minsize = recording_minsize
        try:
            make_app(self)
        finally:
            fake_tkinter.Tk.resizable = original_resizable
            fake_tkinter.Tk.minsize = original_minsize

        self.assertEqual(calls.get("resizable"), (True, True))
        self.assertIn("minsize", calls, "expected a minsize() call so the window can't shrink to nothing")

    def test_create_display_window_shrinks_real_window_and_tries_to_hide_it(self):
        # The real window is now made as small/inert as possible on a
        # best-effort basis -- correctness no longer depends on this
        # actually working (see test_get_window_image_rect_is_patched_*
        # below), only on it being unobtrusive if the OS still shows it.
        with mock.patch("cv2.namedWindow") as mnw, \
             mock.patch("cv2.resizeWindow") as mrw, \
             mock.patch("cv2.moveWindow") as mmw, \
             mock.patch("cv2.getWindowImageRect", return_value=(0, 0, gm.CANVAS_W, gm.CANVAS_H)):
            self.app._create_display_window("Crack Detector Workspace")
        mnw.assert_called_once_with("Crack Detector Workspace", cv2.WINDOW_NORMAL)
        mrw.assert_called_once_with("Crack Detector Workspace", 1, 1)
        mmw.assert_called_once()
        args = mmw.call_args[0]
        self.assertEqual(args[0], "Crack Detector Workspace")
        self.assertLess(args[1], 0)

    def test_get_window_image_rect_is_patched_to_ignore_the_real_hidden_window(self):
        # Regression test for a real-Mac report: the OS snapped the
        # "hidden" helper window back onto the visible screen instead of
        # keeping it off-screen. _create_display_window() must make
        # correctness independent of that -- cv2.getWindowImageRect()
        # should return the fixed canvas size for "Crack Detector
        # Workspace" NO MATTER what the real window reports.
        real_calls = []

        def fake_real_get_window_image_rect(name):
            real_calls.append(name)
            return (37, 42, 999, 999)  # deliberately "wrong" -- must never be used for our window

        with mock.patch("cv2.namedWindow"), mock.patch("cv2.resizeWindow"), mock.patch("cv2.moveWindow"), \
             mock.patch("cv2.getWindowImageRect", side_effect=fake_real_get_window_image_rect), \
             mock.patch("cv2._crackseg_get_window_image_rect_patched", False, create=True):
            self.app._create_display_window("Crack Detector Workspace")

            result = cv2.getWindowImageRect("Crack Detector Workspace")
            self.assertEqual(result, (0, 0, gm.CANVAS_W, gm.CANVAS_H))

            # A DIFFERENT window name (e.g. the unrelated "Save Warning"
            # popup) must still fall through to the real function untouched.
            other_result = cv2.getWindowImageRect("Save Warning")
            self.assertEqual(other_result, (37, 42, 999, 999))
            self.assertIn("Save Warning", real_calls)
        # Exiting the `with` block above restores the true original
        # cv2.getWindowImageRect (mock.patch's own job) -- nothing further
        # to clean up; the patch _create_display_window() applies for
        # real only lives for the duration of this `with` block here.

    def test_display_frame_routes_to_the_embedded_canvas(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with mock.patch.object(self.app._canvas, "update_frame") as mupd:
            self.app._display_frame("Crack Detector Workspace", frame)
        mupd.assert_called_once_with(frame)

    def test_pixel_scale_prompt_returns_askfloat_result(self):
        fake_tkinter.simpledialog.next_return = 7.5
        val = self.app._prompt_pixel_scale(0.05)
        self.assertEqual(val, 7.5)

    def test_calibration_distance_prompt_returns_askfloat_result(self):
        fake_tkinter.simpledialog.next_return = 12.0
        val = self.app._prompt_calibration_distance()
        self.assertEqual(val, 12.0)

    def test_report_fatal_error_shows_a_messagebox(self):
        fake_tkinter.messagebox.last_call = None
        self.app._report_fatal_error("No valid file found in: /some/path/Images")
        kind, title, message = fake_tkinter.messagebox.last_call
        self.assertEqual(kind, "error")
        self.assertIn("No valid file found", message)

    def test_mode_choice_dialog_returns_the_clicked_value(self):
        dialog = gm._ModeChoiceDialog(self.app._tk_root)
        dialog._choose("2")
        self.assertEqual(dialog.result, "2")

    def test_mode_choice_dialog_falls_back_to_1_on_window_close(self):
        dialog = gm._ModeChoiceDialog(self.app._tk_root)
        dialog._on_close()
        self.assertEqual(dialog.result, "1")

    def test_mode_choice_dialog_centers_itself_on_screen(self):
        # USER REQUEST: this used to appear at whatever default position Tk
        # picked -- now explicitly centered against the real screen size.
        dialog = gm._ModeChoiceDialog(self.app._tk_root)
        dialog.top._test_width, dialog.top._test_height = 400, 200
        dialog._center_on_screen()
        screen_w, screen_h = dialog.top.winfo_screenwidth(), dialog.top.winfo_screenheight()
        expected = f"+{(screen_w - 400) // 2}+{(screen_h - 200) // 2}"
        self.assertEqual(dialog.top.kw.get("geometry_arg"), expected)

    def test_prompt_mode_choice_hides_the_main_window_while_open_and_restores_it(self):
        # USER REQUEST: no other window should be visible/distracting while
        # this initial choice is up -- the main window is hidden, not
        # destroyed, and comes back right after the choice is made.
        with mock.patch.object(gm, "_ModeChoiceDialog") as mock_dialog_cls:
            mock_dialog = mock.Mock()
            mock_dialog.result = "1"
            mock_dialog.top = fake_tkinter.Toplevel(self.app._tk_root)
            mock_dialog_cls.return_value = mock_dialog
            self.assertFalse(getattr(self.app._tk_root, "withdrawn", False))
            self.app._prompt_mode_choice()
        self.assertFalse(self.app._tk_root.withdrawn, "the main window must be restored after the dialog closes")

    def test_window_close_queues_escape_not_a_silent_kill(self):
        self.app._pending_keys.clear()
        self.app._on_window_close()
        self.assertEqual(self.app._pending_keys, [27])

    def test_pump_extra_events_calls_tk_update(self):
        before = self.app._tk_root.updated if hasattr(self.app._tk_root, "updated") else 0
        self.app._pump_extra_events()
        self.assertEqual(self.app._tk_root.updated, before + 1)

    def test_wait_key_never_calls_the_real_cv2_waitkey(self):
        # CRASH REGRESSION TEST -- see _wait_key's docstring for the full
        # mechanism (confirmed via a real Mac crash log: cv2.waitKey()
        # pumps the whole process's native event loop, which can deliver
        # a Tk redraw callback outside of Tcl/Tk's expected calling
        # context and abort the interpreter). This must NEVER call the
        # real cv2.waitKey in GUI mode, for ANY delay value.
        with mock.patch("cv2.waitKey") as mwait:
            for delay in (0, 1, 30, 100):
                result = self.app._wait_key(delay)
                self.assertEqual(result, -1)
        mwait.assert_not_called()

    def test_acknowledge_save_warning_shows_a_dialog_not_a_cv2_popup(self):
        # Also implicitly a crash regression test: the default
        # implementation calls cv2.waitKey(0), which is unsafe here (see
        # test_wait_key_never_calls_the_real_cv2_waitkey) -- this GUI
        # override must replace it entirely, never touching cv2 at all.
        with mock.patch("cv2.namedWindow") as mnw, mock.patch("cv2.waitKey") as mwait:
            self.app._acknowledge_save_warning()
        mnw.assert_not_called()
        mwait.assert_not_called()

    def test_confirm_crack_filter_removal_shows_a_yes_no_dialog(self):
        with mock.patch("cv2.namedWindow") as mnw, mock.patch("cv2.waitKey") as mwait, \
             mock.patch("tkinter.messagebox.askyesno", return_value=True) as mask:
            result = self.app._confirm_crack_filter_removal("BLDG001", ["a.json", "b.json"], [0])
        mnw.assert_not_called()
        mwait.assert_not_called()
        mask.assert_called_once()
        self.assertTrue(result)

    def test_toggle_help_menu_shows_the_pdf_panel_not_the_overlay_flag(self):
        # The GUI override must swap panels, and must NOT touch
        # cfg.show_help_menu -- that flag drives the OLD text-overlay
        # rendering in smart_segmentation.py's render_scene(), unused
        # once the real PDF manual takes over in GUI mode.
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        with mock.patch.dict(sys.modules, {"fitz": fake_module}):
            self.app._toggle_help_menu()
        self.assertTrue(self.app._pdf_panel_visible)
        self.assertFalse(self.app.cfg.show_help_menu, "must not touch the headless-mode overlay flag")

    def test_toggle_help_menu_twice_returns_to_the_photo_canvas(self):
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        with mock.patch.dict(sys.modules, {"fitz": fake_module}):
            self.app._toggle_help_menu()
            self.app._toggle_help_menu()
        self.assertFalse(self.app._pdf_panel_visible)

    def test_toggle_fullscreen_resizes_the_tk_window_not_a_hidden_cv2_one(self):
        # USER REPORT: the base class's [F] targets the real, hidden
        # OpenCV window GUI mode never draws to -- toggling it made that
        # window pop up for real, fullscreen and blank. The GUI override
        # must only ever touch the Tk window's own geometry.
        with mock.patch("cv2.setWindowProperty") as mock_cv2_prop:
            self.app.cfg.is_fullscreen = False
            self.app._tk_root.geometry("900x700+50+50")
            self.app._toggle_fullscreen()
            self.assertTrue(self.app.cfg.is_fullscreen)
            screen_w, screen_h = self.app._get_screen_dimensions()
            self.assertEqual(self.app._tk_root.kw.get("geometry_arg"), f"{screen_w}x{screen_h}+0+0")
            self.app._toggle_fullscreen()
            self.assertFalse(self.app.cfg.is_fullscreen)
            self.assertEqual(self.app._tk_root.kw.get("geometry_arg"), "900x700+50+50")
        mock_cv2_prop.assert_not_called()

    def test_shortcuts_sidebar_is_a_child_of_the_middle_row_next_to_the_canvas(self):
        # USER REQUEST: a reference panel on the right of the Canvas for
        # shortcuts not already in the toolbar (e.g. N).
        self.assertIn(self.app._sidebar.frame, self.app._middle_row.children)
        self.assertIn(self.app._canvas.container, self.app._middle_row.children)

    def test_shortcuts_sidebar_lists_the_key_commands_not_in_the_toolbar(self):
        keys = [key for _, _, key, _ in gm._SIDEBAR_SHORTCUTS]
        self.assertIn("N", keys, "the user's own example (toggle blue crack overlay) must be listed")
        toolbar_codes = {code for _, code in self.app._toolbar_shortcuts()}
        for _, _, key, code in gm._SIDEBAR_SHORTCUTS:
            self.assertNotIn(code, toolbar_codes, f"{key!r} is already a toolbar button -- shouldn't be duplicated in the sidebar")

    def test_shortcuts_sidebar_buttons_are_real_and_clickable(self):
        # USER REPORT: the sidebar used to be a static text list -- "i
        # comandi non funzionano". Every entry must now be a real button
        # that sends its key code through the same path as a toolbar click.
        self.assertEqual(len(self.app._sidebar.buttons), len(gm._SIDEBAR_SHORTCUTS))
        _, label, key, code = gm._SIDEBAR_SHORTCUTS[0]
        btn = self.app._sidebar.buttons[0]
        self.assertIn(label, btn.text)
        self.assertIn(key, btn.text)
        self.app._pending_keys.clear()
        btn.command()
        self.assertEqual(self.app._pending_keys, [code])

    def test_shortcuts_sidebar_buttons_are_plain_tk_not_ttk(self):
        # USER REPORT (app wouldn't open on Mac at all): ttk::button has a
        # materially smaller option set than tk.Button -- passing an
        # option it doesn't support (wraplength, needed to keep the
        # sidebar's text from being clipped) raises a real TclError at
        # widget-creation time, crashing the whole app before any window
        # is even shown, with no console to reveal why on a double-clicked
        # .app. Every sidebar button must be a plain tk.Button, never ttk.
        for btn in self.app._sidebar.buttons:
            self.assertIsInstance(btn, gm.tk.Button)
            self.assertNotIsInstance(btn, gm.ttk.Button)

    def test_shortcuts_sidebar_is_vertically_scrollable_and_holds_every_button(self):
        # USER REPORT: the last sidebar button was clipped on a shorter
        # window, with no way to reach it. Scrollable means every button
        # exists and is reachable no matter the window height.
        self.assertIsInstance(self.app._sidebar._canvas, gm.tk.Canvas)
        self.assertEqual(len(self.app._sidebar.buttons), len(gm._SIDEBAR_SHORTCUTS))
        last_icon, last_label, last_key, _ = gm._SIDEBAR_SHORTCUTS[-1]
        last_btn = self.app._sidebar.buttons[-1]
        self.assertIn(last_label, last_btn.text)
        self.assertIn(last_key, last_btn.text)

    def test_shortcuts_sidebar_buttons_are_a_single_compact_line(self):
        # USER REQUEST: shorter buttons, with the [KEY] moved up next to
        # the label on one line instead of its own separate line.
        for btn in self.app._sidebar.buttons:
            self.assertNotIn("\n", btn.text, f"{btn.text!r} should be a single line, not two")

    def test_shortcuts_sidebar_includes_switch_mode_without_an_empty_key_badge(self):
        # USER REQUEST: put Switch Mode (removed from the toolbar to make
        # room for Manual) back within reach, in the sidebar. It has no
        # physical key (button-only, code 6) -- must not show as "[]".
        switch_btn = next(b for b in self.app._sidebar.buttons if "Switch Mode" in b.text)
        self.assertNotIn("[]", switch_btn.text)
        self.app._pending_keys.clear()
        switch_btn.command()
        self.assertEqual(self.app._pending_keys, [6])

    def test_ttk_button_rejects_wraplength_like_the_real_widget_does(self):
        # Guards the regression test above: confirms the fake ttk.Button
        # actually rejects the option real ttk::button doesn't support,
        # the same way real Tk raised the crash on a real Mac -- so this
        # class of bug is caught here next time, not after a delivery.
        with self.assertRaises(gm.tk.TclError):
            gm.ttk.Button(self.app._tk_root, text="x", wraplength=100)

    def test_question_mark_keypress_reaches_the_pdf_toggle(self):
        # Same code path a real '?' keypress, the Help menu item, and
        # the toolbar button all use (process_keypress() ->
        # handle_keyboard() -> _handle_keyboard_dispatch() in
        # smart_segmentation.py) -- exercises that whole chain, not
        # just a direct call to _toggle_help_menu().
        fake_module = fake_fitz.FakeFitzModule(n_pages=1)
        with mock.patch.dict(sys.modules, {"fitz": fake_module}):
            self.app.process_keypress(ord('?'))
        self.assertTrue(self.app._pdf_panel_visible)

    def test_manual_pdf_resolves_next_to_the_source_file_when_not_frozen(self):
        # USER REPORT ("non è possibile aprire il manuale, no such file
        # or directory"): this used to resolve via cfg.SCRIPT_DIR /
        # resolve_script_dir() (config.py), which is fine when running
        # from source (same folder either way) but wrong once packaged
        # -- see the next test.
        path = self.app._resolve_manual_pdf_path()
        expected_dir = os.path.dirname(os.path.abspath(gm.__file__))
        self.assertEqual(os.path.dirname(path), expected_dir)
        self.assertEqual(os.path.basename(path), "Crack_Segmentation_User_Manual.pdf")

    def test_manual_pdf_resolves_next_to_the_frozen_executable_not_the_user_data_folder(self):
        # A PyInstaller onedir build places bundled `datas` (see
        # packaging/crack_segmentation_gui.spec) next to the
        # executable itself -- NEVER in cfg.SCRIPT_DIR's per-user data
        # folder (~/Documents/CrackSegmentation, for Images/exports,
        # empty of anything PyInstaller bundled). This is the exact bug
        # that produced the "no such file or directory" report.
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(sys, "executable", "/opt/CrackSegmentation/CrackSegmentation", create=True):
            path = self.app._resolve_manual_pdf_path()
        self.assertEqual(path, "/opt/CrackSegmentation/Crack_Segmentation_User_Manual.pdf")
        self.assertNotIn("Documents", path)

    def test_echo_twin_logo_resolves_next_to_the_source_file(self):
        path = self.app._resolve_echo_twin_logo_path()
        expected_dir = os.path.dirname(os.path.abspath(gm.__file__))
        self.assertEqual(os.path.dirname(path), expected_dir)
        self.assertEqual(os.path.basename(path), "echo_twin_logo.png")

    def test_mode_choice_dialog_loads_a_real_logo_file(self):
        logo_path = self.app._resolve_echo_twin_logo_path()
        if not os.path.exists(logo_path):
            self.skipTest("echo_twin_logo.png not present in this checkout")
        dialog = gm._ModeChoiceDialog(self.app._tk_root, logo_path)
        self.assertIsNotNone(dialog._logo_photo, "a real logo file must produce a loadable image")

    def test_mode_choice_dialog_tolerates_a_missing_logo_file(self):
        # A missing/broken logo must never block the startup dialog itself.
        dialog = gm._ModeChoiceDialog(self.app._tk_root, "/no/such/logo.png")
        self.assertIsNone(dialog._logo_photo)


class TestStatusBar(unittest.TestCase):
    """StatusBar (the widget) and _StatusBarStream (the stdout/stderr tee
    that feeds it) -- both new in this delivery, added so console
    messages are visible inside the app itself (a double-clicked .app/
    .exe has no attached terminal at all)."""

    def test_font_is_readably_large_and_fewer_lines_are_shown(self):
        # USER REQUEST: bigger, more readable text, even at the cost of
        # showing fewer lines at once (traded 5 lines down to 2).
        bar = gm.StatusBar(fake_tkinter.Tk())
        self.assertGreaterEqual(gm.StatusBar.FONT_SIZE, 11)
        self.assertLessEqual(bar.text.kw.get("height"), 2)

    def test_add_message_appends_a_line(self):
        bar = gm.StatusBar(fake_tkinter.Tk())
        bar.add_message("hello")
        bar.add_message("world")
        self.assertEqual(bar.text.content, "hello\nworld\n")

    def test_history_is_bounded(self):
        bar = gm.StatusBar(fake_tkinter.Tk())
        bar.MAX_LINES = 3
        for i in range(5):
            bar.add_message(f"line {i}")
        self.assertEqual(bar._lines, ["line 2", "line 3", "line 4"])

    def test_scroll_buttons_call_yview_scroll(self):
        bar = gm.StatusBar(fake_tkinter.Tk())
        # scroll() just forwards to the Text widget's own yview_scroll();
        # the stub's default no-op is enough to prove the wiring here --
        # real scrolling behavior is Tk's own, not something this file
        # reimplements.
        bar.scroll(-1)
        bar.scroll(1)  # should not raise

    def test_stream_mirrors_to_both_real_stream_and_status_bar(self):
        bar = gm.StatusBar(fake_tkinter.Tk())

        class _FakeRealStream:
            def __init__(self):
                self.written = ""

            def write(self, text):
                self.written += text

            def flush(self):
                pass

        real = _FakeRealStream()
        stream = gm._StatusBarStream(real, bar)

        stream.write("[INFO] something happened\n")
        stream.write("[AVVISO] partial line without a newline yet")

        self.assertEqual(real.written, "[INFO] something happened\n[AVVISO] partial line without a newline yet")
        self.assertEqual(bar._lines, ["[INFO] something happened"])  # partial line not flushed yet

        stream.write("\n")
        self.assertEqual(bar._lines, ["[INFO] something happened", "[AVVISO] partial line without a newline yet"])

    def test_write_survives_a_broken_status_bar(self):
        # USER REPORT ("the status bar doesn't work"): if updating the Tk
        # widget ever raised for any reason, that exception used to
        # propagate straight out of write() -- and since print() calls
        # sys.stdout.write() under the hood, a single broken status-bar
        # update could silently break whatever application code was
        # trying to print(), not just the bar itself. write() must now
        # swallow that and let the real stream still receive the text.
        class _ExplodingStatusBar:
            def add_message(self, line):
                raise RuntimeError("simulated Tk failure")

        class _FakeRealStream:
            def __init__(self):
                self.written = ""

            def write(self, text):
                self.written += text

            def flush(self):
                pass

        real = _FakeRealStream()
        stream = gm._StatusBarStream(real, _ExplodingStatusBar())
        stream.write("[INFO] this must not raise\n")  # must not propagate RuntimeError
        self.assertEqual(real.written, "[INFO] this must not raise\n")

    def test_stream_works_with_no_real_stream_at_all(self):
        # A frozen "windowed" build (no console) can have sys.stdout be
        # None -- must not raise.
        bar = gm.StatusBar(fake_tkinter.Tk())
        stream = gm._StatusBarStream(None, bar)
        stream.write("hello\n")
        stream.flush()
        self.assertEqual(bar._lines, ["hello"])

    def test_blank_lines_are_not_added_to_the_bar(self):
        bar = gm.StatusBar(fake_tkinter.Tk())
        stream = gm._StatusBarStream(None, bar)
        stream.write("\n\n   \n")
        self.assertEqual(bar._lines, [])

    def test_isatty_is_false(self):
        stream = gm._StatusBarStream(None, gm.StatusBar(fake_tkinter.Tk()))
        self.assertFalse(stream.isatty())

    def test_status_bar_is_embedded_in_master_not_a_separate_window(self):
        root = fake_tkinter.Tk()
        bar = gm.StatusBar(root)
        self.assertFalse(hasattr(bar, "window"), "should be a fixed strip packed into master, not a Toplevel")
        self.assertIn(bar.text, self._collect_descendants(root))

    def test_text_widget_has_a_scrollbar_wired_to_it(self):
        root = fake_tkinter.Tk()
        bar = gm.StatusBar(root)
        scrollbars = [w for w in self._collect_descendants(root) if isinstance(w, gm.ttk.Scrollbar)]
        self.assertEqual(len(scrollbars), 1)

    @staticmethod
    def _collect_descendants(widget):
        acc = []
        for child in getattr(widget, "children", []):
            acc.append(child)
            acc.extend(TestStatusBar._collect_descendants(child))
        return acc


class TestGuiRunEndToEnd(unittest.TestCase):
    """Runs the REAL GuiCrackSegmentation.run() end to end: mode-choice
    dialog, image queue, a queued [S] save (through the exact same
    _pending_keys/process_keypress path a menu click uses), and confirms
    the finished frame reaches the embedded canvas (real cv2.cvtColor()/
    PIL conversion, no errors) while cv2.imshow() -- the OLD, non-embedded
    display path -- is never called at all."""

    def test_undo_then_redo_via_a_real_tk_keypress_event_restores_a_crack(self):
        # USER REPORT: "undo and redo on fractures no longer work" --
        # earlier tests (test_gui_shell_parity.py) called
        # process_keypress()/handle_keyboard() directly with a raw
        # integer code and still pass, so if there IS a regression it
        # has to be somewhere between a REAL keyboard event and that
        # call -- i.e. tk_key_event_to_code()/_on_keypress(). This fires
        # actual <KeyPress> events on the root (exactly what a real U/R
        # keystroke produces) instead of calling process_keypress(ord('u'))
        # directly, to catch a regression the more isolated tests can't.
        app = make_app(self)
        app.cfg.saved_cracks = [{'path': [(0, 0), (1, 1)], 'session_id': str(time.time())}]
        app.cfg.action_history = ['crack']
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = None

        # Real Tk reports keysym="u"/char="u" for a plain U keystroke,
        # and keysym="r"/char="r" for R -- exactly what _on_keypress()
        # receives and feeds through tk_key_event_to_code().
        app._tk_root.fire("<KeyPress>", keysym="u", char="u")
        self.assertEqual(app._pending_keys, [ord('u')],
                          "a real Tk KeyPress for 'u' must translate to ord('u') in the queue")

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(app._pending_keys.pop(0))
        self.assertEqual(app.cfg.saved_cracks, [], "Undo (real keypress) should have removed the crack")
        self.assertEqual(len(app.cfg.redo_history), 1)

        app._tk_root.fire("<KeyPress>", keysym="r", char="r")
        self.assertEqual(app._pending_keys, [ord('r')],
                          "a real Tk KeyPress for 'r' must translate to ord('r') in the queue")

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(app._pending_keys.pop(0))
        self.assertEqual(len(app.cfg.saved_cracks), 1,
                          "Redo (real keypress) must restore the exact crack Undo just removed")

    def test_full_session_via_embedded_canvas(self):
        import shutil
        import tempfile
        import cv2 as cv2_module
        from test_gui_shell_parity import _MockedHighGui

        tmpdir = tempfile.mkdtemp(prefix="crackseg_gui_e2e_")
        try:
            images_dir = os.path.join(tmpdir, "Images")
            os.makedirs(images_dir, exist_ok=True)
            img = np.full((220, 300, 3), 200, dtype=np.uint8)
            cv2_module.line(img, (20, 20), (280, 200), (30, 30, 30), 2)
            cv2_module.imwrite(os.path.join(images_dir, "img_00.png"), img)

            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            saved_stdout, saved_stderr = sys.stdout, sys.stderr
            self.addCleanup(lambda: setattr(sys, "stdout", saved_stdout))
            self.addCleanup(lambda: setattr(sys, "stderr", saved_stderr))
            app = gm.GuiCrackSegmentation(cfg)

            # Simulate: mode-choice dialog already answered "1", then one
            # [S] press queued (exactly what the toolbar/menu "Salva"
            # button does) before the render loop ever calls cv2.waitKey.
            app._pending_keys.append(ord('s'))

            main_window_imshow_calls = []
            real_imshow = cv2_module.imshow

            def tracking_imshow(window_name, frame):
                if window_name == "Crack Detector Workspace":
                    main_window_imshow_calls.append(window_name)
                # Let the (comprehensively mocked, no-op-ish) HighGUI stack
                # handle it -- _MockedHighGui already patches imshow itself;
                # this wrapper just observes calls, it doesn't replace that.

            waitkey_calls = []

            def tracking_waitkey(delay=0):
                waitkey_calls.append(delay)
                return -1

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch("cv2.getWindowImageRect", return_value=(0, 0, gm.CANVAS_W, gm.CANVAS_H)), \
                 _MockedHighGui() as hg:
                # Layer our tracking on top of _MockedHighGui's own imshow/waitKey mocks.
                with mock.patch("cv2.imshow", side_effect=tracking_imshow), \
                     mock.patch("cv2.waitKey", side_effect=tracking_waitkey):
                    app.run()

            self.assertEqual(main_window_imshow_calls, [],
                              "the main canvas must NEVER go through raw cv2.imshow -- "
                              "only through _display_frame() -> the embedded Tk canvas")
            self.assertEqual(waitkey_calls, [],
                              "CRASH REGRESSION: a real GUI session must NEVER call cv2.waitKey() "
                              "at all (see _wait_key's docstring for the confirmed macOS crash mechanism)")

            archive_dir = os.path.join(os.path.dirname(os.path.abspath(gm.__file__)), "already processed images")
            json_files = [f for f in os.listdir(archive_dir) if f.lower().endswith(".json")] \
                if os.path.isdir(archive_dir) else []
            self.assertTrue(json_files, "the queued [S] should have exported/archived exactly like a real keypress")
            shutil.rmtree(archive_dir, ignore_errors=True)
            csv_path = os.path.join(os.path.dirname(os.path.abspath(gm.__file__)), "segmentation_summary_report.csv")
            try:
                os.remove(csv_path)
            except OSError:
                pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
