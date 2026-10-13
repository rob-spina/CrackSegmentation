"""test_v109_go_to_image.py

Headless tests for the 1.0.9 [Go to image] command (code 29): first / last / chosen image, chosen from a list
of 10 names at a time, in Mode 1 and Mode 2, without saving.

Run with:  python3 -m unittest test_v109_go_to_image -v
"""
import os
import sys
import unittest
from unittest import mock

import go_to_image as gti
from test_gui_shell_parity import make_app

NAMES = [f"IMG_{i:03d}.JPG" for i in range(1, 26)]   # 25 images: pages 1-10, 11-20, 21-25


class TestPagedChooser(unittest.TestCase):

    def test_pages_of_ten(self):
        c = gti.PagedChooser(NAMES)
        self.assertEqual(c.page_count, 3)
        self.assertEqual(c.labels()[0], "1. IMG_001.JPG")
        self.assertEqual(len(c.labels()), 10)
        c.next_page()
        c.next_page()
        self.assertEqual(c.labels(), [f"{i}. IMG_{i:03d}.JPG" for i in range(21, 26)])
        c.next_page()
        self.assertEqual(c.page, 2, "no page after the last")

    def test_starts_on_the_page_of_the_current_image(self):
        self.assertEqual(gti.PagedChooser(NAMES, 14).page_range(), (10, 19))

    def test_choosing_the_tenth_shows_the_next_ten(self):
        c = gti.PagedChooser(NAMES)
        self.assertEqual(c.choose(9), 9)               # 10th image
        self.assertEqual(c.page_range(), (10, 19))

    def test_choosing_the_first_of_a_later_page_shows_the_previous_ten(self):
        c = gti.PagedChooser(NAMES, 12)
        self.assertEqual(c.choose(0), 10)              # 11th image
        self.assertEqual(c.page_range(), (0, 9))

    def test_middle_choice_keeps_the_page_and_last_short_page_does_not_move(self):
        c = gti.PagedChooser(NAMES)
        self.assertEqual(c.choose(4), 4)
        self.assertEqual(c.page, 0)
        c = gti.PagedChooser(NAMES, 24)
        self.assertEqual(c.choose(4), 24)              # last image, short page: nothing after it
        self.assertEqual(c.page, 2)

    def test_index_from_label(self):
        self.assertEqual(gti.PagedChooser.index_from_label("12. DJI_0349.JPG"), 11)
        self.assertIsNone(gti.PagedChooser.index_from_label("DJI_0349.JPG"))


class TestNavigation(unittest.TestCase):

    def setUp(self):
        self.app = make_app()
        self.app.cfg.image_queue = [f"/x/Images/{n}" for n in NAMES]
        self.app.cfg.CURRENT_IMAGE_PATH = self.app.cfg.image_queue[3]
        self.app.cfg.queue_idle = False

    def _go(self, target):
        with mock.patch.object(self.app, "_choose_image_index", return_value=target) as choose:
            moved = self.app.process_keypress(gti.GO_TO_IMAGE_CODE)
        return moved, choose

    def test_jump_to_a_chosen_image(self):
        moved, choose = self._go(17)
        self.assertTrue(moved, "the image loop exits")
        choose.assert_called_once_with(NAMES, 3)
        self.assertEqual(self.app._advance_queue_position(3), 17)
        self.assertIsNone(self.app.cfg.navigate_target, "used once")
        self.assertEqual(self.app._advance_queue_position(17), 18, "then Next works as usual")

    def test_first_and_last(self):
        self._go(0)
        self.assertEqual(self.app._advance_queue_position(3), 0)
        self._go(24)
        self.assertEqual(self.app._advance_queue_position(3), 24)

    def test_cancel_or_same_image_stays(self):
        self.assertFalse(self._go(None)[0])
        self.assertFalse(self._go(3)[0])
        self.assertIsNone(self.app.cfg.navigate_target)

    def test_works_in_mode1_and_mode2(self):
        for mode in ("1", "2"):
            self.app.cfg.modalita_scelta = mode
            self.assertTrue(self._go(10)[0], mode)
            self.app._advance_queue_position(3)

    def test_mode1_after_saves_counts_done_images(self):
        # Mode 1: two images already saved this pass, so the queue holds the rest.
        self.app.cfg.modalita_scelta = "1"
        self.app.cfg.queue_items_done = 2
        self._go(5)
        pos = self.app._advance_queue_position(3)
        self.assertEqual(self.app._queue_display_position(pos), (8, 27))

    def test_nothing_while_idle(self):
        self.app.cfg.queue_idle = True
        with mock.patch.object(self.app, "_choose_image_index") as choose:
            self.assertFalse(self.app.process_keypress(gti.GO_TO_IMAGE_CODE))
        choose.assert_not_called()

    def test_terminal_fallback(self):
        for answer, expected in (("f", 0), ("l", 24), ("12", 11), ("99", None), ("", None)):
            with mock.patch("builtins.input", return_value=answer), mock.patch("builtins.print"):
                self.assertEqual(self.app._choose_image_index(NAMES, 3), expected, answer)


class TestDialog(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def _dialog(self, current=3):
        return self.gm._GoToImageDialog(self.app._tk_root, NAMES, current)

    def test_first_last_and_go(self):
        d = self._dialog()
        d._finish(0)
        self.assertEqual(d.result, 0)
        d = self._dialog()
        d.combo.pick(6)
        d._go()
        self.assertEqual(d.result, 6)

    def test_picking_the_tenth_pages_forward(self):
        d = self._dialog()
        d.combo.pick(9)
        self.assertEqual(d.selected, 9)
        self.assertEqual(d.combo.kw["values"][0], "11. IMG_011.JPG")
        self.assertEqual(d.page_label.kw["text"], "Images 11-20 of 25")
        d.combo.pick(0)                                   # 11th: back to 1-10
        self.assertEqual(d.selected, 10)
        self.assertEqual(d.combo.kw["values"][0], "1. IMG_001.JPG")

    def test_page_buttons(self):
        d = self._dialog()
        d._next_page()
        d._next_page()
        self.assertEqual(d.combo.kw["values"][-1], "25. IMG_025.JPG")
        d._previous_page()
        self.assertEqual(d.page_label.kw["text"], "Images 11-20 of 25")

    def test_cancel(self):
        d = self._dialog()
        d._cancel()
        self.assertIsNone(d.result)

    def test_toolbar_and_menu(self):
        toolbar = [code for _, code in self.app._toolbar_shortcuts()]
        self.assertIn(gti.GO_TO_IMAGE_CODE, toolbar)
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(gti.GO_TO_IMAGE_CODE, menu_codes)


if __name__ == "__main__":
    unittest.main()
