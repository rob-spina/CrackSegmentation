"""test_v109_dataset_browser.py

Headless tests for the 1.0.9 [Dataset browser] (code 31): list of Training / Validation / Test set with the
building group number of each photo, and moves between the sets that keep each set's layout.

Run with:  python3 -m unittest test_v109_dataset_browser -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import dataset_browser as db
from config import Config
from smart_segmentation import CrackSegmentation
from test_v108_auto_split import _make


def _place(root, split, photo, with_annotations):
    """Moves a photo made by _make into a set; Training set takes its JSON and masks too."""
    folder = {"training": "Training set", "validation": "Validation set", "test": "Test set"}[split]
    base = os.path.splitext(photo)[0]
    os.makedirs(os.path.join(root, folder, "Images"), exist_ok=True)
    shutil.move(os.path.join(root, "Images", photo), os.path.join(root, folder, "Images", photo))
    if with_annotations:
        for sub, name in (("JSON files", base + ".json"), ("Binary files", base + "-crack_mask.png"),
                          ("Binary files", base + "-seg.png")):
            os.makedirs(os.path.join(root, folder, sub), exist_ok=True)
            shutil.move(os.path.join(root, sub, name), os.path.join(root, folder, sub, name))


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v109_browser_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        _make(self.tmpdir, {"BLDG001": 2, "BLDG002": 2, "BLDG003": 1},
              negatives={"DJI_BLDG003_00__BLDG003.JPG"})
        self.t1, self.t2 = "DJI_BLDG001_00__BLDG001.JPG", "IMG_BLDG001_01__BLDG001.JPG"
        self.v1, self.v2 = "DJI_BLDG002_00__BLDG002.JPG", "IMG_BLDG002_01__BLDG002.JPG"
        self.e1 = "DJI_BLDG003_00__BLDG003.JPG"
        for p in (self.t1, self.t2):
            _place(self.tmpdir, "training", p, True)
        for p in (self.v1, self.v2):
            _place(self.tmpdir, "validation", p, False)
        _place(self.tmpdir, "test", self.e1, False)

    def _app(self, cls=CrackSegmentation):
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        app = cls(cfg)
        app.cfg.queue_idle = True
        return app

    def p(self, *parts):
        return os.path.join(self.tmpdir, *parts)


class TestInventoryAndMoves(_Base):

    def setUp(self):
        super().setUp()
        self.app = self._app()

    def test_group_number(self):
        self.assertEqual(db.group_number("DJI_0349_D__BLDG028.JPG"), "028")
        self.assertEqual(db.group_number("DJI_0349.JPG"), "-")

    def test_inventory_lists_every_set_with_the_building(self):
        rows = {r["photo"]: r for r in self.app.dataset_inventory()}
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[self.t1]["split"], "training")
        self.assertEqual(rows[self.t1]["number"], "001")
        self.assertEqual(rows[self.v2]["split"], "validation")
        self.assertFalse(rows[self.v2]["drone"])
        self.assertTrue(rows[self.e1]["negative"], "its JSON (in the working folder) has no crack")
        self.assertFalse(rows[self.t1]["negative"], "its JSON is inside Training set")

    def test_training_to_test_sends_annotations_back_to_working_folders(self):
        moved, problems = self.app.move_photos_between_sets([("training", self.t1)], "test", whole_building=False)
        self.assertEqual((moved, problems), (1, []))
        base = os.path.splitext(self.t1)[0]
        self.assertTrue(os.path.isfile(self.p("Test set", "Images", self.t1)))
        self.assertTrue(os.path.isfile(self.p("JSON files", base + ".json")))
        self.assertTrue(os.path.isfile(self.p("Binary files", base + "-crack_mask.png")))
        self.assertFalse(os.path.exists(self.p("Training set", "JSON files", base + ".json")))
        self.assertFalse(os.path.exists(self.p("Test set", "JSON files")), "Test set keeps the photo only")

    def test_test_to_training_brings_annotations_in(self):
        moved, _ = self.app.move_photos_between_sets([("test", self.e1)], "training")
        self.assertEqual(moved, 1)
        base = os.path.splitext(self.e1)[0]
        for rel in (("Images", self.e1), ("JSON files", base + ".json"), ("Binary files", base + "-seg.png")):
            self.assertTrue(os.path.isfile(self.p("Training set", *rel)), rel)
            self.assertFalse(os.path.exists(self.p(*rel)), rel)

    def test_validation_to_test_moves_the_photo_only(self):
        base = os.path.splitext(self.v1)[0]
        moved, _ = self.app.move_photos_between_sets([("validation", self.v1)], "test", whole_building=False)
        self.assertEqual(moved, 1)
        self.assertTrue(os.path.isfile(self.p("Test set", "Images", self.v1)))
        self.assertTrue(os.path.isfile(self.p("JSON files", base + ".json")), "annotations stay where they are")
        self.assertIn(self.v2, os.listdir(self.p("Validation set", "Images")))

    def test_whole_building_moves_every_photo_of_the_building(self):
        plan = self.app.plan_set_move([("validation", self.v1)], "test")
        self.assertEqual(plan, [("validation", self.v1), ("validation", self.v2)])
        moved, _ = self.app.move_photos_between_sets([("validation", self.v1)], "test")
        self.assertEqual(moved, 2)
        self.assertFalse(os.path.exists(self.p("Validation set", "Images", self.v2)))

    def test_photos_already_in_the_target_are_left_out(self):
        self.assertEqual(self.app.plan_set_move([("training", self.t1)], "training"), [])

    def test_clash_is_never_overwritten(self):
        base = os.path.splitext(self.t1)[0]
        with open(self.p("JSON files", base + ".json"), "w") as f:
            f.write("another file")
        moved, problems = self.app.move_photos_between_sets([("training", self.t1)], "test", whole_building=False)
        self.assertEqual(moved, 0)
        self.assertIn(base + ".json", problems[0])
        self.assertTrue(os.path.isfile(self.p("Training set", "Images", self.t1)), "the photo stays put")
        with open(self.p("JSON files", base + ".json")) as f:
            self.assertEqual(f.read(), "another file")

    def test_key_from_mode2_and_idle(self):
        with mock.patch.object(self.app, "_toggle_dataset_browser") as toggle:
            self.app.cfg.queue_idle = False
            self.assertFalse(self.app.process_keypress(db.DATASET_BROWSER_CODE), "the image stays on screen")
            self.app.cfg.queue_idle = True
            self.assertFalse(self.app._handle_idle_key(db.DATASET_BROWSER_CODE))
        self.assertEqual(toggle.call_count, 2)

    def test_terminal_fallback_prints_the_sets(self):
        with mock.patch("builtins.print") as printed:
            self.app._toggle_dataset_browser()
        out = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("Training set (2 photo(s))", out)
        self.assertIn("BLDG  003", out)


class TestPanel(_Base):

    def setUp(self):
        super().setUp()
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        import crack_segmentation_gui as gm
        self.gm = gm
        saved = sys.stdout, sys.stderr
        self.addCleanup(lambda: setattr(sys, "stdout", saved[0]) or setattr(sys, "stderr", saved[1]))
        self.app = self._app(gm.GuiCrackSegmentation)
        self.panel = self.app._dataset_panel

    def _items(self):
        tree = self.panel.tree
        return {tree.item(i)["text"]: i for s in tree.get_children() for i in (s,) + tree.get_children(s)}

    def test_toggle_covers_the_image_and_lists_the_sets(self):
        self.app._toggle_dataset_browser()
        self.assertTrue(self.app._dataset_panel_visible)
        items = self._items()
        self.assertIn("Training set  (2 photo(s))", items)
        self.assertEqual(self.panel.tree.item(items[self.v1])["values"], ("002", "drone"))
        self.assertEqual(self.panel.tree.item(items[self.e1])["values"], ("003", "drone, no cracks"))
        self.app._toggle_dataset_browser()
        self.assertFalse(self.app._dataset_panel_visible)

    def test_manual_and_browser_replace_each_other(self):
        self.app._toggle_dataset_browser()
        with mock.patch.object(self.app._pdf_panel, "show"):
            self.app._toggle_help_menu()
        self.assertFalse(self.app._dataset_panel_visible)
        self.assertTrue(self.app._pdf_panel_visible)
        self.app._toggle_dataset_browser()
        self.assertFalse(self.app._pdf_panel_visible)
        self.assertTrue(self.app._dataset_panel_visible)

    def test_selecting_a_set_row_selects_its_photos(self):
        self.panel.refresh()
        items = self._items()
        self.panel.tree.selection_set(items["Validation set  (2 photo(s))"])
        self.assertEqual(self.panel.selected_photos(), [("validation", self.v1), ("validation", self.v2)])

    def test_move_from_the_panel(self):
        self.panel.refresh()
        self.panel.tree.selection_set(self._items()[self.t1])
        with mock.patch.object(self.gm.messagebox, "askyesno", return_value=True) as asked:
            self.assertTrue(self.panel.move_selected("validation"))
        message = asked.call_args[0][1]
        self.assertIn("Move 2 photo(s) to Validation set", message)
        self.assertIn("same building", message)
        self.assertIn("CV folds again", message)
        self.assertIn("2 photo(s) moved", self.panel.status.kw["text"])
        self.assertIn("Validation set  (4 photo(s))", self._items())

    def test_cancel_and_empty_selection_move_nothing(self):
        self.panel.refresh()
        self.assertFalse(self.panel.move_selected("test"))
        self.assertIn("Select one or more", self.panel.status.kw["text"])
        self.panel.tree.selection_set(self._items()[self.e1])
        with mock.patch.object(self.gm.messagebox, "askyesno", return_value=False):
            self.assertFalse(self.panel.move_selected("training"))
        self.assertTrue(os.path.isfile(self.p("Test set", "Images", self.e1)))

    def test_sidebar_and_menu(self):
        self.assertIn(db.DATASET_BROWSER_CODE, self.app._sidebar.buttons_by_code)
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(db.DATASET_BROWSER_CODE, menu_codes)


if __name__ == "__main__":
    unittest.main()
