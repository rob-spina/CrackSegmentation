"""test_v108_auto_split.py

Headless tests for the 1.0.8 [Auto split] button (code 28): every segmented photo of 'Images' is split into
Training / Validation / Test set in one go, by building group, following the percentages asked.

Run with:  python3 -m unittest test_v108_auto_split -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import auto_split as aus
from config import Config
from smart_segmentation import CrackSegmentation


def _make(root, groups, no_json=(), negatives=()):
    """groups: {tag: n photos}; every photo gets a JSON (with a crack unless listed in negatives) and masks."""
    for sub in ("Images", "JSON files", "Binary files"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    photos = []
    for tag, n in groups.items():
        for i in range(n):
            base = f"DJI_{tag}_{i:02d}__{tag}" if i % 2 == 0 else f"IMG_{tag}_{i:02d}__{tag}"
            photos.append(base + ".JPG")
            open(os.path.join(root, "Images", base + ".JPG"), "w").close()
            if base + ".JPG" in no_json:
                continue
            shapes = "[]" if base + ".JPG" in negatives else \
                '[{"label": "crack_x", "shape_type": "linestrip", "points": [[0, 0], [5, 5]]}]'
            with open(os.path.join(root, "JSON files", base + ".json"), "w") as f:
                f.write('{"shapes": %s}' % shapes)
            for suffix in ("-crack_mask.png", "-seg.png"):
                open(os.path.join(root, "Binary files", base + suffix), "w").close()
    return photos


class TestHelpers(unittest.TestCase):

    def test_parse_percent(self):
        self.assertEqual(aus.parse_split_percent("70/15/15"), (0.7, 0.15, 0.15))
        self.assertEqual(aus.parse_split_percent("80 0 20"), (0.8, 0.0, 0.2))
        self.assertEqual(aus.parse_split_percent("7,1.5,1.5"), (0.7, 0.15, 0.15))
        for bad in ("70/30", "a/b/c", "0/50/50", "-10/60/50"):
            with self.assertRaises(ValueError, msg=bad):
                aus.parse_split_percent(bad)

    def test_groups_follow_the_percentages(self):
        sizes = [7, 7, 7, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3] + [2] * 10 + [1] * 12      # the real dataset: 93 photos
        groups = {f"BLDG{i:03d}": n for i, n in enumerate(sizes)}
        plan = aus.assign_groups(groups, {"training": 0.7, "validation": 0.15, "test": 0.15})
        n = {s: sum(groups[g] for g, t in plan.items() if t == s) for s in aus.SPLIT_ORDER}
        self.assertEqual(sum(n.values()), 93)
        self.assertLessEqual(abs(n["training"] - 65), 3, n)
        self.assertLessEqual(abs(n["validation"] - 14), 3, n)
        self.assertLessEqual(abs(n["test"] - 14), 3, n)

    def test_every_set_gets_a_similar_mix_of_drone_and_crack_free_photos(self):
        # 20 drone buildings (2 photos), 20 phone buildings (2 photos), 10 crack-free buildings (1 photo, drone).
        groups = {f"D{i:02d}": (2, 2, 0) for i in range(20)}
        groups.update({f"P{i:02d}": (2, 0, 0) for i in range(20)})
        groups.update({f"N{i:02d}": (1, 1, 1) for i in range(10)})
        plan = aus.assign_groups(groups, {"training": 0.7, "validation": 0.15, "test": 0.15})
        for split in aus.SPLIT_ORDER:
            mine = [groups[g] for g, t in plan.items() if t == split]
            n, drone, neg = (sum(v[i] for v in mine) for i in range(3))
            self.assertGreater(n, 0, split)
            self.assertLessEqual(abs(drone / n - 50 / 90), 0.15, (split, n, drone))
            self.assertLessEqual(abs(neg / n - 10 / 90), 0.10, (split, n, neg))

    def test_zero_share_gets_nothing_and_fixed_groups_stay(self):
        groups = {"A": 5, "B": 3, "C": 2, "D": 2}
        plan = aus.assign_groups(groups, {"training": 0.8, "validation": 0.0, "test": 0.2},
                                 fixed={"C": "validation"}, already={"validation": 1})
        self.assertEqual(plan["C"], "validation", "a group already in a set stays there")
        self.assertNotIn("validation", [plan[g] for g in "ABD"])

    def test_deterministic(self):
        groups = {f"G{i}": (i % 4) + 1 for i in range(20)}
        f = {"training": 0.7, "validation": 0.15, "test": 0.15}
        self.assertEqual(aus.assign_groups(groups, f), aus.assign_groups(dict(reversed(list(groups.items()))), f))


class TestButton(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_auto_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.photos = _make(self.tmpdir, {"BLDG001": 6, "BLDG002": 4, "BLDG003": 3, "BLDG004": 3, "BLDG005": 2,
                                          "BLDG006": 2},
                            no_json={"IMG_BLDG006_01__BLDG006.JPG"}, negatives={"DJI_BLDG005_00__BLDG005.JPG"})
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        self.app = CrackSegmentation(cfg)
        self.app.cfg.modalita_scelta = "1"
        self.app.cfg.queue_idle = True

    def _press(self, percent="70/15/15", confirm=True):
        with mock.patch.object(self.app, "_ask_auto_split_percent", return_value=percent), \
                mock.patch.object(self.app, "_confirm_auto_split", return_value=confirm) as asked, \
                mock.patch.object(self.app, "_notify_auto_split") as notified:
            moved = self.app.run_auto_split()
        return moved, asked, notified

    def _ls(self, *parts):
        path = os.path.join(self.tmpdir, *parts)
        return sorted(os.listdir(path)) if os.path.isdir(path) else []

    def _group_sets(self):
        where = {}
        for name in ("Training set", "Validation set", "Test set"):
            for photo in self._ls(name, "Images"):
                where.setdefault(photo.split("__")[1].split(".")[0], set()).add(name)
        return where

    def test_moves_every_segmented_photo_by_building(self):
        moved, asked, _ = self._press()
        self.assertTrue(moved)
        self.assertEqual(self._ls("Images"), ["IMG_BLDG006_01__BLDG006.JPG"], "only the photo without JSON stays")
        self.assertTrue(all(len(s) == 1 for s in self._group_sets().values()), "a building split across sets")
        total = sum(len(self._ls(n, "Images")) for n in ("Training set", "Validation set", "Test set"))
        self.assertEqual(total, 19)
        message = asked.call_args[0][0]
        self.assertIn("by building (70/15/15)", message)
        self.assertIn("1 photo(s) without a JSON are not moved", message)
        self.assertIn("1 without cracks", message)

    def test_training_takes_json_and_masks_others_the_photo_only(self):
        self._press()
        train = self._ls("Training set", "Images")
        self.assertTrue(train)
        base = os.path.splitext(train[0])[0]
        self.assertIn(base + ".json", self._ls("Training set", "JSON files"))
        self.assertIn(base + "-crack_mask.png", self._ls("Training set", "Binary files"))
        for name in ("Validation set", "Test set"):
            for photo in self._ls(name, "Images"):
                b = os.path.splitext(photo)[0]
                self.assertIn(b + ".json", self._ls("JSON files"), "JSON stays in the working folder")
                self.assertIn(b + "-crack_mask.png", self._ls("Binary files"))
            self.assertEqual(self._ls(name, "JSON files"), [])

    def test_80_0_20_leaves_validation_empty(self):
        self._press("80/0/20")
        self.assertEqual(self._ls("Validation set", "Images"), [])
        self.assertTrue(self._ls("Test set", "Images"))

    def test_group_already_in_a_set_stays_there(self):
        os.makedirs(os.path.join(self.tmpdir, "Test set", "Images"))
        os.replace(os.path.join(self.tmpdir, "Images", "DJI_BLDG001_00__BLDG001.JPG"),
                   os.path.join(self.tmpdir, "Test set", "Images", "DJI_BLDG001_00__BLDG001.JPG"))
        _, asked, _ = self._press()
        self.assertEqual(self._group_sets()["BLDG001"], {"Test set"})
        self.assertIn("already in a set stay in that set", asked.call_args[0][0])

    def test_cancel_and_bad_percentages_move_nothing(self):
        moved, _, _ = self._press(confirm=False)
        self.assertFalse(moved)
        moved, asked, notified = self._press("70/30")
        self.assertFalse(moved)
        asked.assert_not_called()
        self.assertIn("80/0/20", notified.call_args[0][0])
        self.assertEqual(len(self._ls("Images")), 20)

    def test_from_mode2_the_queue_is_reloaded(self):
        self.app.cfg.queue_idle = False
        self.app.cfg.modalita_scelta = "2"
        self.app.cfg.CURRENT_IMAGE_PATH = os.path.join(self.tmpdir, "Images", self.photos[0])
        with mock.patch.object(self.app, "_ask_auto_split_percent", return_value="70/15/15"), \
                mock.patch.object(self.app, "_confirm_auto_split", return_value=True), \
                mock.patch.object(self.app, "_notify_auto_split"):
            self.assertTrue(self.app.process_keypress(aus.AUTO_SPLIT_CODE))
        self.assertTrue(self.app.cfg.reload_queue_requested)

    def test_works_on_the_idle_screen(self):
        with mock.patch.object(self.app, "run_auto_split") as run:
            self.app._handle_idle_key(aus.AUTO_SPLIT_CODE)
        run.assert_called_once()


class TestSidebar(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_button_in_the_dataset_section_and_menu(self):
        left = [b.text for b in self.app._sidebar.left.buttons]
        self.assertTrue(any("Auto split" in t for t in left))
        btn = self.app._sidebar.buttons_by_code[aus.AUTO_SPLIT_CODE]
        self.app._pending_keys.clear()
        btn.command()
        self.assertEqual(self.app._pending_keys, [aus.AUTO_SPLIT_CODE])
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(aus.AUTO_SPLIT_CODE, menu_codes)

    def test_gui_asks_with_the_default(self):
        self.gm.simpledialog.next_string = None
        with mock.patch.object(self.gm.simpledialog, "askstring", return_value="70/15/15") as ask:
            self.assertEqual(self.app._ask_auto_split_percent("70/15/15"), "70/15/15")
        self.assertEqual(ask.call_args.kwargs.get("initialvalue"), "70/15/15")


if __name__ == "__main__":
    unittest.main()
