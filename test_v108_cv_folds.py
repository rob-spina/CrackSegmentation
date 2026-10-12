"""test_v108_cv_folds.py

Headless tests for the 1.0.8 [CV folds] button (code 26): the photos of 'Training set' are split into K
folds by building group (each group in one fold, folds kept even) and written to 'Training set/folds.csv'.

Run with:  python3 -m unittest test_v108_cv_folds -v
"""
import csv
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cross_validation as cv
from test_gui_shell_parity import make_app


def _make_training_set(root, groups):
    """groups: {group tag or None: number of photos}; every photo gets a JSON and a crack mask."""
    for sub in ("Images", "JSON files", "Binary files"):
        os.makedirs(os.path.join(root, "Training set", sub), exist_ok=True)
    photos = []
    for g, n in groups.items():
        for i in range(n):
            base = f"IMG_{g or 'free'}_{i:02d}" + (f"__{g}" if g else "")
            photos.append(base + ".JPG")
            open(os.path.join(root, "Training set", "Images", base + ".JPG"), "w").close()
            open(os.path.join(root, "Training set", "JSON files", base + ".json"), "w").close()
            open(os.path.join(root, "Training set", "Binary files", base + "-crack_mask.png"), "w").close()
    return photos


class TestAssignment(unittest.TestCase):

    def test_each_group_in_one_fold_and_folds_even(self):
        photos = [f"p{i}__BLDG{g:03d}.jpg" for g, n in enumerate([7, 7, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1]) for i in range(n)]
        folds = cv.assign_group_folds(photos, 5)
        by_group = {}
        for p, f in folds.items():
            by_group.setdefault(cv.photo_group(p), set()).add(f)
        self.assertTrue(all(len(s) == 1 for s in by_group.values()), "a building split across folds")
        sizes = [list(folds.values()).count(f) for f in range(1, 6)]
        self.assertEqual(sum(sizes), len(photos))
        self.assertLessEqual(max(sizes) - min(sizes), 2, sizes)

    def test_reproducible(self):
        photos = [f"p{i}__BLDG{i % 7:03d}.jpg" for i in range(30)]
        self.assertEqual(cv.assign_group_folds(photos, 4), cv.assign_group_folds(list(reversed(photos)), 4))

    def test_photo_without_group_is_its_own_group(self):
        self.assertEqual(cv.photo_group("IMG_1.jpg"), "single:IMG_1")
        self.assertEqual(cv.photo_group("x__bldg007.JPG"), "BLDG007")


class TestButton(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_cv_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.photos = _make_training_set(self.tmpdir, {"BLDG001": 4, "BLDG002": 3, "BLDG003": 3, "BLDG004": 2, None: 2})
        os.remove(os.path.join(self.tmpdir, "Training set", "Binary files", "IMG_free_01-crack_mask.png"))
        open(os.path.join(self.tmpdir, "Training set", "Binary files", "IMG_BLDG001_00__BLDG001-ignore_mask.png"),
             "w").close()
        self.app = make_app()
        self.app.cfg.SCRIPT_DIR = self.tmpdir
        self.csv_path = os.path.join(self.tmpdir, "Training set", "folds.csv")

    def _press(self, k=3, confirm=True):
        with mock.patch.object(self.app, "_ask_cv_fold_count", return_value=k), \
                mock.patch.object(self.app, "_confirm_cv_folds", return_value=confirm) as asked, \
                mock.patch.object(self.app, "_notify_cv_folds") as notified:
            self.app.process_keypress(cv.CV_FOLDS_CODE)
        return asked, notified

    def _rows(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_writes_folds_csv(self):
        asked, _ = self._press(k=3)
        rows = self._rows()
        self.assertEqual(sorted(r["photo"] for r in rows), sorted(self.photos))
        self.assertEqual({r["fold"] for r in rows}, {"1", "2", "3"})
        for g in ("BLDG001", "BLDG002", "BLDG003", "BLDG004"):
            self.assertEqual(len({r["fold"] for r in rows if r["group"] == g}), 1, g)
        row = next(r for r in rows if r["photo"] == "IMG_BLDG001_00__BLDG001.JPG")
        self.assertEqual(row["image"], "Images/IMG_BLDG001_00__BLDG001.JPG")
        self.assertEqual(row["json"], "JSON files/IMG_BLDG001_00__BLDG001.json")
        self.assertEqual(row["crack_mask"], "Binary files/IMG_BLDG001_00__BLDG001-crack_mask.png")
        self.assertEqual(row["ignore_mask"], "Binary files/IMG_BLDG001_00__BLDG001-ignore_mask.png")
        self.assertEqual(row["uncertain_mask"], "")
        message = asked.call_args[0][0]
        self.assertIn("Fold 1:", message)
        self.assertIn("1 photo(s) without a crack mask", message)

    def test_cancel_writes_nothing(self):
        self._press(confirm=False)
        self.assertFalse(os.path.exists(self.csv_path))

    def test_more_folds_than_buildings_is_refused(self):
        asked, notified = self._press(k=7)   # 4 groups + 2 single photos = 6 groups
        asked.assert_not_called()
        self.assertIn("at most 6 folds", notified.call_args[0][0])
        self.assertFalse(os.path.exists(self.csv_path))

    def test_empty_training_set_is_refused(self):
        shutil.rmtree(os.path.join(self.tmpdir, "Training set"))
        asked, notified = self._press()
        asked.assert_not_called()
        self.assertIn("holds no photo", notified.call_args[0][0])

    def test_rewriting_asks_to_replace(self):
        self._press()
        asked, _ = self._press()
        self.assertIn("Replace the existing folds.csv", asked.call_args[0][0])

    def test_works_while_idle(self):
        self.app.cfg.queue_idle = True
        with mock.patch.object(self.app, "make_cv_folds") as make:
            self.app._handle_idle_key(cv.CV_FOLDS_CODE)
        make.assert_called_once()


class TestSidebar(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
        import fake_tkinter  # noqa: F401
        from test_gui_panel_wiring import make_app as make_gui_app
        import crack_segmentation_gui as gm
        self.gm = gm
        self.app = make_gui_app(self)

    def test_button_and_menu_send_code_26(self):
        btn = next(b for b in self.app._sidebar.buttons if "CV folds" in b.text)
        self.app._pending_keys.clear()
        btn.command()
        self.assertEqual(self.app._pending_keys, [cv.CV_FOLDS_CODE])
        menu_codes = [code for _, commands in self.gm._ALL_MENU_GROUPS for _, code in commands]
        self.assertIn(cv.CV_FOLDS_CODE, menu_codes)

    def test_gui_asks_the_fold_count_with_the_default(self):
        self.assertEqual(self.app._ask_cv_fold_count(5), 5)


if __name__ == "__main__":
    unittest.main()
