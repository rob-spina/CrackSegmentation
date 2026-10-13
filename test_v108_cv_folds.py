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

    def test_copies_each_fold_into_its_folder(self):
        self._press(k=3)
        folds_dir = os.path.join(self.tmpdir, "Training set", "folds")
        self.assertEqual(sorted(os.listdir(folds_dir)), ["fold_1", "fold_2", "fold_3"])
        rows = self._rows()
        for k in ("1", "2", "3"):
            fold = os.path.join(folds_dir, f"fold_{k}")
            expected = sorted(r["photo"] for r in rows if r["fold"] == k)
            self.assertEqual(sorted(os.listdir(os.path.join(fold, "Images"))), expected)
            jsons = sorted(os.listdir(os.path.join(fold, "JSON files")))
            self.assertEqual(jsons, sorted(os.path.splitext(p)[0] + ".json" for p in expected))
        row = next(r for r in rows if r["photo"] == "IMG_BLDG001_00__BLDG001.JPG")
        binaries = os.listdir(os.path.join(folds_dir, f"fold_{row['fold']}", "Binary files"))
        self.assertIn("IMG_BLDG001_00__BLDG001-crack_mask.png", binaries)
        self.assertIn("IMG_BLDG001_00__BLDG001-ignore_mask.png", binaries)
        self.assertEqual(len(os.listdir(os.path.join(self.tmpdir, "Training set", "Images"))), len(self.photos),
                         "the Training set keeps every photo (copies, not moves)")

    def test_pressed_again_replaces_the_fold_folders(self):
        self._press(k=3)
        self._press(k=2)
        folds_dir = os.path.join(self.tmpdir, "Training set", "folds")
        self.assertEqual(sorted(os.listdir(folds_dir)), ["fold_1", "fold_2"])
        n = sum(len(os.listdir(os.path.join(folds_dir, f, "Images"))) for f in os.listdir(folds_dir))
        self.assertEqual(n, len(self.photos), "no stale copy left from the 3-fold split")

    def test_photo_saved_without_cracks_is_a_negative(self):
        ts = os.path.join(self.tmpdir, "Training set")
        open(os.path.join(ts, "Images", "clean__BLDG050.JPG"), "w").close()
        with open(os.path.join(ts, "JSON files", "clean__BLDG050.json"), "w") as f:
            f.write('{"shapes": []}')
        with open(os.path.join(ts, "JSON files", "IMG_free_01.json"), "w") as f:   # has a crack, mask missing
            f.write('{"shapes": [{"label": "crack_x", "shape_type": "linestrip", "points": [[0, 0], [5, 5]]}]}')
        asked, _ = self._press(k=3)
        rows = {r["photo"]: r for r in self._rows()}
        self.assertEqual(rows["clean__BLDG050.JPG"]["negative"], "1")
        self.assertEqual(rows["clean__BLDG050.JPG"]["crack_mask"], "")
        self.assertEqual(rows["IMG_free_01.JPG"]["negative"], "0", "a crack without its mask is not a negative")
        self.assertEqual(rows["IMG_BLDG001_00__BLDG001.JPG"]["negative"], "0")
        message = asked.call_args[0][0]
        self.assertIn("1 without cracks", message)
        self.assertIn("1 photo(s) without a crack mask", message, "only the real problem is reported")
        self.assertIn("IMG_free_01.JPG", message)
        fold = rows["clean__BLDG050.JPG"]["fold"]
        self.assertIn("clean__BLDG050.JPG",
                      os.listdir(os.path.join(ts, "folds", f"fold_{fold}", "Images")), "negatives are copied too")

    def test_cancel_writes_nothing(self):
        self._press(confirm=False)
        self.assertFalse(os.path.exists(self.csv_path))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "Training set", "folds")))

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
        self.assertIn("are replaced", asked.call_args[0][0])

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


class TestEmptyCrackMask(unittest.TestCase):
    """v1.0.8: a photo saved with no crack gets an all-black crack mask (a negative example)."""

    def setUp(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_v108_empty_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        os.makedirs(os.path.join(self.tmpdir, "Images"))
        self.photo = os.path.join(self.tmpdir, "Images", "clean_wall.png")
        cv2.imwrite(self.photo, np.full((300, 400, 3), 180, np.uint8))
        from config import Config
        from smart_segmentation import CrackSegmentation
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        self.app = CrackSegmentation(cfg)
        self.app.cfg.modalita_scelta = "1"
        self.app._set_queue_folders()
        self.app.cfg.image_queue = [self.photo]

    def _save(self):
        with mock.patch.object(self.app, "_acknowledge_save_warning"), \
                mock.patch.object(self.app, "run_post_save_group_crack_check"), \
                mock.patch.object(self.app, "_play_save_beep"), \
                mock.patch("smart_segmentation.resolve_script_dir", return_value=self.tmpdir):
            self.app.initialize_image_session(self.photo)
            self.app.process_keypress(ord('s'))
        binary = os.path.join(self.tmpdir, "Binary files")
        return [f for f in os.listdir(binary) if f.endswith("-crack_mask.png")], binary

    def test_all_black_mask_of_the_photo_size(self):
        masks, binary = self._save()
        self.assertEqual(len(masks), 1)
        mask = self.cv2.imread(os.path.join(binary, masks[0]), self.cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (300, 400))
        self.assertEqual(int(mask.max()), 0)

    def test_switch_off_restores_the_old_behaviour(self):
        self.app.cfg.CRACK_MASK_WRITE_EMPTY = False
        masks, _ = self._save()
        self.assertEqual(masks, [])

    def test_photo_with_black_mask_is_a_negative_in_folds(self):
        root = os.path.join(self.tmpdir, "TS")
        for sub in ("JSON files", "Binary files"):
            os.makedirs(os.path.join(root, sub))
        with open(os.path.join(root, "JSON files", "w.json"), "w") as f:
            f.write('{"shapes": []}')
        self.cv2.imwrite(os.path.join(root, "Binary files", "w-crack_mask.png"), self.np.zeros((10, 10), self.np.uint8))
        self.assertTrue(cv.is_negative_photo(root, "JSON files/w.json", "Binary files/w-crack_mask.png"))
        with open(os.path.join(root, "JSON files", "w.json"), "w") as f:
            f.write('{"shapes": [{"label": "crack_w", "shape_type": "linestrip", "points": [[0, 0], [5, 5]]}]}')
        self.assertFalse(cv.is_negative_photo(root, "JSON files/w.json", "Binary files/w-crack_mask.png"))
