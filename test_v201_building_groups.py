"""test_v201_building_groups.py

Headless tests for the 2.0.1 [Group similar photos] button and command line (building_groups.py, code 141):
  - photos of the same (synthetic) facade seen from different viewpoints end up in one group, different facades
    in different groups; EXIF GPS links photos taken at the same site;
  - assign_tags: new numbers after the highest tag in the data folder, untagged photos join an existing tag;
  - renaming moves photo, JSON (imagePath) and masks; the queue and the photo on screen follow;
  - preview sheets and report; nothing renamed when the operator says no or cancels; progress is reported.

Run with:  python3 -m unittest test_v201_building_groups -v
"""
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import building_groups as bg
import crack_segmentation_gui as gm
from test_gui_shell_parity import make_app


def _facade(seed, size=(900, 1200)):
    """A textured 'building' with plenty of SIFT corners."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), rng.integers(60, 200, 3), np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        cw, ch = int(rng.integers(12, 70)), int(rng.integers(12, 70))
        cv2.rectangle(img, (x, y), (x + cw, y + ch), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    for _ in range(80):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(4, 20)),
                   tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def _view(facade, k):
    """The facade from another viewpoint: perspective warp + shift + brightness."""
    h, w = facade.shape[:2]
    d = 40 + 25 * k
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d // 2], [w - d // 3, 0], [w, h - d // 2], [d // 2, h]])
    out = cv2.warpPerspective(facade, cv2.getPerspectiveTransform(src, dst), (w, h))
    return cv2.convertScaleAbs(out, alpha=1.0 + 0.05 * k, beta=5 * k)


def _write_gps_jpeg(path, img, lat, lon):
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()

    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        return (float(d), float(m), round(((v - d) * 60 - m) * 60, 4))
    exif[0x8825] = {1: "N", 2: dms(lat), 3: "E", 4: dms(lon)}
    pil.save(path, exif=exif)


class _Folder(unittest.TestCase):

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="crackseg_groups_")
        self.addCleanup(shutil.rmtree, self.data, True)
        self.images = os.path.join(self.data, "Images")
        self.jsons = os.path.join(self.data, "JSON files")
        self.binary = os.path.join(self.data, "Binary files")
        for d in (self.images, self.jsons, self.binary):
            os.makedirs(d)

    def _photo(self, name, img):
        cv2.imwrite(os.path.join(self.images, name), img)

    def _buildings(self):
        """3 facades x 3 views, plus 2 lone photos; hash-like names, as in a downloaded dataset."""
        names = {}
        for b in range(3):
            facade = _facade(100 + b)
            for k in range(3):
                name = f"{b}{k}a7f{b * 31 + k:04x}.png"
                self._photo(name, _view(facade, k))
                names[name] = b
        for b in (7, 8):
            name = f"lone{b}.png"
            self._photo(name, _facade(500 + b))
            names[name] = b
        return names


class TestFindGroups(_Folder):

    def test_same_facade_grouped_others_apart(self):
        truth = self._buildings()
        photos = bg.list_photos(self.images)
        seen = []
        result = bg.find_groups(self.images, photos, bg.GroupingParams(neighbours=4),
                                progress=lambda **s: seen.append(s["phase"]))
        groups = [sorted(truth[p] for p in g) for g in result["groups"]]
        self.assertEqual(sorted(map(tuple, groups)), [(0, 0, 0), (1, 1, 1), (2, 2, 2), (7,), (8,)])
        self.assertIn("Reading the photos", seen)
        self.assertIn("Comparing similar photos", seen)
        self.assertLessEqual(result["pairs"], 11 * 4)

    def test_gps_links_photos_taken_at_the_same_site(self):
        _write_gps_jpeg(os.path.join(self.images, "a.jpg"), _facade(1), 37.5000, 15.0000)
        _write_gps_jpeg(os.path.join(self.images, "b.jpg"), _facade(2), 37.50005, 15.00005)   # ~7 m away
        _write_gps_jpeg(os.path.join(self.images, "c.jpg"), _facade(3), 37.6000, 15.1000)     # km away
        self.assertIsNotNone(bg.read_exif_site(os.path.join(self.images, "a.jpg")))
        result = bg.find_groups(self.images, bg.list_photos(self.images), bg.GroupingParams(neighbours=1))
        self.assertIn(["a.jpg", "b.jpg"], result["groups"])
        self.assertIn(["c.jpg"], result["groups"])
        self.assertEqual(result["gps"], 3)

    def test_cancel(self):
        self._buildings()
        with self.assertRaises(bg.GroupingCancelled):
            bg.find_groups(self.images, bg.list_photos(self.images), bg.GroupingParams(), cancelled=lambda: True)


class TestTags(_Folder):

    def test_new_groups_numbered_after_the_highest_tag(self):
        os.makedirs(os.path.join(self.data, "Test set", "Images"))
        open(os.path.join(self.data, "Test set", "Images", "x__BLDG041.jpg"), "w").close()
        self.assertEqual(bg.highest_building_index(self.data), 41)
        plan, info = bg.assign_tags([["a.jpg", "b.jpg"], ["c.jpg"]], 42)
        self.assertEqual(plan, {"a.jpg": 42, "b.jpg": 42, "c.jpg": 43})
        self.assertEqual(info, {"joined": 0, "mixed": []})

    def test_untagged_photos_join_an_existing_tag(self):
        plan, info = bg.assign_tags([["a__BLDG003.jpg", "b__BLDG003.jpg", "n.jpg", "x__BLDG009.jpg"],
                                     ["t__BLDG005.jpg"]], 10)
        self.assertEqual(plan, {"n.jpg": 3})
        self.assertEqual(info, {"joined": 1, "mixed": [[3, 9]]})

    def test_rename_moves_json_and_masks(self):
        self._photo("p1.png", _facade(1))
        with open(os.path.join(self.jsons, "p1.json"), "w") as f:
            json.dump({"imagePath": "p1.png", "shapes": []}, f)
        for sfx in ("-crack_mask", "-seg"):
            open(os.path.join(self.binary, f"p1{sfx}.png"), "w").close()
        open(os.path.join(self.binary, "p10-crack_mask.png"), "w").close()   # another photo: untouched
        new = bg.rename_with_tag(self.data, "p1.png", 7)
        self.assertEqual(new, "p1__BLDG007.png")
        self.assertEqual(sorted(os.listdir(self.images)), ["p1__BLDG007.png"])
        with open(os.path.join(self.jsons, "p1__BLDG007.json")) as f:
            self.assertEqual(json.load(f)["imagePath"], "p1__BLDG007.png")
        self.assertEqual(sorted(os.listdir(self.binary)),
                         ["p10-crack_mask.png", "p1__BLDG007-crack_mask.png", "p1__BLDG007-seg.png"])


class TestPreviewAndCli(_Folder):

    def test_cli_preview_then_apply(self):
        truth = self._buildings()
        out = io.StringIO()
        with redirect_stdout(out):
            bg.main(["--data-dir", self.data, "--neighbours", "4"])
        self.assertEqual(sorted(os.listdir(self.images)), sorted(truth))     # preview only
        preview = os.path.join(self.data, bg.PREVIEW_DIR_NAME)
        sheets = sorted(f for f in os.listdir(preview) if f.endswith(".jpg"))
        self.assertEqual(len([s for s in sheets if s.startswith("group_")]), 3)
        self.assertEqual(len([s for s in sheets if s.startswith("alone_")]), 1)
        with open(os.path.join(preview, bg.REPORT_NAME)) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 11)
        self.assertIn("run again with --apply", out.getvalue())
        with redirect_stdout(io.StringIO()):
            bg.main(["--data-dir", self.data, "--neighbours", "4", "--apply"])
        tagged = os.listdir(self.images)
        self.assertTrue(all("__BLDG" in n for n in tagged))
        by_tag = {}
        for n in tagged:
            by_tag.setdefault(bg.building_index(n), set()).add(truth[n.split("__BLDG")[0] + ".png"])
        self.assertEqual(sorted(len(v) for v in by_tag.values()), [1, 1, 1, 1, 1])   # one building per tag
        self.assertEqual(len(by_tag), 5)


class TestButton(_Folder):

    def setUp(self):
        super().setUp()
        self.truth = self._buildings()
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR = self.data
        cfg.BUILDING_GROUPS_NEIGHBOURS = 4
        cfg.queue_idle = False
        cfg.modalita_scelta = "1"
        cfg.image_queue = [os.path.join(self.images, n) for n in sorted(self.truth)]
        cfg.CURRENT_IMAGE_PATH = cfg.image_queue[0]
        cfg.JSON_OUTPUT_PATH = os.path.join(self.jsons, os.path.splitext(sorted(self.truth)[0])[0] + ".json")

    def _run(self, confirm=True):
        with mock.patch.object(self.app, "_confirm_building_groups", return_value=confirm) as asked, \
                mock.patch.object(self.app, "_notify_building_groups") as told, \
                mock.patch.object(self.app, "_pump_extra_events"), redirect_stdout(io.StringIO()):
            renamed = self.app.process_keypress(bg.GROUP_PHOTOS_CODE)
        return asked, told

    def test_rename_and_queue_follows(self):
        asked, told = self._run()
        message = asked.call_args[0][0]
        self.assertIn("11 photo(s) in 5 group(s)", message)
        self.assertIn("3 group(s) of 2 or more photos", message)
        self.assertIn("11 photo(s) will be renamed", message)
        self.assertTrue(all("__BLDG" in n for n in os.listdir(self.images)))
        cfg = self.app.cfg
        self.assertTrue(all(os.path.exists(p) for p in cfg.image_queue))
        self.assertIn("__BLDG", cfg.CURRENT_IMAGE_PATH)
        self.assertTrue(cfg.JSON_OUTPUT_PATH.endswith(os.path.splitext(os.path.basename(cfg.CURRENT_IMAGE_PATH))[0]
                                                      + ".json"))
        self.assertIn("11 photo(s) renamed", told.call_args[0][0])

    def test_no_keeps_the_names(self):
        self._run(confirm=False)
        self.assertEqual(sorted(os.listdir(self.images)), sorted(self.truth))
        self.assertTrue(os.listdir(os.path.join(self.data, bg.PREVIEW_DIR_NAME)))

    def test_escape_stops_and_renames_nothing(self):
        self.app._pending_keys.append(27)
        asked, _ = self._run()
        asked.assert_not_called()
        self.assertEqual(sorted(os.listdir(self.images)), sorted(self.truth))

    def test_second_run_has_nothing_to_rename(self):
        self._run()
        asked, told = self._run()
        asked.assert_not_called()
        self.assertIn("Every photo already has its building tag", told.call_args[0][0])


class TestGui(unittest.TestCase):

    def test_sidebar_menu_and_progress_text(self):
        codes = [entry[3] for _, entries in gm._LEFT_SIDEBAR_SECTIONS for entry in entries]
        self.assertIn(bg.GROUP_PHOTOS_CODE, codes)
        self.assertIn(bg.GROUP_PHOTOS_CODE, [code for _, code in gm._FILE_COMMANDS])   # next to Auto split
        import time
        head, text, percent = bg.progress_text({"phase": "Comparing similar photos", "done": 50, "total": 200,
                                                "start": time.time() - 100})
        self.assertEqual(head, "Comparing similar photos")
        self.assertEqual(percent, 25.0)
        self.assertEqual(text, "50 of 200 (25%) - about 5 min left")

    def test_gui_window_opens_updates_and_cancels(self):
        import test_gui_panel_wiring as wiring
        app = wiring.make_app(self)
        state = {"phase": "Reading the photos", "done": 3, "total": 10, "start": 0}
        app._group_progress_open(state)
        window = app._group_window
        self.assertEqual(window.bar.kw.get("value"), 30.0)
        self.assertFalse(app._sam_cancel_requested())
        window.cancel()
        self.assertTrue(app._sam_cancel_requested())
        app._group_progress_close()
        self.assertIsNone(app._group_window)


if __name__ == "__main__":
    unittest.main()
