"""test_v201_auto_ignore.py -- 2.0.1 [Auto ignore] (code 162): ignore regions drawn automatically around trees,
fences, people, vehicles, rubble, from a SegFormer ADE20K class map (ignore_infer.py, replaced here by a stand-in
script), plus auto_ignore_batch.py --evaluate against hand-drawn ignore masks.

Run with:  python3 -m unittest test_v201_auto_ignore -v
"""
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import auto_ignore as ai
import auto_ignore_batch as aib
from auto_ignore import AUTO_IGNORE_CODE, AUTO_IGNORE_SOURCE
from config import Config
from test_v200_sam import _sam_app, _press, W, H

ID2LABEL = {0: "wall", 1: "building;edifice", 2: "sky", 4: "tree", 12: "person;individual;someone",
            32: "fence;fencing", 9: "grass"}

# Stand-in for ignore_infer.py: a class map at half size -- a tree with holes on the left, a person, a tiny
# fence speck, building elsewhere.
_FAKE_IGNORE = textwrap.dedent('''
    import argparse, json, sys
    import cv2, numpy as np
    p = argparse.ArgumentParser()
    for a in ("--image", "--out", "--labels-out", "--model", "--long-side", "--device"):
        p.add_argument(a, default="")
    args = p.parse_args()
    if "fail" in args.model:
        print("[IGNORE] missing package: transformers"); sys.exit(4)
    h, w = cv2.imread(args.image).shape[:2]
    c = np.full((h // 2, w // 2), 1, np.uint8)
    if "empty" not in args.model:
        c[0:60, 0:40] = 4                 # tree touching the top-left corner (photo 0-120 x 0-80)
        c[20:25, 15:20] = 1               # a hole in the crown
        c[70:95, 100:112] = 12            # person
        c[5:6, 140:141] = 32              # 1 px fence speck: dropped
    cv2.imwrite(args.out, c)
    json.dump({"model": args.model, "id2label": %s}, open(args.labels_out, "w"))
    print("[IGNORE] photo 1/1 done")
''' % json.dumps({str(k): v for k, v in ID2LABEL.items()}))


class TestPure(unittest.TestCase):

    def test_class_ids_synonyms(self):
        ids = ai.class_ids(ID2LABEL, ("tree", "person", "fence", "car"))
        self.assertEqual(set(ids), {4, 12, 32})
        self.assertEqual(ids[12], "person")

    def test_slug(self):
        self.assertEqual(ai.model_slug("nvidia/segformer-b5-finetuned-ade-640-640", 1024),
                         "nvidia__segformer-b5-finetuned-ade-640-640-L1024")

    def test_holes_filled_specks_dropped_edges_snapped(self):
        classes = np.full((100, 150), 1, np.uint8)
        classes[0:60, 0:40] = 4
        classes[20:25, 15:20] = 1
        classes[5, 140] = 4
        polys, found = ai.regions_from_classes(classes, ID2LABEL, ("tree",), (300, 200), margin_px=0)
        self.assertEqual(len(polys), 1)
        mask = ai.polygons_mask(polys, (300, 200))
        self.assertEqual(mask[45, 35], 255)                 # the hole inside the crown is covered
        self.assertEqual(mask[0, 0], 255)                   # reaches the photo corner
        self.assertEqual(mask[150, 200], 0)
        self.assertAlmostEqual(found["tree"], 2400 / 15000.0, places=2)

    def test_margin_grows_the_region(self):
        classes = np.full((100, 150), 1, np.uint8)
        classes[40:60, 60:80] = 12
        tight = ai.polygons_mask(ai.regions_from_classes(classes, ID2LABEL, ("person",), (300, 200),
                                                         margin_px=0)[0], (300, 200))
        wide = ai.polygons_mask(ai.regions_from_classes(classes, ID2LABEL, ("person",), (300, 200),
                                                        margin_px=12)[0], (300, 200))
        self.assertGreater(np.count_nonzero(wide), np.count_nonzero(tight))

    def test_nothing(self):
        polys, found = ai.regions_from_classes(np.ones((50, 50), np.uint8), ID2LABEL, ("tree",), (100, 100))
        self.assertEqual((polys, found), ([], {}))


class _AppBase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = _sam_app(self.tmp)
        script = os.path.join(self.tmp, "fake_ignore.py")
        with open(script, "w") as f:
            f.write(_FAKE_IGNORE)
        self.app._auto_ignore_script_path = lambda: script
        self.app.cfg.saved_ignores = []
        self.app.cfg.queue_idle = False
        self.notes = []
        self.app._notify_auto_ignore = self.notes.append

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_it(self, confirm=True):
        with mock.patch.object(self.app, "_confirm_auto_ignore_replace", return_value=confirm):
            return _press(self.app, AUTO_IGNORE_CODE)

    def autos(self):
        return [r for r in self.app.cfg.saved_ignores if r.get("source") == AUTO_IGNORE_SOURCE]


class TestButton(_AppBase):

    def test_regions_added(self):
        self.run_it()
        autos = self.autos()
        self.assertEqual(len(autos), 2)                     # tree + person, not the speck
        mask = self.app._build_ignore_area_mask()
        self.assertEqual(mask[2, 2], 255)                   # tree at the corner
        self.assertEqual(mask[165, 210], 255)               # person
        self.assertEqual(mask[100, 260], 0)                 # building
        self.assertTrue(os.path.isfile(self.app._auto_ignore_map_path()))
        self.assertEqual(self.app.cfg.saved_cracks, [])

    def test_cached_map_reused(self):
        self.run_it()
        with mock.patch.object(self.app, "_run_sam_process") as run:
            self.run_it()
        run.assert_not_called()
        self.assertEqual(len(self.autos()), 2)              # replaced, not doubled

    def test_hand_drawn_regions_kept_and_replace_declined(self):
        hand = {'path': [(200, 10), (290, 10), (290, 60)], 'active': True, 'session_id': 'x'}
        self.app.cfg.saved_ignores.append(hand)
        self.run_it()
        self.assertIn(hand, self.app.cfg.saved_ignores)
        before = list(self.app.cfg.saved_ignores)
        self.run_it(confirm=False)
        self.assertEqual(self.app.cfg.saved_ignores, before)

    def test_undo_and_redo_the_whole_run(self):
        self.run_it()
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('u'))
        self.assertEqual(self.autos(), [])
        with mock.patch.object(self.app, "refresh_zoom_viewport"):
            self.app.process_keypress(ord('r'))
        self.assertEqual(len(self.autos()), 2)

    def test_json_round_trip_keeps_the_auto_flag(self):
        self.run_it()
        shapes = self.app._build_ignore_shapes()
        self.assertTrue(all(s["flags"].get("auto_ignore") for s in shapes))
        loaded = self.app._build_loaded_ignore([(0, 0), (5, 0), (5, 5)], shapes[0]["flags"])
        self.assertEqual(loaded.get("source"), AUTO_IGNORE_SOURCE)
        self.assertNotIn("source", self.app._build_loaded_ignore([(0, 0), (5, 0), (5, 5)], {"ignore": True}))

    def test_model_missing(self):
        self.app.cfg.AUTO_IGNORE_MODEL = "fail/model"
        self.run_it()
        self.assertEqual(self.autos(), [])
        self.assertIn("transformers", self.notes[-1])

    def test_nothing_found(self):
        self.app.cfg.AUTO_IGNORE_MODEL = "empty/model"
        self.run_it()
        self.assertEqual(self.autos(), [])
        self.assertIn("nothing to ignore", self.notes[-1])

    def test_class_list_change_needs_no_new_run(self):
        self.run_it()
        self.app.cfg.AUTO_IGNORE_CLASSES = ("person",)
        with mock.patch.object(self.app, "_run_sam_process") as run:
            self.run_it()
        run.assert_not_called()
        self.assertEqual(len(self.autos()), 1)

    def test_no_photo(self):
        self.app.cfg.queue_idle = True
        self.assertFalse(self.app.run_auto_ignore())

    def test_command(self):
        cmd = self.app._auto_ignore_command("a.jpg", "o.png", "l.json")
        self.assertEqual(cmd[0], sys.executable)            # the SAM Python
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--long-side") + 1], "1024")


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = Config()
        os.makedirs(os.path.join(self.tmp, "Images"))
        os.makedirs(os.path.join(self.tmp, "Binary files"))
        mdir = aib.maps_dir(self.tmp, self.cfg)
        os.makedirs(mdir)
        with open(os.path.join(mdir, "labels.json"), "w") as f:
            json.dump({"id2label": {str(k): v for k, v in ID2LABEL.items()}}, f)
        for name, hand_tree in (("a", True), ("b", False)):
            cv2.imwrite(os.path.join(self.tmp, "Images", name + ".jpg"), np.zeros((200, 300, 3), np.uint8))
            classes = np.full((100, 150), 1, np.uint8)
            classes[0:60, 0:40] = 4
            classes[80:100, 0:150] = 9                     # grass the hand did not mark
            cv2.imwrite(os.path.join(mdir, name + ".png"), classes)
            manual = np.zeros((200, 300), np.uint8)
            if hand_tree:
                manual[0:120, 0:80] = 255
            cv2.imwrite(os.path.join(self.tmp, "Binary files", name + "-ignore_mask.png"), manual)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scores_and_class_hints(self):
        lines = []
        self.cfg.AUTO_IGNORE_CLASSES = ("tree",)
        self.cfg.AUTO_IGNORE_MARGIN_PX = 0
        rows = aib.evaluate(self.tmp, self.cfg, os.path.join(self.tmp, "r.csv"), out=lines.append)
        self.assertEqual([r[0] for r in rows], ["a.jpg", "b.jpg"])
        self.assertGreater(rows[0][1], 0.85)               # IoU of the tree
        self.assertEqual(rows[1][3], 1.0)                  # nothing hand-drawn: recall 1
        self.assertLess(rows[1][2], 0.01)                  # but all of it is outside
        text = "\n".join(lines)
        self.assertIn("tree", text)
        self.assertIn("in the list", text)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "r.csv")))

    def test_no_maps(self):
        shutil.rmtree(aib.maps_dir(self.tmp, self.cfg))
        lines = []
        self.assertEqual(aib.evaluate(self.tmp, self.cfg, out=lines.append), [])
        self.assertIn("No class maps", lines[0])

    def test_photos_and_masks_found(self):
        photos = aib.find_photos(self.tmp)
        self.assertEqual(len(photos), 2)
        self.assertTrue(aib.manual_mask_path(self.tmp, photos[0]).endswith("a-ignore_mask.png"))


class TestInferScript(unittest.TestCase):

    def test_args_and_resize(self):
        import ignore_infer
        args = ignore_infer.parse_args(["--image", "a.jpg", "--out", "b.png"])
        self.assertEqual(args.long_side, 1024)
        with self.assertRaises(SystemExit):
            ignore_infer.parse_args(["--image", "a.jpg"])
        small = ignore_infer.resized(np.zeros((3000, 4000, 3), np.uint8), 1024)
        self.assertEqual(small.shape[:2], (768, 1024))
        self.assertEqual(ignore_infer.resized(np.zeros((300, 400, 3), np.uint8), 1024).shape[:2], (300, 400))


class TestGui(unittest.TestCase):

    def test_button_menu_lock(self):
        import crack_segmentation_gui as gm
        self.assertTrue(any(code == AUTO_IGNORE_CODE for _, code in gm._TOOL_COMMANDS))
        self.assertIn(AUTO_IGNORE_CODE, gm._SIDEBAR_ENABLED_STATES)
        with open(gm.__file__, encoding="utf-8") as f:
            self.assertIn('"Auto ignore", "", AUTO_IGNORE_CODE', f.read())


if __name__ == "__main__":
    unittest.main()
