"""test_v200_sam_batch.py

Headless tests for sam_batch.py (2.0.0): SAM auto-segment for a whole data folder from the command line.
A stand-in predictor replaces the model (no PyTorch needed):
  - every photo of 'Images' gets its JSON (cracks with "source": "sam") and its binary crack mask;
  - photos that already have a JSON are skipped by default; --existing update keeps the operator's cracks and
    replaces only an earlier SAM run;
  - --limit / --only, the CSV report, an error on one photo does not stop the others;
  - no photo is renamed into a building group; the model is loaded once.

Run with:  python3 -m unittest test_v200_sam_batch -v
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import sam_batch

H, W = 200, 300


def _prob_for(path):
    """Stand-in model: a strong crack along y=100 and a branch down from x=150; 'broken' photos fail."""
    if "broken" in os.path.basename(path):
        raise RuntimeError("model failure")
    prob = np.zeros((H, W), np.float32)
    cv2.line(prob, (20, 100), (280, 100), 0.9, 3)
    cv2.line(prob, (150, 100), (150, 180), 0.9, 3)
    return prob


class TestSamBatch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "Images"))
        for name in ("a.jpg", "b.png", "c.JPG"):
            cv2.imwrite(os.path.join(self.tmp, "Images", name), np.full((H, W, 3), 190, np.uint8))
        open(os.path.join(self.tmp, "Images", "notes.txt"), "w").close()
        self.loads = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _factory(self, args):
        self.loads += 1
        return _prob_for

    def _run(self, *extra):
        args = sam_batch.parse_args(["--data-dir", self.tmp, *extra])
        return sam_batch.run(args, predictor_factory=self._factory)

    def _json(self, base):
        with open(os.path.join(self.tmp, "JSON files", base + ".json"), encoding="utf-8") as f:
            return json.load(f)

    def test_whole_folder(self):
        rows = self._run()
        self.assertEqual([r["status"] for r in rows], ["segmented"] * 3)
        self.assertEqual(self.loads, 1)
        for base in ("a", "b", "c"):
            shapes = [s for s in self._json(base)["shapes"] if s["shape_type"] == "linestrip"]
            self.assertEqual(len(shapes), 2)   # main crack + branch
            self.assertTrue(all(s["source"] == "sam" for s in shapes))
            mask = cv2.imread(os.path.join(self.tmp, "Binary files", f"{base}-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(int((mask > 0).sum()), 500)
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "Images"))), ["a.jpg", "b.png", "c.JPG", "notes.txt"])

    def test_report(self):
        self._run()
        with open(os.path.join(self.tmp, "sam_batch_report.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["photo"] for r in rows], ["a.jpg", "b.png", "c.JPG"])
        self.assertEqual(rows[0]["sam_cracks"], "2")

    def test_existing_json_skipped_by_default(self):
        self._run("--only", "a.jpg")
        before = os.path.getmtime(os.path.join(self.tmp, "JSON files", "a.json"))
        rows = self._run()
        self.assertEqual(sorted(r["photo"] for r in rows), ["b.png", "c.JPG"])
        self.assertEqual(os.path.getmtime(os.path.join(self.tmp, "JSON files", "a.json")), before)

    def test_update_keeps_operator_cracks(self):
        self._run("--only", "a")
        data = self._json("a")
        own = {"label": "crack", "points": [[30, 40], [120, 40]], "shape_type": "linestrip", "group_id": None,
               "flags": {}, "description": ""}
        data["shapes"].append(own)
        with open(os.path.join(self.tmp, "JSON files", "a.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
        rows = self._run("--only", "a", "--existing", "update")
        self.assertEqual(rows[0]["status"], "updated")
        self.assertEqual(rows[0]["operator_cracks_kept"], 1)
        shapes = [s for s in self._json("a")["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(sum(1 for s in shapes if s.get("source") == "sam"), 2)   # replaced, not doubled
        self.assertEqual(sum(1 for s in shapes if not s.get("source")), 1)

    def test_limit(self):
        rows = self._run("--limit", "2")
        self.assertEqual(len(rows), 2)

    def test_error_on_one_photo_does_not_stop_the_rest(self):
        cv2.imwrite(os.path.join(self.tmp, "Images", "broken.jpg"), np.full((H, W, 3), 190, np.uint8))
        rows = self._run()
        self.assertEqual(sorted(r["status"] for r in rows), ["error", "segmented", "segmented", "segmented"])

    def test_uncertain_option(self):
        self._run("--only", "a", "--uncertain-below", "0.95")
        shapes = [s for s in self._json("a")["shapes"] if s["shape_type"] == "linestrip"]
        self.assertTrue(all((s.get("flags") or {}).get("uncertain_manual") for s in shapes))

    def test_nothing_to_do_does_not_load_the_model(self):
        self._run()
        self.loads = 0
        self.assertEqual(self._run(), [])
        self.assertEqual(self.loads, 0)

    def test_missing_images_folder(self):
        shutil.rmtree(os.path.join(self.tmp, "Images"))
        with self.assertRaises(SystemExit):
            self._run()

    def test_model_arguments_shared_with_sam_infer(self):
        args = sam_batch.parse_args(["--data-dir", self.tmp, "--sam-checkpoint", "s.pth", "--lora-weights", "l.pth"])
        self.assertEqual((args.model_type, args.lora_r, args.tile), ("vit_h", 4, 512))
        self.assertTrue(args.code_dir.endswith("cracksam"))


if __name__ == "__main__":
    unittest.main()
