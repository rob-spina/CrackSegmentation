"""test_v201_active_learning.py

Headless tests for the 2.0.1 [Active learning] button (active_learning.py, code 153), active_learning_prep.py and the
new sam_batch.py options:
  - the newest 'SAM model/rounds/round_NN/best_lora.pth' becomes the SAM model; without rounds the model in use
    (round 0) is used;
  - the photos still to label are ranked by uncertainty from the crack maps and the ACTIVE_LEARNING_BATCH most
    uncertain are loaded in Mode 1 with SAM click to accept on; missing maps: compute here / rank the others / cancel;
  - Test set IoU / F1 written to history.csv once per model and compared with the previous round;
  - active_learning_prep.py exports Training + Validation set (never Test set) as images/ + masks/;
  - sam_batch.py --images-subdir / --skip-labelled.

Run with:  python3 -m unittest test_v201_active_learning -v
"""
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import active_learning as al
import active_learning_prep as prep
import crack_segmentation_gui as gm
import sam_batch
from sam_autoseg import model_tag, probability_cache_path
from test_gui_shell_parity import make_app

H, W = 40, 60


def _prob_map(uncertain_share):
    """A map whose crack-like pixels are a given share in the hesitant band."""
    prob = np.zeros((H, W), np.uint8)
    n_unc = int(round(100 * uncertain_share))
    flat = prob.reshape(-1)
    flat[:100 - n_unc] = 240         # sure
    flat[100 - n_unc:100] = 128      # hesitant
    return prob


class _Data(unittest.TestCase):

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="crackseg_al_")
        self.addCleanup(shutil.rmtree, self.data, True)
        for d in ("Images", "JSON files", "Binary files", "SAM model"):
            os.makedirs(os.path.join(self.data, d))
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR = self.data
        cfg.queue_idle = True
        cfg.ACTIVE_LEARNING_BATCH = 2
        self.base_weights = os.path.join(self.data, "SAM model", "cracksam_vit_h_lora_r4.pth")
        with open(self.base_weights, "wb") as f:
            f.write(b"facade390 weights")
        self.settings = {"backend": "sam_lora", "lora_weights": self.base_weights,
                         "sam_checkpoint": self.base_weights, "python": sys.executable}
        self.app._save_sam_settings(self.settings)

    def _photo(self, name, folder="Images", labelled=False):
        path = os.path.join(self.data, folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, np.full((H, W, 3), 120, np.uint8))
        if labelled:
            with open(os.path.join(self.data, "JSON files", os.path.splitext(name)[0] + ".json"), "w") as f:
                json.dump({"shapes": []}, f)
        return path

    def _map(self, weights, photo, prob):
        path = probability_cache_path(os.path.join(self.data, "SAM model"), model_tag(weights), photo)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        time.sleep(0.01)
        cv2.imwrite(path, prob)
        os.utime(path, (time.time() + 5, time.time() + 5))
        return path

    def _round(self, n, content):
        path = os.path.join(self.data, "SAM model", "rounds", f"round_{n:02d}", "best_lora.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        os.utime(path, (time.time() + n, time.time() + n))
        return path

    def _press(self, choice=None):
        with mock.patch.object(self.app, "_usable_sam_settings", side_effect=lambda: self.app._load_sam_settings()), \
                mock.patch.object(self.app, "_confirm_active_learning", return_value=choice) as ask, \
                mock.patch.object(self.app, "_notify_active_learning") as told, redirect_stdout(io.StringIO()):
            result = self.app.process_keypress(al.ACTIVE_LEARNING_CODE) if not self.app.cfg.queue_idle \
                else self.app._handle_idle_key(al.ACTIVE_LEARNING_CODE)
        return result, ask, told


class TestButton(_Data):

    def test_round_0_ranks_and_loads_the_most_uncertain(self):
        shares = {"a.jpg": 0.1, "b.jpg": 0.6, "c.jpg": 0.3, "d.jpg": 0.9}
        for name, share in shares.items():
            self._map(self.base_weights, self._photo(name), _prob_map(share))
        self._photo("done.jpg", labelled=True)
        result, ask, told = self._press()
        self.assertTrue(result)
        ask.assert_not_called()
        cfg = self.app.cfg
        self.assertEqual([os.path.basename(p) for p in cfg.image_queue], ["d.jpg", "b.jpg"])
        self.assertEqual(cfg.modalita_scelta, "1")
        self.assertTrue(cfg.sam_accept_state["sticky"])
        self.assertIn("round 0", told.call_args[0][0])
        tag = model_tag(self.base_weights)
        with open(os.path.join(self.data, "SAM model", "rounds", f"ranking_{tag}.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["photo"] for r in rows], ["d.jpg", "b.jpg", "c.jpg", "a.jpg"])
        self.assertEqual([r["proposed"] for r in rows], ["yes", "yes", "", ""])

    def test_newest_round_becomes_the_model(self):
        r1 = self._round(1, b"round one")
        r2 = self._round(2, b"round two")
        for name, share in {"a.jpg": 0.2, "b.jpg": 0.5}.items():
            self._map(r2, self._photo(name), _prob_map(share))
        result, _, told = self._press()
        self.assertTrue(result)
        self.assertEqual(self.app._load_sam_settings()["lora_weights"], r2)
        self.assertIn("round_02", told.call_args[0][0])
        self.assertNotEqual(model_tag(r1), model_tag(r2))

    def test_missing_maps_rank_only_or_cancel(self):
        self._map(self.base_weights, self._photo("a.jpg"), _prob_map(0.4))
        self._photo("b.jpg")                                   # no map
        result, ask, _ = self._press(None)
        self.assertFalse(result)
        self.assertIn("1 without this model's crack map", ask.call_args[0][0])
        result, _, _ = self._press("skip")
        self.assertTrue(result)
        self.assertEqual([os.path.basename(p) for p in self.app.cfg.image_queue], ["a.jpg"])

    def test_missing_maps_computed_here(self):
        self._photo("a.jpg")
        calls = []

        def fake_model(settings):
            calls.append(self.app.cfg.CURRENT_IMAGE_PATH)
            return _prob_map(0.5)
        with mock.patch.object(self.app, "_sam_run_model_on_photo", side_effect=fake_model):
            result, _, _ = self._press("local")
        self.assertTrue(result)
        self.assertEqual([os.path.basename(p) for p in calls], ["a.jpg"])
        self.assertTrue(os.path.isfile(probability_cache_path(os.path.join(self.data, "SAM model"),
                                                              model_tag(self.base_weights), calls[0])))

    def test_everything_labelled(self):
        self._photo("a.jpg", labelled=True)
        result, _, told = self._press()
        self.assertFalse(result)
        self.assertIn("already labelled", told.call_args[0][0])

    def test_test_set_score_and_history(self):
        self._map(self.base_weights, self._photo("a.jpg"), _prob_map(0.4))
        test_photo = self._photo("t1.jpg", folder=os.path.join("Test set", "Images"))
        gt = np.zeros((H, W), np.uint8)
        gt[10:20, :] = 255
        cv2.imwrite(os.path.join(self.data, "Binary files", "t1-crack_mask.png"), gt)
        pred = np.zeros((H, W), np.uint8)
        pred[10:15, :] = 230                                   # half of the crack found
        self._map(self.base_weights, test_photo, pred)
        _, _, told = self._press()
        self.assertIn("IoU 0.500", told.call_args[0][0])
        r1 = self._round(1, b"round one")
        self._map(r1, self._photo("b.jpg"), _prob_map(0.4))
        self._map(r1, os.path.join(self.data, "Images", "a.jpg"), _prob_map(0.4))
        better = np.zeros((H, W), np.uint8)
        better[10:20, :] = 230
        self._map(r1, test_photo, better)
        _, _, told = self._press()
        self.assertIn("IoU 1.000", told.call_args[0][0])
        self.assertIn("previous model (round 0", told.call_args[0][0])
        self.assertIn("better", told.call_args[0][0])
        with open(os.path.join(self.data, "SAM model", "rounds", al.HISTORY_NAME)) as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_from_a_photo_the_queue_restarts(self):
        for name, share in {"a.jpg": 0.2, "b.jpg": 0.5}.items():
            self._map(self.base_weights, self._photo(name), _prob_map(share))
        cfg = self.app.cfg
        cfg.queue_idle = False
        cfg.CURRENT_IMAGE_PATH = os.path.join(self.data, "Images", "a.jpg")
        cfg.modalita_scelta = "2"
        with mock.patch.object(self.app, "_settle_unsaved_changes", return_value=True):
            result, _, _ = self._press()
        self.assertTrue(result, "the image loop ends: the new queue opens")
        self.assertEqual(cfg.navigate_target, 0)
        self.assertEqual(cfg.modalita_scelta, "1")


class TestPrep(unittest.TestCase):

    def test_exports_training_and_validation_not_test(self):
        data = tempfile.mkdtemp(prefix="crackseg_alprep_")
        out = tempfile.mkdtemp(prefix="crackseg_alout_")
        self.addCleanup(shutil.rmtree, data, True)
        self.addCleanup(shutil.rmtree, out, True)
        for set_name, photo, mask_where in (("Training set", "t__BLDG001.JPG", "Training set"),
                                            ("Validation set", "v.png", ""), ("Test set", "x.jpg", ""),
                                            ("Training set", "nomask.jpg", None)):
            os.makedirs(os.path.join(data, set_name, "Images"), exist_ok=True)
            cv2.imwrite(os.path.join(data, set_name, "Images", photo), np.zeros((8, 8, 3), np.uint8))
            if mask_where is not None:
                folder = os.path.join(data, mask_where, "Binary files")
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, os.path.splitext(photo)[0] + "-crack_mask.png"),
                            np.zeros((8, 8), np.uint8))
        with redirect_stdout(io.StringIO()) as printed:
            prep.main(["--data-dir", data, "--out", out])
        self.assertEqual(sorted(os.listdir(os.path.join(out, "images"))), ["t__BLDG001.jpg", "v.jpg"])
        self.assertEqual(sorted(os.listdir(os.path.join(out, "masks"))), ["t__BLDG001.png", "v.png"])
        self.assertIn("nomask.jpg", printed.getvalue())
        self.assertIsNotNone(cv2.imread(os.path.join(out, "images", "v.jpg")))   # a png behind a .jpg name


class TestPrepUncertain(unittest.TestCase):

    def test_uncertain_cracks_become_the_ignore_band(self):
        data = tempfile.mkdtemp(prefix="crackseg_alunc_")
        out = tempfile.mkdtemp(prefix="crackseg_aluncout_")
        self.addCleanup(shutil.rmtree, data, True)
        self.addCleanup(shutil.rmtree, out, True)
        imgs = os.path.join(data, "Training set", "Images")
        bins = os.path.join(data, "Training set", "Binary files")
        os.makedirs(imgs)
        os.makedirs(bins)
        cv2.imwrite(os.path.join(imgs, "p.jpg"), np.zeros((10, 10, 3), np.uint8))
        crack = np.zeros((10, 10), np.uint8)
        crack[2:5] = 255                                 # the app's crack mask: every crack, uncertain ones too
        unc = np.zeros((10, 10), np.uint8)
        unc[3:5] = 255                                   # the uncertain ones
        cv2.imwrite(os.path.join(bins, "p-crack_mask.png"), crack)
        cv2.imwrite(os.path.join(bins, "p-crack_uncertain_mask.png"), unc)
        with redirect_stdout(io.StringIO()) as printed:
            prep.main(["--data-dir", data, "--out", out])
        m = cv2.imread(os.path.join(out, "masks", "p.png"), cv2.IMREAD_GRAYSCALE)
        self.assertTrue((m[2] == 255).all())             # the crack stays a crack
        self.assertTrue((m[3:5] == 128).all())           # the doubt: ignore band
        self.assertTrue((m[6:] == 0).all())
        self.assertIn("1 with uncertain cracks", printed.getvalue())
        ign = np.zeros((10, 10), np.uint8)
        ign[7:9, 0:5] = 255                              # an [I] ignore region (too many small cracks)
        cv2.imwrite(os.path.join(bins, "p-ignore_mask.png"), ign)
        with redirect_stdout(io.StringIO()):
            prep.main(["--data-dir", data, "--out", out])
        m = cv2.imread(os.path.join(out, "masks", "p.png"), cv2.IMREAD_GRAYSCALE)
        self.assertTrue((m[7:9, 0:5] == 128).all())
        self.assertTrue((m[7:9, 5:] == 0).all())
        with redirect_stdout(io.StringIO()):
            prep.main(["--data-dir", data, "--out", out, "--uncertain", "background"])
        m = cv2.imread(os.path.join(out, "masks", "p.png"), cv2.IMREAD_GRAYSCALE)
        self.assertFalse((m == 128).any())

    def test_score_ignores_uncertain_pixels(self):
        gt = np.zeros((4, 4), np.uint8)
        gt[0] = 255
        pred = np.zeros((4, 4), np.uint8)
        pred[:2] = 255
        ign = np.zeros((4, 4), np.uint8)
        ign[1] = 255
        self.assertEqual(al.iou_f1(pred, gt, 0.5), (4, 4, 0))
        self.assertEqual(al.iou_f1(pred, gt, 0.5, ign), (4, 0, 0))


class TestSamBatchOptions(unittest.TestCase):

    def test_images_subdir_and_skip_labelled(self):
        data = tempfile.mkdtemp(prefix="crackseg_albatch_")
        self.addCleanup(shutil.rmtree, data, True)
        for sub, name in (("Images", "a.jpg"), ("Images", "b.jpg"), (os.path.join("Test set", "Images"), "t.jpg")):
            os.makedirs(os.path.join(data, sub), exist_ok=True)
            cv2.imwrite(os.path.join(data, sub, name), np.zeros((16, 16, 3), np.uint8))
        os.makedirs(os.path.join(data, "JSON files"))
        with open(os.path.join(data, "JSON files", "a.json"), "w") as f:
            f.write("{}")
        seen = []

        def factory(args):
            def predict(path):
                seen.append(os.path.basename(path))
                return np.zeros((16, 16), np.float32)
            return predict
        base = ["--data-dir", data, "--prob-only", "--lora-weights", "", "--checkpoint", "x.pt"]
        with redirect_stdout(io.StringIO()):
            sam_batch.run(sam_batch.parse_args(base + ["--skip-labelled"]), factory)
            sam_batch.run(sam_batch.parse_args(base + ["--images-subdir", os.path.join("Test set", "Images")]),
                          factory)
        self.assertEqual(seen, ["b.jpg", "t.jpg"])


class TestGui(unittest.TestCase):

    def test_button_and_menu(self):
        codes = [entry[3] for _, entries in gm._LEFT_SIDEBAR_SECTIONS for entry in entries]
        self.assertIn(al.ACTIVE_LEARNING_CODE, codes)
        self.assertIn(al.ACTIVE_LEARNING_CODE, [code for _, code in gm._TOOL_COMMANDS])


if __name__ == "__main__":
    unittest.main()
