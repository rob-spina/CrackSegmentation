"""active_learning.py
v2.0.1 [Active learning] (right sidebar "AI" section, Tools menu, code 153), mixed into CrackSegmentation, plus the
helpers used on Leonardo (active_learning_prep.py, docs/sam_model/active_learning_round_leonardo.sbatch).

One round of the loop:
  1. the operator labels the proposed photos with [SAM click to accept] (Training / Validation set; the Test set stays
     fixed, so that the rounds can be compared);
  2. on Leonardo the model is fine-tuned again, starting from the previous round's best_lora.pth, on every labelled
     photo; the job also computes the new model's crack maps of the photos still to label and of the Test set;
  3. the new best_lora.pth (and the maps) are copied back into '<data>/SAM model/rounds/round_NN/' (and the maps
     into '<data>/SAM model/probability maps/<tag>/');
  4. this button: takes the newest round's weights as the SAM model, gets the crack map of every photo still to
     label (from Leonardo when there, otherwise computed here after a confirmation), ranks the photos by how much the
     model hesitates (share of crack-like pixels with 0.3 <= p <= 0.7) and loads the ACTIVE_LEARNING_BATCH (50) most
     uncertain ones in Mode 1 with [SAM click to accept] on. When the Test set has labels and maps, the new model's
     IoU / F1 there is added to '<data>/SAM model/rounds/history.csv' and compared with the previous round.
Round 0 is the model in use (SAM + CrackSeg9K + Facade390): the button also works before any round exists.
"""
import csv
import datetime
import glob
import os
import time

import cv2
import numpy as np

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME, IGNORED_MASK_SUFFIXES
from sam_autoseg import model_tag, probability_cache_path, probability_uncertainty, read_cached_probability

ACTIVE_LEARNING_CODE = 153
ROUNDS_DIR_NAME = "rounds"
HISTORY_NAME = "history.csv"
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def newest_round_weights(model_dir):
    """(round name, path) of the newest best_lora.pth under '<SAM model>/rounds/', or (None, None)."""
    found = glob.glob(os.path.join(model_dir, ROUNDS_DIR_NAME, "*", "best_lora.pth"))
    if not found:
        return None, None
    path = max(found, key=lambda p: (os.path.getmtime(p), p))
    return os.path.basename(os.path.dirname(path)), path


def photos_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(PHOTO_EXTENSIONS) and not f.startswith(".")
                  and not os.path.splitext(f)[0].endswith("-seg"))


def ground_truth_mask(data_dir, photo):
    """The photo's labelled crack mask (crack = 255), searched next to it and in the data folder, or None."""
    base = os.path.splitext(os.path.basename(photo))[0]
    set_dir = os.path.dirname(os.path.dirname(photo))
    for folder in (os.path.join(set_dir, BINARY_DIR_NAME), os.path.join(data_dir, BINARY_DIR_NAME)):
        path = os.path.join(folder, f"{base}-crack_mask.png")
        if os.path.isfile(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None


def uncertain_mask(data_dir, photo):
    """The pixels left out of the score: uncertain cracks, [I] ignore regions and [D] detachments (as in training),
    or None."""
    base = os.path.splitext(os.path.basename(photo))[0]
    set_dir = os.path.dirname(os.path.dirname(photo))
    out = None
    for suffix in IGNORED_MASK_SUFFIXES:
        for folder in (os.path.join(set_dir, BINARY_DIR_NAME), os.path.join(data_dir, BINARY_DIR_NAME)):
            path = os.path.join(folder, base + suffix)
            if os.path.isfile(path):
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    out = m if out is None or out.shape != m.shape else np.maximum(out, m)
                break
    return out


def iou_f1(prob_u8, gt, threshold=0.5, ignore=None):
    """(tp, fp, fn) pixel counts; the uncertain cracks (ignore) count neither way, as in train.py."""
    pred = prob_u8 >= int(round(threshold * 255))
    truth = gt > 127
    if ignore is not None and ignore.shape == truth.shape:
        keep = ~(ignore > 127)   # the app's crack mask holds the uncertain cracks too: leave them all out
        pred, truth = pred & keep, truth & keep
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return tp, fp, fn


class ActiveLearningMixin:

    # --- Hooks (terminal default; the GUI asks with dialogs) -------------------------------------

    def _confirm_active_learning(self, message):
        """'local' = compute the missing maps here, 'skip' = rank only the photos with a map, None = cancel."""
        print("\n" + message)
        try:
            answer = input("Type c to compute them here, r to rank only the photos with a map, ENTER to cancel: ")
        except Exception:
            return None
        return {"c": "local", "r": "skip"}.get(answer.strip().lower())

    def _notify_active_learning(self, message):
        print(message)

    # --- The button ------------------------------------------------------------------------------

    def run_active_learning(self):
        """Code 153. True when a new Mode-1 queue of photos to label was loaded."""
        cfg = self.cfg
        data_dir = str(cfg.SCRIPT_DIR)
        if not cfg.queue_idle and cfg.CURRENT_IMAGE_PATH and not self._settle_unsaved_changes("active learning"):
            print(" [ACTIVE LEARNING] Cancelled.")
            return False
        settings = self._active_learning_settings()
        if not settings:
            return False
        tag = model_tag(settings.get("lora_weights", ""), settings.get("checkpoint", ""),
                        getattr(self.cfg, "SAM_UPSCALE", None))
        todo = self._photos_to_label(data_dir)
        if not todo:
            self._notify_active_learning("Active learning: every photo of 'Images' is already labelled.")
            return False
        maps = {p: probability_cache_path(self._sam_model_dir(), tag, p) for p in todo}
        missing = [p for p in todo if not os.path.isfile(maps[p])]
        if missing:
            choice = self._confirm_active_learning(
                f"Active learning, model {self._round_label(settings)}: {len(todo)} photo(s) to label, "
                f"{len(missing)} without this model's crack map.\n\n"
                f"Compute them here (about 1.5-4 min per 4K photo, i.e. roughly {len(missing) * 2.5 / 60:.1f} h), or "
                f"compute them on Leonardo with active_learning_round_leonardo.sbatch (about 9 s per photo) and copy "
                f"them back, or rank only the {len(todo) - len(missing)} photo(s) that already have one.")
            if choice is None:
                print(" [ACTIVE LEARNING] Cancelled.")
                return False
            if choice == "local" and not self._compute_maps_here(settings, missing, maps):
                return False
        ranked = self._rank_by_uncertainty(todo, maps)
        if not ranked:
            self._notify_active_learning("Active learning: no crack map available to rank the photos.")
            return False
        batch = ranked[:max(1, int(cfg.ACTIVE_LEARNING_BATCH))]
        self._write_round_report(tag, ranked, len(batch))
        test_line = self._evaluate_round_on_test_set(settings, tag)
        self._load_active_learning_queue([p for p, _ in batch])
        summary = (f"Active learning, model {self._round_label(settings)}: the {len(batch)} most uncertain of "
                   f"{len(ranked)} photo(s) are loaded in Mode 1 with SAM click to accept on (uncertainty "
                   f"{batch[0][1]:.2f} -> {batch[-1][1]:.2f}).\n{test_line}\n\nWhen they are labelled: Auto split, "
                   "then the next round on Leonardo (docs/sam_model/active_learning_round_leonardo.sbatch).")
        print(" [ACTIVE LEARNING] " + summary)
        self._notify_active_learning(summary)
        return True

    def _round_label(self, settings):
        path = settings.get("lora_weights", "")
        parent = os.path.basename(os.path.dirname(path))
        return parent if os.path.basename(os.path.dirname(os.path.dirname(path))) == ROUNDS_DIR_NAME \
            else f"round 0 ({os.path.basename(path)})"

    def _active_learning_settings(self):
        """The SAM settings, switched to the newest round's best_lora.pth when there is one."""
        settings = self._usable_sam_settings()
        if not settings:
            return None
        name, path = newest_round_weights(self._sam_model_dir())
        if path and os.path.abspath(os.path.expanduser(settings.get("lora_weights", ""))) != os.path.abspath(path):
            settings = dict(settings, backend="sam_lora", lora_weights=path)
            try:
                self._save_sam_settings(settings)
            except OSError as e:
                print(f" [ACTIVE LEARNING] Could not save the model settings: {e}")
            print(f" [ACTIVE LEARNING] SAM model switched to {name}/best_lora.pth (the new round).")
        return settings

    def _photos_to_label(self, data_dir):
        jsons = os.path.join(data_dir, JSON_DIR_NAME)
        return [p for p in photos_in(os.path.join(data_dir, IMAGES_DIR_NAME))
                if not os.path.isfile(os.path.join(jsons, os.path.splitext(os.path.basename(p))[0] + ".json"))]

    def _compute_maps_here(self, settings, photos, maps):
        """The missing crack maps, computed by the model here, one photo after the other ([Esc] stops)."""
        previous = self.cfg.CURRENT_IMAGE_PATH
        start = time.time()
        try:
            for i, photo in enumerate(photos, 1):
                if self._sam_cancel_requested():
                    print(" [ACTIVE LEARNING] Stopped: the maps computed so far are kept.")
                    return False
                print(f" [ACTIVE LEARNING] Crack map {i}/{len(photos)}: {os.path.basename(photo)} "
                      f"({(time.time() - start) / 60:.1f} min so far)")
                self.cfg.CURRENT_IMAGE_PATH = photo
                prob = self._sam_run_model_on_photo_path(settings, photo)
                if prob is None:
                    return False
                from sam_autoseg import write_cached_probability
                write_cached_probability(maps[photo], prob)
        finally:
            self.cfg.CURRENT_IMAGE_PATH = previous
        return True

    def _sam_run_model_on_photo_path(self, settings, photo):
        img = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        cfg = self.cfg
        saved = (cfg.H_img, cfg.W_img)
        cfg.H_img, cfg.W_img = img.shape[:2]
        try:
            return self._sam_run_model_on_photo(settings)
        finally:
            cfg.H_img, cfg.W_img = saved

    def _rank_by_uncertainty(self, photos, maps):
        ranked = []
        for p in photos:
            prob = cv2.imread(maps[p], cv2.IMREAD_GRAYSCALE) if os.path.isfile(maps[p]) else None
            if prob is not None:
                ranked.append((p, probability_uncertainty(prob)))
        ranked.sort(key=lambda r: (-r[1], os.path.basename(r[0])))
        return ranked

    def _rounds_dir(self):
        return os.path.join(self._sam_model_dir(), ROUNDS_DIR_NAME)

    def _write_round_report(self, tag, ranked, batch):
        os.makedirs(self._rounds_dir(), exist_ok=True)
        path = os.path.join(self._rounds_dir(), f"ranking_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "photo", "uncertainty", "proposed"])
            for i, (p, u) in enumerate(ranked, 1):
                w.writerow([i, os.path.basename(p), u, "yes" if i <= batch else ""])

    def _evaluate_round_on_test_set(self, settings, tag):
        """IoU / F1 of this model on the labelled Test set photos that have its crack map; appended to
        history.csv once per model. Returns a line for the summary."""
        data_dir = str(self.cfg.SCRIPT_DIR)
        test_dir = os.path.join(data_dir, self.cfg.DATASET_SPLIT_DIRS["test"], IMAGES_DIR_NAME)
        tp = fp = fn = n = 0
        for photo in photos_in(test_dir):
            gt = ground_truth_mask(data_dir, photo)
            prob = read_cached_probability(probability_cache_path(self._sam_model_dir(), tag, photo), photo,
                                           gt.shape if gt is not None else None) if gt is not None else None
            if prob is None:
                continue
            a, b, c = iou_f1(prob, gt, self.cfg.SAM_THRESHOLD, uncertain_mask(data_dir, photo))
            tp, fp, fn, n = tp + a, fp + b, fn + c, n + 1
        if not n:
            return ("Test set: no labelled photo with this model's crack map yet, so no score (the Leonardo job "
                    "computes the Test set maps).")
        iou = tp / max(1, tp + fp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        history = os.path.join(self._rounds_dir(), HISTORY_NAME)
        rows = []
        if os.path.isfile(history):
            with open(history, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        previous = next((r for r in reversed(rows) if r["tag"] != tag), None)
        if not any(r["tag"] == tag for r in rows):
            rows.append({"date": datetime.date.today().isoformat(), "round": self._round_label(settings), "tag": tag,
                         "test_photos": n, "iou": f"{iou:.4f}", "f1": f"{f1:.4f}"})
            os.makedirs(self._rounds_dir(), exist_ok=True)
            with open(history, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["date", "round", "tag", "test_photos", "iou", "f1"])
                w.writeheader()
                w.writerows(rows)
        line = f"Test set ({n} photo(s)): IoU {iou:.3f}, F1 {f1:.3f}"
        if previous:
            line += (f" -- previous model ({previous['round']}): IoU {float(previous['iou']):.3f}, "
                     f"F1 {float(previous['f1']):.3f} ({'better' if iou > float(previous['iou']) else 'not better'})")
        return line

    def _load_active_learning_queue(self, photos):
        """Mode 1 on exactly these photos, most uncertain first, with SAM click to accept on from the first one."""
        cfg = self.cfg
        cfg.modalita_scelta = "1"
        cfg.image_queue[:] = list(photos)
        cfg.sam_accept_state["sticky"] = True
        if not cfg.queue_idle:
            cfg.navigate_target = 0
            cfg.queue_item_consumed = False
