"""training_validation.py
v1.0.6 [Validation]: picks the segmented photos fit for training and moves them, with their JSON and their
'Binary files' exports, into 'Suitable for training'. Mixed into CrackSegmentation (smart_segmentation.py).

A photo qualifies on the per-crack scores saved in its JSON (reliability_pct, confidence, uncertain flag),
against the VALIDATION_* thresholds in config.py. Near-duplicate photos (difference hash) are then reduced
to the best-scoring one, also against the photos already moved by an earlier validation.
"""
import csv
import json
import os
import shutil
import time

import cv2
import numpy as np

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME


VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
# Exports of a photo in 'Binary files', named <photo><suffix><ext>.
BINARY_EXPORT_SUFFIXES = ("-seg", "-crack_mask", "-detachment_mask", "-crack_uncertain_mask")
REPORT_FIELDS = ["photo", "decision", "reason", "cracks", "mean_reliability_pct", "mean_confidence",
                 "uncertain_cracks", "detachments", "similar_to"]


# --- pure helpers -------------------------------------------------------------------------

def _is_crack_shape(shape):
    return shape.get("shape_type") in ("linestrip", "linestring", "line") and "detachment" not in str(shape.get("label", ""))


def _is_uncertain_shape(shape, uncertain_label):
    flags = shape.get("flags") or {}
    return bool(flags.get("uncertain")) or str(shape.get("label", "")).startswith(uncertain_label)


def photo_quality_from_json(json_path, uncertain_label="crack_incerta"):
    """Per-photo summary of the crack scores saved in a LabelMe JSON. None if the JSON cannot be read."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    shapes = data.get("shapes") or []
    cracks = [s for s in shapes if _is_crack_shape(s)]
    rel = [float(s["reliability_pct"]) for s in cracks if s.get("reliability_pct") is not None]
    conf = [float(s["confidence"]) for s in cracks if s.get("confidence") is not None]
    n_uncertain = sum(1 for s in cracks if _is_uncertain_shape(s, uncertain_label))
    return {
        "cracks": len(cracks),
        "detachments": sum(1 for s in shapes if s.get("shape_type") == "polygon"),
        "mean_reliability_pct": float(np.mean(rel)) if rel else None,
        "mean_confidence": float(np.mean(conf)) if conf else None,
        "uncertain_cracks": n_uncertain,
        "uncertain_ratio": (n_uncertain / float(len(cracks))) if cracks else None,
    }


def evaluate_photo_quality(quality, cfg):
    """(suitable, reason) for one photo summary against the VALIDATION_* thresholds of cfg."""
    if quality is None:
        return False, "JSON unreadable"
    if quality["cracks"] == 0:
        return False, "no cracks annotated"
    min_rel, min_conf = cfg.VALIDATION_MIN_MEAN_RELIABILITY_PCT, cfg.VALIDATION_MIN_MEAN_CONFIDENCE
    max_unc = cfg.VALIDATION_MAX_UNCERTAIN_RATIO
    if min_rel is not None:
        if quality["mean_reliability_pct"] is None:
            return False, "no reliability scores (reopen it in Mode 2 and save)"
        if quality["mean_reliability_pct"] < min_rel:
            return False, f"mean reliability {quality['mean_reliability_pct']:.0f}% < {min_rel:.0f}%"
    if min_conf is not None:
        if quality["mean_confidence"] is None:
            return False, "no confidence scores (reopen it in Mode 2 and save)"
        if quality["mean_confidence"] < min_conf:
            return False, f"mean confidence {quality['mean_confidence']:.2f} < {min_conf:.2f}"
    if max_unc is not None and quality["uncertain_ratio"] > max_unc:
        return False, f"uncertain cracks {quality['uncertain_cracks']}/{quality['cracks']} > {100 * max_unc:.0f}%"
    return True, "meets every threshold"


def quality_score(quality):
    """Ranking used to decide which of two near-duplicate photos is kept (higher is better)."""
    rel = quality.get("mean_reliability_pct")
    conf = quality.get("mean_confidence")
    score = (rel / 100.0) if rel is not None else 0.0
    if conf is not None:
        score *= conf
    return score


def read_image_any_path(path, flags=cv2.IMREAD_GRAYSCALE):
    """cv2.imread that also works with non-ASCII paths (Windows). None if unreadable."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def difference_hash(image_path, hash_size=16):
    """256-bit difference hash (bool array): robust to resizing, exposure and small shifts. None if unreadable."""
    gray = read_image_any_path(image_path)
    if gray is None:
        return None
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hash_distance(a, b):
    return int(np.count_nonzero(a != b))


def find_photo_for_json(json_name, images_by_lower_name):
    """Real file name in 'Images' of the photo a JSON belongs to (a legacy '-seg' suffix is dropped). None if missing."""
    base = os.path.splitext(json_name)[0]
    if base.endswith("-seg"):
        base = base[:-4]
    for ext in VALID_IMAGE_EXTENSIONS:
        real = images_by_lower_name.get(f"{base}{ext}".lower())
        if real:
            return real
    return None


def binary_exports_of(photo_base, binary_names):
    """File names in 'Binary files' exported for one photo (overlay and masks)."""
    found = []
    for name in binary_names:
        stem = os.path.splitext(name)[0]
        if any(stem == f"{photo_base}{suffix}" for suffix in BINARY_EXPORT_SUFFIXES):
            found.append(name)
    return sorted(found)


def unique_destination(folder, name):
    """folder/name, or folder/name_1, _2, ... when that file already exists (nothing is ever overwritten)."""
    dest = os.path.join(folder, name)
    stem, ext = os.path.splitext(name)
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(folder, f"{stem}_{n}{ext}")
        n += 1
    return dest


# --- mixin --------------------------------------------------------------------------------

class TrainingValidationMixin:
    """[Validation] button (code 17). Relies on CrackSegmentation's cfg and folder helpers."""

    def _validation_root(self):
        return os.path.join(str(self.cfg.SCRIPT_DIR), self.cfg.VALIDATION_DIR_NAME)

    def _validation_available(self):
        """Only once the queue is finished: the current photo is the last one, so Next can no longer move on."""
        queue = self.cfg.image_queue
        if not queue or self.cfg.queue_idle:
            return True
        current = self.cfg.CURRENT_IMAGE_PATH
        if current in queue:
            return queue.index(current) >= len(queue) - 1
        # Mode 1 [S] already took the photo out of the (sorted) queue: last when nothing sorts after it.
        return bool(current) and not any(p > current for p in queue)

    def _collect_validation_candidates(self):
        """Every JSON in 'JSON files' whose photo is in 'Images', with its quality summary."""
        images_dir = os.path.join(str(self.cfg.SCRIPT_DIR), IMAGES_DIR_NAME)
        json_dir = os.path.join(str(self.cfg.SCRIPT_DIR), JSON_DIR_NAME)
        if not os.path.isdir(images_dir) or not os.path.isdir(json_dir):
            return []
        images_by_lower = {f.lower(): f for f in os.listdir(images_dir)}
        candidates = []
        for json_name in sorted(os.listdir(json_dir)):
            if not json_name.lower().endswith(".json"):
                continue
            photo = find_photo_for_json(json_name, images_by_lower)
            if photo is None:
                continue
            json_path = os.path.join(json_dir, json_name)
            candidates.append({
                "photo": photo,
                "base": os.path.splitext(photo)[0],
                "json_name": json_name,
                "image_path": os.path.join(images_dir, photo),
                "json_path": json_path,
                "quality": photo_quality_from_json(json_path, self.cfg.CRACK_UNCERTAIN_LABEL),
            })
        return candidates

    def _existing_training_hashes(self):
        """(photo name, hash) of the photos already in 'Suitable for training/Images'."""
        folder = os.path.join(self._validation_root(), IMAGES_DIR_NAME)
        if not os.path.isdir(folder):
            return []
        kept = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                h = difference_hash(os.path.join(folder, name))
                if h is not None:
                    kept.append((name, h))
        return kept

    def _drop_near_duplicates(self, suitable, plan):
        """Keeps the best-scoring photo of each group of too-similar ones (photos already moved always win)."""
        max_dist = self.cfg.VALIDATION_DUPLICATE_MAX_HASH_DISTANCE
        kept_hashes = self._existing_training_hashes()
        for cand in sorted(suitable, key=lambda c: quality_score(c["quality"]), reverse=True):
            h = difference_hash(cand["image_path"])
            if h is None:
                plan["rejected"].append((cand, "photo unreadable"))
                continue
            twin = None
            if max_dist is not None:
                twin = next((name for name, other in kept_hashes if hash_distance(h, other) <= max_dist), None)
            if twin is not None:
                plan["duplicates"].append((cand, twin))
                continue
            kept_hashes.append((cand["photo"], h))
            plan["selected"].append(cand)

    def plan_training_validation(self):
        """{'selected', 'rejected': [(cand, reason)], 'duplicates': [(cand, kept photo)]} -- nothing is moved yet."""
        plan = {"selected": [], "rejected": [], "duplicates": []}
        suitable = []
        for cand in self._collect_validation_candidates():
            ok, reason = evaluate_photo_quality(cand["quality"], self.cfg)
            cand["reason"] = reason
            if ok:
                suitable.append(cand)
            else:
                plan["rejected"].append((cand, reason))
        self._drop_near_duplicates(suitable, plan)
        plan["selected"].sort(key=lambda c: c["photo"].lower())
        return plan

    def _validation_summary_message(self, plan):
        n_sel, n_rej, n_dup = len(plan["selected"]), len(plan["rejected"]), len(plan["duplicates"])
        cfg = self.cfg
        lines = [
            f"Suitable for training: {n_sel} photo(s)",
            f"Below the quality thresholds: {n_rej}",
            f"Discarded as too similar to another photo: {n_dup}",
            "",
            f"Thresholds: mean reliability >= {cfg.VALIDATION_MIN_MEAN_RELIABILITY_PCT}%, "
            f"mean confidence >= {cfg.VALIDATION_MIN_MEAN_CONFIDENCE}, "
            f"uncertain cracks <= {cfg.VALIDATION_MAX_UNCERTAIN_RATIO}.",
            "Only saved annotations are evaluated: save the current photo first if you changed it.",
        ]
        if n_sel:
            lines += ["", f"Move the {n_sel} photo(s), their JSON and their masks/overlays into "
                          f"'{cfg.VALIDATION_DIR_NAME}'?"]
        return "\n".join(lines)

    def _confirm_training_validation(self, message):
        """Asks before moving anything. A GUI wrapper overrides this with a dialog."""
        print("\n" + message)
        try:
            return input("Type y to move the files, or ENTER to cancel: ").strip().lower() == "y"
        except Exception:
            return False

    def _notify_training_validation_result(self, message):
        """Final summary. A GUI wrapper overrides this with a dialog."""
        print(message)

    def _move_into(self, src, dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
        dest = unique_destination(dest_folder, os.path.basename(src))
        shutil.move(src, dest)
        return dest

    def _move_validated_photo(self, cand, binary_names):
        """Moves the photo, its JSON and its 'Binary files' exports; returns how many files were moved."""
        root = self._validation_root()
        binary_dir = os.path.join(str(self.cfg.SCRIPT_DIR), BINARY_DIR_NAME)
        moved = 0
        self._move_into(cand["image_path"], os.path.join(root, IMAGES_DIR_NAME))
        self._move_into(cand["json_path"], os.path.join(root, JSON_DIR_NAME))
        moved += 2
        for name in binary_exports_of(cand["base"], binary_names):
            self._move_into(os.path.join(binary_dir, name), os.path.join(root, BINARY_DIR_NAME))
            moved += 1
        return moved

    def _report_row(self, cand, decision, reason, similar_to=""):
        q = cand["quality"] or {}
        rel, conf = q.get("mean_reliability_pct"), q.get("mean_confidence")
        return {
            "photo": cand["photo"], "decision": decision, "reason": reason,
            "cracks": q.get("cracks", ""),
            "mean_reliability_pct": "" if rel is None else round(rel, 1),
            "mean_confidence": "" if conf is None else round(conf, 3),
            "uncertain_cracks": q.get("uncertain_cracks", ""),
            "detachments": q.get("detachments", ""),
            "similar_to": similar_to,
        }

    def _write_validation_report(self, plan):
        """validation_report_<timestamp>.csv in 'Suitable for training', one row per evaluated photo."""
        root = self._validation_root()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"validation_report_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        rows = [self._report_row(c, "suitable", c.get("reason", "")) for c in plan["selected"]]
        rows += [self._report_row(c, "below thresholds", r) for c, r in plan["rejected"]]
        rows += [self._report_row(c, "too similar", "near-duplicate photo", twin) for c, twin in plan["duplicates"]]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _print_validation_plan(self, plan):
        for cand in plan["selected"]:
            print(f"  [OK] {cand['photo']}")
        for cand, reason in plan["rejected"]:
            print(f"  [--] {cand['photo']}: {reason}")
        for cand, twin in plan["duplicates"]:
            print(f"  [==] {cand['photo']}: too similar to {twin}")

    def execute_training_validation(self, plan):
        """Moves the selected photos and writes the CSV report. Returns (photos moved, report path)."""
        binary_dir = os.path.join(str(self.cfg.SCRIPT_DIR), BINARY_DIR_NAME)
        binary_names = os.listdir(binary_dir) if os.path.isdir(binary_dir) else []
        moved = 0
        for cand in plan["selected"]:
            try:
                self._move_validated_photo(cand, binary_names)
                moved += 1
            except OSError as e:
                print(f"[VALIDATION ERROR] Could not move {cand['photo']}: {e}")
        report = self._write_validation_report(plan)
        return moved, report

    def run_training_validation(self):
        """Code 17 [Validation]. Returns True when photos were moved (the image queue must then be reloaded)."""
        if not self._validation_available():
            print(" [VALIDATION] Available only on the last image, once Next can no longer move on.")
            return False
        print("\n[VALIDATION] Evaluating the segmented photos...")
        plan = self.plan_training_validation()
        self._print_validation_plan(plan)
        if not plan["selected"]:
            self._notify_training_validation_result(
                self._validation_summary_message(plan) + "\n\nNo photo to move.")
            return False
        if not self._confirm_training_validation(self._validation_summary_message(plan)):
            print("[VALIDATION] Cancelled: nothing was moved.")
            return False
        moved, report = self.execute_training_validation(plan)
        self._notify_training_validation_result(
            f"{moved} photo(s) moved to '{self._validation_root()}'.\nReport: {report}")
        return moved > 0
