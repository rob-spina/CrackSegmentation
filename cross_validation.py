"""cross_validation.py
v1.0.8 [CV folds] (sidebar, code 26), mixed into CrackSegmentation (smart_segmentation.py).
Splits the photos of 'Training set' into K folds for cross-validation, by building group: every photo of a
group (__BLDGnnn) lands in the same fold, so no building is seen in training and validation of the same
round. Groups are dealt largest first to the fold with the fewest photos, which keeps the folds even.
The result is 'Training set/folds.csv', one row per photo with its fold and the paths of its JSON and masks
(relative to 'Training set'), and 'Training set/folds/fold_1 .. fold_K', each a COPY of its photos with their
JSON and 'Binary files' exports (Images / JSON files / Binary files), ready for a training script that wants
one folder per fold. Pressed again, both are replaced; 'Training set' itself is never changed.
A photo saved without any crack (JSON with no crack shape, hence no crack mask) is a NEGATIVE example:
its whole label is background. folds.csv marks it with negative = 1, so the training script uses it.
"""
import csv
import os
import re
import shutil

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME
from training_validation import VALID_IMAGE_EXTENSIONS, binary_exports_of, photo_quality_from_json


CV_FOLDS_CODE = 26
FOLDS_FIELDS = ["photo", "group", "fold", "image", "json", "crack_mask", "uncertain_mask", "ignore_mask", "negative"]
_GROUP_TAG = re.compile(r"__(BLDG\d+)", re.IGNORECASE)


def photo_group(photo):
    """Building group of a photo ('BLDG028'); a photo without a group is a group of its own."""
    m = _GROUP_TAG.search(photo)
    return m.group(1).upper() if m else "single:" + os.path.splitext(photo)[0]


def assign_group_folds(photos, k):
    """{photo: fold (1..k)} keeping each group in one fold. Groups go largest first (then by name) to the
    fold holding the fewest photos so far (lowest number on a tie), so the result is reproducible."""
    groups = {}
    for photo in photos:
        groups.setdefault(photo_group(photo), []).append(photo)
    sizes = [0] * k
    folds = {}
    for group in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        target = min(range(k), key=lambda i: (sizes[i], i))
        sizes[target] += len(groups[group])
        for photo in groups[group]:
            folds[photo] = target + 1
    return folds


def _relative_if_exists(root, *parts):
    return "/".join(parts) if os.path.isfile(os.path.join(root, *parts)) else ""


def is_negative_photo(root, json_rel, crack_mask_rel):
    """True for a photo saved without any crack: a readable JSON with no crack shape and no crack mask."""
    if crack_mask_rel or not json_rel:
        return False
    quality = photo_quality_from_json(os.path.join(root, json_rel))
    return quality is not None and quality["cracks"] == 0


def fold_rows(root, folds):
    """One folds.csv row per photo, with the paths of its annotation files ('' when missing)."""
    rows = []
    for photo in sorted(folds, key=lambda p: (folds[p], p)):
        base = os.path.splitext(photo)[0]
        rows.append(_fold_row(root, photo, base, folds[photo]))
    return rows


def _fold_row(root, photo, base, fold):
    row = {
        "photo": photo, "group": photo_group(photo), "fold": fold,
        "image": f"{IMAGES_DIR_NAME}/{photo}",
        "json": _relative_if_exists(root, JSON_DIR_NAME, f"{base}.json"),
        "crack_mask": _relative_if_exists(root, BINARY_DIR_NAME, f"{base}-crack_mask.png"),
        "uncertain_mask": _relative_if_exists(root, BINARY_DIR_NAME, f"{base}-crack_uncertain_mask.png"),
        "ignore_mask": _relative_if_exists(root, BINARY_DIR_NAME, f"{base}-ignore_mask.png"),
    }
    row["negative"] = 1 if is_negative_photo(root, row["json"], row["crack_mask"]) else 0
    return row


def folds_summary(rows, k):
    """Text lines: photos and groups per fold, and the photos left without a crack mask."""
    lines = []
    for fold in range(1, k + 1):
        in_fold = [r for r in rows if r["fold"] == fold]
        groups = {r["group"] for r in in_fold}
        drone = sum(1 for r in in_fold if r["photo"].upper().startswith("DJI_"))
        negatives = sum(1 for r in in_fold if r["negative"])
        lines.append(f"Fold {fold}: {len(in_fold)} photo(s), {len(groups)} building(s), {drone} from drone"
                     + (f", {negatives} without cracks" if negatives else ""))
    missing = [r["photo"] for r in rows if not r["crack_mask"] and not r["negative"]]
    if missing:
        lines.append(f"{len(missing)} photo(s) without a crack mask (they cannot be trained on): "
                     + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
    return lines


class CrossValidationMixin:
    """[CV folds] button. Relies on CrackSegmentation's cfg and DatasetSplitMixin._split_root()."""

    def _training_set_photos(self):
        folder = os.path.join(self._split_root("training"), IMAGES_DIR_NAME)
        if not os.path.isdir(folder):
            return []
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS))

    # --- dialogs (a GUI wrapper overrides all three) ------------------------------------------

    def _ask_cv_fold_count(self, default):
        """Number of folds, or None if cancelled."""
        try:
            text = input(f"Number of cross-validation folds [{default}]: ").strip()
        except EOFError:
            return None
        try:
            return int(text) if text else default
        except ValueError:
            return None

    def _confirm_cv_folds(self, message):
        print("\n" + message)
        try:
            return input("Type y to write folds.csv, or ENTER to cancel: ").strip().lower() in ("y", "s")
        except EOFError:
            return False

    def _notify_cv_folds(self, message):
        print(message)

    # --- the button ---------------------------------------------------------------------

    def _cv_fold_count_problem(self, k, photos):
        """Why k folds cannot be made from these photos, or None."""
        n_groups = len({photo_group(p) for p in photos})
        if not photos:
            return f"'{self.cfg.DATASET_SPLIT_DIRS['training']}' holds no photo yet: move photos there first."
        if k < 2:
            return "At least 2 folds are needed."
        if n_groups < k:
            return f"Only {n_groups} building group(s) in the training set: at most {n_groups} folds."
        return None

    def _write_folds_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FOLDS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _cv_folds_message(self, path, rows, k):
        n_groups = len({r["group"] for r in rows})
        folds_dir = os.path.join(os.path.dirname(path), self.cfg.CV_FOLDS_DIR)
        head = [f"{len(rows)} photo(s) of {n_groups} building(s) into {k} folds (each building in one fold):", ""]
        action = (f"Write {os.path.basename(path)} and copy the photos, JSON and masks into "
                  f"'{self.cfg.CV_FOLDS_DIR}/fold_1' .. 'fold_{k}'?")
        if os.path.exists(path) or os.path.isdir(folds_dir):
            action += f"\nThe existing {os.path.basename(path)} and '{self.cfg.CV_FOLDS_DIR}' folder are replaced."
        return "\n".join(head + folds_summary(rows, k) + ["", action])

    def _copy_fold_folders(self, root, rows):
        """Fresh 'folds/fold_k/{Images, JSON files, Binary files}' copies. Returns how many files were copied."""
        folds_dir = os.path.join(root, self.cfg.CV_FOLDS_DIR)
        if os.path.isdir(folds_dir):
            shutil.rmtree(folds_dir)   # only ever the folder this button generated
        binary_dir = os.path.join(root, BINARY_DIR_NAME)
        binary_names = os.listdir(binary_dir) if os.path.isdir(binary_dir) else []
        copied = 0
        for row in rows:
            dest = os.path.join(folds_dir, f"fold_{row['fold']}")
            base = os.path.splitext(row["photo"])[0]
            sources = [(row["image"], IMAGES_DIR_NAME)]
            if row["json"]:
                sources.append((row["json"], JSON_DIR_NAME))
            sources += [(f"{BINARY_DIR_NAME}/{n}", BINARY_DIR_NAME) for n in binary_exports_of(base, binary_names)]
            for rel, sub in sources:
                os.makedirs(os.path.join(dest, sub), exist_ok=True)
                shutil.copy2(os.path.join(root, rel), os.path.join(dest, sub))
                copied += 1
        return copied

    def make_cv_folds(self):
        """Code 26 [CV folds]: asks K, shows the split, writes 'Training set/folds.csv'. Returns its path or None."""
        photos = self._training_set_photos()
        k = self._ask_cv_fold_count(self.cfg.CV_FOLDS)
        if k is None:
            return None
        problem = self._cv_fold_count_problem(k, photos)
        if problem:
            self._notify_cv_folds(problem)
            return None
        root = self._split_root("training")
        rows = fold_rows(root, assign_group_folds(photos, k))
        path = os.path.join(root, self.cfg.CV_FOLDS_FILE)
        if not self._confirm_cv_folds(self._cv_folds_message(path, rows, k)):
            print(" [CV FOLDS] Cancelled: nothing written.")
            return None
        self._write_folds_and_copies(root, path, rows, k)
        return path

    def _write_folds_and_copies(self, root, path, rows, k):
        self._write_folds_csv(path, rows)
        try:
            copied = self._copy_fold_folders(root, rows)
        except OSError as e:
            self._notify_cv_folds(f"{os.path.basename(path)} written, but copying the fold folders failed: {e}")
            return
        folds_dir = os.path.join(root, self.cfg.CV_FOLDS_DIR)
        print(f" [CV FOLDS] {len(rows)} photo(s) in {k} folds: {path}; {copied} file(s) copied into {folds_dir}")
        self._notify_cv_folds(f"{k} fold folders ready in:\n{folds_dir}\n\n"
                              f"{copied} file(s) copied (photos, JSON, masks); folds.csv lists the same split.")
