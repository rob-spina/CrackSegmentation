"""auto_split.py
v1.0.8 [Auto split] (sidebar, code 28), mixed into CrackSegmentation (smart_segmentation.py).
Splits EVERY segmented photo of 'Images' (photo with its JSON) into Training / Validation / Test set in one go,
instead of one photo at a time with the Training / Test / Validation set buttons:
  - the percentages are asked (default AUTO_SPLIT_PERCENT in config.py, e.g. 70/15/15; 80/0/20 with
    cross-validation, which needs no validation set);
  - the split is by BUILDING: every photo of a group (__BLDGnnn) goes to the same set; a group that already
    has photos in a set (moved by hand earlier) stays in that set;
  - groups are dealt largest first to the set where they best follow the percentages -- photo count first, then
    the share of drone photos and of photos without cracks, so every set gets a similar mix;
  - the plan (photos, buildings, drone photos and photos without cracks per set) is shown and nothing is
    moved before the confirmation. The files move as with the single buttons: Training set takes the photo,
    its JSON and its 'Binary files' exports; Test and Validation set take the photo only.
Works in either mode, also with no photo on screen.
"""
import os

from config import IMAGES_DIR_NAME, JSON_DIR_NAME
from training_validation import VALID_IMAGE_EXTENSIONS
from cross_validation import photo_group, is_negative_photo


AUTO_SPLIT_CODE = 28
SPLIT_ORDER = ("training", "validation", "test")


def parse_split_percent(text):
    """'70/15/15' (or '70 15 15', '70,15,15') -> (0.70, 0.15, 0.15) for training, validation, test.
    Raises ValueError unless there are three non-negative numbers with a positive sum (normalised to 1)."""
    parts = [p for p in str(text).replace(",", "/").replace(";", "/").replace(" ", "/").split("/") if p.strip()]
    if len(parts) != 3:
        raise ValueError("three numbers are needed: training / validation / test")
    values = [float(p) for p in parts]
    if any(v < 0 for v in values) or sum(values) <= 0 or values[0] <= 0:
        raise ValueError("the numbers must be >= 0, with a positive training share")
    total = sum(values)
    return tuple(v / total for v in values)


def _as_vector(value):
    """A group's size as (photos, drone photos, photos without cracks); a bare int counts photos only."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value, 0, 0)


def assign_groups(groups, fractions, fixed=None, already=None):
    """{group: split}. groups = {group: photos} or {group: (photos, drone photos, photos without cracks)};
    fractions = {split: share}; fixed = {group: split} for groups already in a set; already = {split: photos
    already in it}. Fixed groups keep their set. The others go largest first (then by name) to the set
    where they best follow the shares -- first the photo count, then the drone and crack-free shares, so
    every set gets a similar mix. Sets with a zero share never receive anything. Deterministic."""
    fixed = dict(fixed or {})
    sizes = {g: _as_vector(v) for g, v in groups.items()}
    counts = {s: [0, 0, 0] for s in fractions}
    for s, n in (already or {}).items():
        if s in counts:
            counts[s][0] += n
    totals = [sum(v[i] for v in sizes.values()) + (sum((already or {}).values()) if i == 0 else 0) for i in range(3)]
    plan = {}
    for group, split in fixed.items():
        if group in sizes:
            plan[group] = split
            counts[split] = [c + v for c, v in zip(counts[split], sizes[group])]
    open_splits = [s for s in SPLIT_ORDER if fractions.get(s, 0) > 0]
    weights = (1.0, 0.5, 0.5)    # photo count first, then the drone and crack-free mix

    def cost(split, add):
        total = 0.0
        for s in open_splits:
            for i in range(3):
                value = counts[s][i] + (add[i] if s == split else 0)
                total += weights[i] * ((value - fractions[s] * totals[i]) / max(1.0, totals[i])) ** 2
        return total

    free = sorted((g for g in sizes if g not in plan), key=lambda g: (-sizes[g][0], g))
    for group in free:
        target = min(open_splits, key=lambda s: (cost(s, sizes[group]), SPLIT_ORDER.index(s)))
        plan[group] = target
        counts[target] = [c + v for c, v in zip(counts[target], sizes[group])]
    _refine(plan, free, sizes, counts, open_splits, lambda: cost(None, (0, 0, 0)))
    return plan


def _refine(plan, free, sizes, counts, open_splits, total_cost, max_rounds=50):
    """Local search after the greedy pass: moves a group, or swaps two, between sets while that lowers the
    cost (the greedy order alone can leave one set with most drone or crack-free photos). Fixed groups
    never move. Deterministic: candidates are tried in a fixed order, the first improvement is kept."""
    def shift(group, src, dst):
        counts[src] = [c - v for c, v in zip(counts[src], sizes[group])]
        counts[dst] = [c + v for c, v in zip(counts[dst], sizes[group])]
        plan[group] = dst

    for _ in range(max_rounds):
        best, improved = total_cost(), False
        for g in free:
            for dst in open_splits:
                src = plan[g]
                if dst == src:
                    continue
                shift(g, src, dst)
                if total_cost() < best - 1e-12:
                    best, improved = total_cost(), True
                else:
                    shift(g, dst, src)
        for i, g in enumerate(free):
            for h in free[i + 1:]:
                a, b = plan[g], plan[h]
                if a == b:
                    continue
                shift(g, a, b)
                shift(h, b, a)
                if total_cost() < best - 1e-12:
                    best, improved = total_cost(), True
                else:
                    shift(g, b, a)
                    shift(h, a, b)
        if not improved:
            return


class AutoSplitMixin:
    """[Auto split] button. Relies on DatasetSplitMixin (_split_root, _move_files_to_split) and the queue helpers."""

    # --- what there is to split -------------------------------------------------------------

    def _segmented_photos_in_images(self):
        """(photos with a JSON, photos without one) of 'Images', as file names."""
        images = os.path.join(str(self.cfg.SCRIPT_DIR), IMAGES_DIR_NAME)
        if not os.path.isdir(images):
            return [], []
        with_json, without = [], []
        for name in sorted(os.listdir(images)):
            if name.lower().endswith(VALID_IMAGE_EXTENSIONS) and not name.lower().endswith("-seg.jpg"):
                (with_json if self._photo_json_in_working_folder(name) else without).append(name)
        return with_json, without

    def _photo_json_in_working_folder(self, photo):
        base = os.path.splitext(photo)[0]
        for name in (f"{base}.json", f"{base}-seg.json"):
            path = os.path.join(str(self.cfg.SCRIPT_DIR), JSON_DIR_NAME, name)
            if os.path.isfile(path):
                return path
        return None

    def _groups_already_in_sets(self):
        """({group: split} for the groups that already have photos in a set, {split: photos already in it})."""
        fixed, already = {}, {}
        for split in SPLIT_ORDER:
            folder = os.path.join(self._split_root(split), IMAGES_DIR_NAME)
            names = [f for f in os.listdir(folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)] \
                if os.path.isdir(folder) else []
            already[split] = len(names)
            for name in names:
                fixed.setdefault(photo_group(name), split)
        return fixed, already

    # --- the plan -------------------------------------------------------------------------

    def plan_auto_split(self, fractions):
        """{'assign': {photo: split}, 'skipped': [photos without JSON], 'already': {split: n}, 'fractions': ...}."""
        photos, skipped = self._segmented_photos_in_images()
        by_group = {}
        for photo in photos:
            by_group.setdefault(photo_group(photo), []).append(photo)
        fixed, already = self._groups_already_in_sets()
        sizes = {g: (len(ps), sum(1 for p in ps if p.upper().startswith("DJI_")),
                     sum(1 for p in ps if self._is_negative_in_images(p))) for g, ps in by_group.items()}
        groups = assign_groups(sizes, dict(zip(SPLIT_ORDER, fractions)), fixed, already)
        assign = {photo: groups[g] for g, ps in by_group.items() for photo in ps}
        kept = sorted(g for g in by_group if g in fixed)
        return {"assign": assign, "skipped": skipped, "already": already, "fractions": fractions, "kept": kept}

    def _is_negative_in_images(self, photo):
        json_path = self._photo_json_in_working_folder(photo)
        rel = os.path.relpath(json_path, str(self.cfg.SCRIPT_DIR)) if json_path else ""
        return is_negative_photo(str(self.cfg.SCRIPT_DIR), rel)

    def _auto_split_message(self, plan):
        names = self.cfg.DATASET_SPLIT_DIRS
        pct = "/".join(f"{round(100 * f)}" for f in plan["fractions"])
        lines = [f"Split {len(plan['assign'])} segmented photo(s) of 'Images' by building ({pct}):", ""]
        for split in SPLIT_ORDER:
            photos = [p for p, s in plan["assign"].items() if s == split]
            buildings = len({photo_group(p) for p in photos})
            drone = sum(1 for p in photos if p.upper().startswith("DJI_"))
            negatives = sum(1 for p in photos if self._is_negative_in_images(p))
            extra = f" (+{plan['already'][split]} already there)" if plan["already"].get(split) else ""
            lines.append(f"{names[split]}: {len(photos)} photo(s){extra}, {buildings} building(s), "
                         f"{drone} from drone, {negatives} without cracks")
        if plan["kept"]:
            lines.append(f"\n{len(plan['kept'])} building(s) already in a set stay in that set.")
        if plan["skipped"]:
            lines.append(f"{len(plan['skipped'])} photo(s) without a JSON are not moved (segment them first).")
        lines += ["", "Training set takes photo, JSON and masks; Validation and Test set the photo only.",
                  "Move the photos now?"]
        return "\n".join(lines)

    # --- dialogs (a GUI wrapper overrides them) -----------------------------------------------

    def _ask_auto_split_percent(self, default):
        try:
            return input(f"Training / Validation / Test percentages [{default}]: ").strip() or default
        except EOFError:
            return None

    def _confirm_auto_split(self, message):
        print("\n" + message)
        try:
            return input("Type y to move the photos, or ENTER to cancel: ").strip().lower() in ("y", "s")
        except EOFError:
            return False

    def _notify_auto_split(self, message):
        print(message)

    # --- the button -----------------------------------------------------------------------

    def _ask_auto_split_fractions(self):
        """The three shares, or None when cancelled or not valid (the reason is shown)."""
        text = self._ask_auto_split_percent(self.cfg.AUTO_SPLIT_PERCENT)
        if text is None:
            return None
        try:
            return parse_split_percent(text)
        except ValueError as e:
            self._notify_auto_split(f"'{text}': {e}. Example: 70/15/15, or 80/0/20 with cross-validation.")
            return None

    def _execute_auto_split(self, plan):
        moved, errors = 0, []
        for photo, split in sorted(plan["assign"].items()):
            path = os.path.join(str(self.cfg.SCRIPT_DIR), IMAGES_DIR_NAME, photo)
            try:
                self._move_files_to_split(split, path, self._photo_json_in_working_folder(photo))
                moved += 1
            except OSError as e:
                errors.append(f"{photo}: {e}")
        return moved, errors

    def run_auto_split(self):
        """Code 28 [Auto split]. Returns True when photos were moved (the image queue must then be reloaded)."""
        if not self.cfg.queue_idle and self.cfg.CURRENT_IMAGE_PATH and not self._settle_unsaved_changes("the sets"):
            print(" [AUTO SPLIT] Cancelled: nothing moved.")
            return False
        fractions = self._ask_auto_split_fractions()
        if fractions is None:
            return False
        plan = self.plan_auto_split(fractions)
        if not plan["assign"]:
            self._notify_auto_split("No segmented photo (photo with its JSON) left in 'Images' to split.")
            return False
        if not self._confirm_auto_split(self._auto_split_message(plan)):
            print(" [AUTO SPLIT] Cancelled: nothing moved.")
            return False
        moved, errors = self._execute_auto_split(plan)
        self._all_segmented_cache = None
        summary = f"{moved} photo(s) moved into the sets." + (
            f"\n{len(errors)} could not be moved:\n" + "\n".join(errors[:5]) if errors else "")
        print(f" [AUTO SPLIT] {summary}")
        self._notify_auto_split(summary + "\n\nNext: CV folds (on the Training set), then training.")
        return moved > 0
