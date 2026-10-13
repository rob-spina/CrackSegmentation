"""dataset_browser.py
v1.0.9 [Dataset browser] (left sidebar, File menu, code 31), mixed into CrackSegmentation.
Lists every photo of 'Training set', 'Validation set' and 'Test set' with the number of its building group,
and moves photos from one set to another. The GUI shows the list in a full-window panel (like the manual);
this module holds what the panel needs, so it is the same with or without a window.

Moving keeps the layout of the sets: 'Training set' holds photo, JSON and 'Binary files' exports, while for
'Validation set' and 'Test set' only the photo is in the set and its JSON and masks stay in the working
'JSON files' / 'Binary files' folders. So:
  - Training -> Validation/Test: the photo goes to the set, its JSON and masks back to the working folders;
  - Validation/Test -> Training: the photo goes to the set together with its JSON and masks;
  - Validation <-> Test: the photo only.
With "whole building" (the default) every photo of the same building group in the sets moves together, so no
building ends up in two sets. A file is never overwritten: a clash leaves that photo where it was.
"""
import os
import re
import shutil

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME
from training_validation import VALID_IMAGE_EXTENSIONS, binary_exports_of
from cross_validation import photo_group, is_negative_photo


DATASET_BROWSER_CODE = 31
SPLITS = ("training", "validation", "test")
_GROUP_NUMBER = re.compile(r"BLDG(\d+)", re.IGNORECASE)


def group_number(photo):
    """'028' for '..._0349_D__BLDG028.JPG', '-' for a photo without a building group."""
    m = _GROUP_NUMBER.search(photo_group(photo))
    return m.group(1) if m else "-"


class DatasetBrowserMixin:
    """Inventory of the sets and moves between them. Relies on DatasetSplitMixin._split_root() and cfg."""

    # --- where a photo's files live -----------------------------------------------------------

    def _photo_folder(self, split):
        return os.path.join(self._split_root(split), IMAGES_DIR_NAME)

    def _annotation_folders(self, split):
        """(JSON folder, Binary folder) holding the annotations of a photo of that set."""
        base = self._split_root(split) if split == "training" else str(self.cfg.SCRIPT_DIR)
        return os.path.join(base, JSON_DIR_NAME), os.path.join(base, BINARY_DIR_NAME)

    def _annotation_files(self, split, photo):
        """[(folder kind, path)] of the photo's JSON and 'Binary files' exports, where they are now."""
        json_dir, binary_dir = self._annotation_folders(split)
        base = os.path.splitext(photo)[0]
        found = [(JSON_DIR_NAME, os.path.join(json_dir, n)) for n in (f"{base}.json", f"{base}-seg.json")
                 if os.path.isfile(os.path.join(json_dir, n))]
        names = os.listdir(binary_dir) if os.path.isdir(binary_dir) else []
        found += [(BINARY_DIR_NAME, os.path.join(binary_dir, n)) for n in binary_exports_of(base, names)]
        return found

    # --- inventory ----------------------------------------------------------------------

    def dataset_inventory(self):
        """[{'split', 'photo', 'group', 'number', 'drone', 'negative'}] of every photo in the three sets."""
        rows = []
        for split in SPLITS:
            folder = self._photo_folder(split)
            names = sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)) \
                if os.path.isdir(folder) else []
            json_dir = self._annotation_folders(split)[0]
            for photo in names:
                json_name = os.path.splitext(photo)[0] + ".json"
                rows.append({"split": split, "photo": photo, "group": photo_group(photo), "number": group_number(photo),
                             "drone": photo.upper().startswith("DJI_"),
                             "negative": is_negative_photo(json_dir, json_name)})
        return rows

    # --- moves --------------------------------------------------------------------------

    def _expand_to_buildings(self, picked):
        """Every (split, photo) of the sets sharing a building group with a picked photo."""
        groups = {photo_group(photo) for _, photo in picked}
        return sorted({(r["split"], r["photo"]) for r in self.dataset_inventory() if r["group"] in groups} | set(picked))

    def plan_set_move(self, picked, target, whole_building=True):
        """[(split, photo)] that would move to target (photos already in target are left out)."""
        items = self._expand_to_buildings(picked) if whole_building else sorted(set(picked))
        return [(s, p) for s, p in items if s != target]

    def _file_moves(self, split, photo, target):
        """[(src, dest)] for one photo: the photo, plus its annotations when their home folder changes."""
        moves = [(os.path.join(self._photo_folder(split), photo), os.path.join(self._photo_folder(target), photo))]
        if self._annotation_folders(split) != self._annotation_folders(target):
            json_dir, binary_dir = self._annotation_folders(target)
            for kind, path in self._annotation_files(split, photo):
                moves.append((path, os.path.join(json_dir if kind == JSON_DIR_NAME else binary_dir,
                                                 os.path.basename(path))))
        return moves

    def move_photos_between_sets(self, picked, target, whole_building=True):
        """Moves the picked (split, photo) -- and with whole_building every photo of their buildings -- to
        target. Returns (photos moved, [problems]). A photo whose files would overwrite something stays put."""
        moved, problems = 0, []
        for split, photo in self.plan_set_move(picked, target, whole_building):
            moves = self._file_moves(split, photo, target)
            clash = next((dest for _, dest in moves if os.path.exists(dest)), None)
            if clash:
                problems.append(f"{photo}: {os.path.basename(clash)} already exists in the destination")
                continue
            try:
                for src, dest in moves:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(src, dest)
                moved += 1
            except OSError as e:
                problems.append(f"{photo}: {e}")
        return moved, problems

    def _toggle_dataset_browser(self):
        """Code 31. Without a window: prints the sets with the building numbers. The GUI shows its panel."""
        names = self.cfg.DATASET_SPLIT_DIRS
        rows = self.dataset_inventory()
        for split in SPLITS:
            mine = [r for r in rows if r["split"] == split]
            print(f"\n{names[split]} ({len(mine)} photo(s))")
            for r in mine:
                print(f"   BLDG {r['number']:>4}   {r['photo']}")
