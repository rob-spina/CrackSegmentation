"""dataset_split.py
v1.0.8 dataset split buttons, mixed into CrackSegmentation (smart_segmentation.py). Mode 2 only.
  - Training set (sidebar, code 23): moves the photo on screen, its JSON and its 'Binary files' exports
    (masks and -seg overlay) into 'Training set/Images', 'Training set/JSON files', 'Training set/Binary files'.
  - Test set (code 24) / Validation set (code 25): move only the original photo into 'Test set/Images' or
    'Validation set/Images'; its JSON and masks stay where they are.
Folders are created when missing and existing files are never overwritten. The next photo of the queue
is then shown. Folder names are set by DATASET_SPLIT_DIRS in config.py.
"""
import os

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME
from training_validation import binary_exports_of


TRAINING_SET_CODE = 23
TEST_SET_CODE = 24
VALIDATION_SET_CODE = 25
SPLIT_BY_CODE = {TRAINING_SET_CODE: "training", TEST_SET_CODE: "test", VALIDATION_SET_CODE: "validation"}
# Only the training set takes the annotations along with the photo.
SPLIT_MOVES_ANNOTATIONS = {"training": True, "test": False, "validation": False}


class DatasetSplitMixin:
    """Training / Test / Validation set buttons. Relies on CrackSegmentation's cfg, save helpers and
    TrainingValidationMixin._move_into()."""

    def _dataset_split_available(self):
        """Mode 2 with a photo on screen."""
        cfg = self.cfg
        return cfg.modalita_scelta == "2" and not cfg.queue_idle and bool(cfg.CURRENT_IMAGE_PATH)

    def _split_root(self, split):
        return os.path.join(str(self.cfg.SCRIPT_DIR), self.cfg.DATASET_SPLIT_DIRS[split])

    def _ask_save_before_split(self, set_name):
        """True = save first, False = move the files as saved on disk, None = cancel.
        Asked only when the photo has changes not yet saved. A GUI wrapper overrides this with a dialog."""
        try:
            answer = input(f"This photo has unsaved changes. Save them before moving it to '{set_name}'? "
                           "[y = save / n = don't save / ENTER = cancel]: ").strip().lower()
        except EOFError:
            return None
        return {"y": True, "s": True, "n": False}.get(answer)

    def _settle_unsaved_changes(self, set_name):
        """False when the operator cancelled. Unsaved changes are written first if asked to."""
        if not self.cfg.action_history:
            return True
        choice = self._ask_save_before_split(set_name)
        if choice is None:
            return False
        if choice:
            self._save_annotations_now("DATASET SPLIT", None)
        return True

    def _move_photo_to_split(self, split):
        """Moves the photo on screen (plus JSON and binary exports for the training set).
        Returns the list of moved destination paths."""
        cfg = self.cfg
        root = self._split_root(split)
        photo = cfg.CURRENT_IMAGE_PATH
        moved = [self._move_into(photo, os.path.join(root, IMAGES_DIR_NAME))]
        if not SPLIT_MOVES_ANNOTATIONS[split]:
            return moved
        if cfg.JSON_OUTPUT_PATH and os.path.exists(cfg.JSON_OUTPUT_PATH):
            moved.append(self._move_into(cfg.JSON_OUTPUT_PATH, os.path.join(root, JSON_DIR_NAME)))
        binary_dir = os.path.join(str(cfg.SCRIPT_DIR), BINARY_DIR_NAME)
        if os.path.isdir(binary_dir):
            base = os.path.splitext(os.path.basename(photo))[0]
            for name in binary_exports_of(base, os.listdir(binary_dir)):
                moved.append(self._move_into(os.path.join(binary_dir, name), os.path.join(root, BINARY_DIR_NAME)))
        return moved

    def _handle_dataset_split_key(self, code):
        """Codes 23/24/25. True once the photo left 'Images': the image loop then moves on to the next one."""
        split = SPLIT_BY_CODE[code]
        set_name = self.cfg.DATASET_SPLIT_DIRS[split]
        if not self._dataset_split_available():
            print(f" [DATASET SPLIT] '{set_name}' works in Mode 2 only, with a photo on screen.")
            return False
        if SPLIT_MOVES_ANNOTATIONS[split] and not os.path.exists(self.cfg.JSON_OUTPUT_PATH or ""):
            print(f" [DATASET SPLIT] This photo has no saved JSON: save it before moving it to '{set_name}'.")
            return False
        if not self._settle_unsaved_changes(set_name):
            print(" [DATASET SPLIT] Cancelled: nothing moved.")
            return False
        try:
            moved = self._move_photo_to_split(split)
        except OSError as e:
            print(f" [DATASET SPLIT] Could not move {os.path.basename(self.cfg.CURRENT_IMAGE_PATH)}: {e}")
            return False
        self._leave_photo_moved_to_split()
        print(f" [DATASET SPLIT] {os.path.basename(moved[0])} -> '{set_name}' ({len(moved)} file(s) moved).")
        return True

    def _leave_photo_moved_to_split(self):
        """The photo is no longer in 'Images': out of the queue, and the next one takes its place."""
        cfg = self.cfg
        if cfg.CURRENT_IMAGE_PATH in cfg.image_queue:
            cfg.image_queue.remove(cfg.CURRENT_IMAGE_PATH)
            cfg.queue_item_consumed = True
        cfg.translate_state["active"], cfg.translate_state["idx"], cfg.translate_state["start"] = False, None, None
        self._all_segmented_cache = None
