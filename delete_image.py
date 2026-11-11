"""delete_image.py
v2.0.1 [Delete image] (left sidebar "Dataset" section, File menu, code 149), mixed into CrackSegmentation.
Takes an unsuitable photo (blurred, no building, a duplicate ...) out of the dataset it was loaded from:
the photo, its JSON and its 'Binary files' exports (overlay and masks) are MOVED to '<data folder>/Deleted images'
(same sub-folders), never erased, so a mistake can be undone by moving them back. Asks for confirmation; its
[To be reviewed] mark is removed; the next photo of the queue opens.
"""
import os

from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME
from training_validation import binary_exports_of

DELETE_IMAGE_CODE = 149
DELETED_DIR_NAME = "Deleted images"


class DeleteImageMixin:

    def _confirm_delete_image(self, message):
        """True = delete. Default: terminal input(); the GUI asks with a dialog."""
        print("\n" + message)
        try:
            return input("Type y to delete it, or ENTER to keep it: ").strip().lower() in ("y", "s")
        except Exception:
            return False

    def _deleted_root(self):
        return os.path.join(str(self.cfg.SCRIPT_DIR), DELETED_DIR_NAME)

    def run_delete_image(self):
        """Code 149. True once the photo left the dataset: the image loop then opens the next one."""
        cfg = self.cfg
        photo = cfg.CURRENT_IMAGE_PATH
        if getattr(cfg, "queue_idle", False) or not photo or not os.path.exists(photo):
            print(" [DELETE IMAGE] No photo on screen.")
            return False
        name = os.path.basename(photo)
        source = os.path.basename(os.path.dirname(photo))
        json_path = cfg.JSON_OUTPUT_PATH if cfg.JSON_OUTPUT_PATH and os.path.exists(cfg.JSON_OUTPUT_PATH) else None
        binary_dir = os.path.join(os.path.dirname(os.path.dirname(photo)), BINARY_DIR_NAME)
        base = os.path.splitext(name)[0]
        exports = binary_exports_of(base, os.listdir(binary_dir)) if os.path.isdir(binary_dir) else []
        parts = ["the photo"] + (["its JSON"] if json_path else []) + ([f"{len(exports)} mask/overlay file(s)"]
                                                                       if exports else [])
        message = (f"Delete {name} from '{source}'?\n\n"
                   f"{', '.join(parts)} will be moved to '{DELETED_DIR_NAME}' in the data folder "
                   "(not erased: move them back to restore the photo).")
        if not self._confirm_delete_image(message):
            print(" [DELETE IMAGE] Cancelled: the photo stays.")
            return False
        root = self._deleted_root()
        try:
            moved = [self._move_into(photo, os.path.join(root, IMAGES_DIR_NAME))]
            if json_path:
                moved.append(self._move_into(json_path, os.path.join(root, JSON_DIR_NAME)))
            for f in exports:
                moved.append(self._move_into(os.path.join(binary_dir, f), os.path.join(root, BINARY_DIR_NAME)))
        except OSError as e:
            print(f" [DELETE IMAGE] Could not move {name}: {e}")
            return False
        self._forget_deleted_photo(photo)
        self._leave_photo_moved_to_split()   # out of the queue; the next photo takes its place
        print(f" [DELETE IMAGE] {name} removed from '{source}': {len(moved)} file(s) moved to '{DELETED_DIR_NAME}'.")
        return True

    def _forget_deleted_photo(self, photo):
        """Its [To be reviewed] mark goes; SAM click to accept closes on it (and reopens on the next photo)."""
        from review_flags import review_key
        index = self._load_review_index()
        if review_key(photo) in index:
            del index[review_key(photo)]
            self._save_review_index(index)
            self._forget_review_cache()
        state = self.cfg.sam_accept_state
        if state.get("active"):
            sticky = state.get("sticky")
            state["unsaved"] = False
            self._exit_sam_accept("photo deleted" + (" -- it reopens on the next photo" if sticky else ""))
            state["sticky"] = sticky
        self.cfg.action_history = []
