"""review_flags.py
v2.0.1 [To be reviewed] (left sidebar "Dataset" section, Tools menu, code 140), mixed into CrackSegmentation.
Marks the photo on screen as doubtful -- a crack traced in a doubtful way, a doubt about the whole photo -- so it
can be found again later:
  - the button toggles the mark of the photo on screen; an optional note says what the doubt is;
  - the marks live in '<data folder>/to_be_reviewed.json' ({photo name: {"note", "since"}}), so the photos keep
    their names (the JSON, the binary masks, the building tag __BLDGxxx and the SAM crack-map cache all find the
    photo by its name); the photo's JSON also carries them, in its LabelMe "flags"
    ({"to_be_reviewed": true, "review_note": ...}), so they travel with the dataset;
  - while the photo on screen is marked the status bar shows "TO BE REVIEWED" and the button stays pressed;
  - Mode 2 (at start-up and with [Switch Mode]): when some of the segmented photos are marked and some are not,
    a dialog asks which ones to load: only those to be reviewed, only the others, or all of them. The choice is
    kept for the reloads of the same pass (e.g. after [Validation]).
"""
import datetime
import json
import os
import re

REVIEW_CODE = 140
REVIEW_INDEX_NAME = "to_be_reviewed.json"
REVIEW_FLAG = "to_be_reviewed"
REVIEW_NOTE_FLAG = "review_note"
# Mode-2 queue filters.
REVIEW_FILTER_ONLY = "review"       # only the photos to be reviewed
REVIEW_FILTER_OTHERS = "reviewed"   # only the labelled photos NOT to be reviewed
REVIEW_FILTER_ALL = "all"
REVIEW_FILTERS = (REVIEW_FILTER_ONLY, REVIEW_FILTER_OTHERS, REVIEW_FILTER_ALL)

_BUILDING_TAG = re.compile(r"__[^_]+$")


def review_key(photo_path):
    """The name a photo is marked under: its base name without extension, '-seg' and building tag (__BLDG001),
    lower case -- so a mark survives [B] / the building-group rename and an extension in capitals."""
    base = os.path.splitext(os.path.basename(str(photo_path)))[0]
    if base.endswith("-seg"):
        base = base[:-4]
    return _BUILDING_TAG.sub("", base).lower()


class ReviewFlagsMixin:

    # --- The index ------------------------------------------------------------------------------

    def _review_index_path(self):
        return os.path.join(str(self.cfg.SCRIPT_DIR), REVIEW_INDEX_NAME)

    def _load_review_index(self):
        try:
            with open(self._review_index_path(), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_review_index(self, index):
        path = self._review_index_path()
        try:
            if not index:
                if os.path.exists(path):
                    os.remove(path)
                return True
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, path)
            return True
        except OSError as e:
            print(f"[TO BE REVIEWED] Cannot write {path}: {e}")
            return False

    def is_photo_to_be_reviewed(self, photo_path=None):
        path = photo_path if photo_path is not None else self.cfg.CURRENT_IMAGE_PATH
        return bool(path) and review_key(path) in self._review_index_cache()

    def _review_index_cache(self):
        """The index, re-read only when the file changed (the HUD asks every frame)."""
        path = self._review_index_path()
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            stamp = None
        cache = self.cfg.review_state
        if cache.get("stamp", "unset") != stamp or cache.get("path") != path:
            cache["index"] = self._load_review_index() if stamp is not None else {}
            cache["stamp"], cache["path"] = stamp, path
        return cache["index"]

    def _forget_review_cache(self):
        self.cfg.review_state["stamp"] = "unset"

    def review_note(self, photo_path=None):
        path = photo_path if photo_path is not None else self.cfg.CURRENT_IMAGE_PATH
        entry = self._review_index_cache().get(review_key(path)) if path else None
        return (entry or {}).get("note", "")

    # --- The button -----------------------------------------------------------------------------

    def _prompt_review_note(self, current_note):
        """Optional note on the doubt. Default: terminal input(); the GUI asks with a dialog. None = cancelled."""
        try:
            return input("[TO BE REVIEWED] Note on the doubt (optional, ENTER to skip): ").strip()
        except Exception:
            return ""

    def _toggle_review_flag(self):
        """Marks / unmarks the photo on screen."""
        path = self.cfg.CURRENT_IMAGE_PATH
        if not path:
            return
        index = dict(self._load_review_index())
        key = review_key(path)
        if key in index:
            del index[key]
            marked, note = False, ""
            message = "no longer marked as to be reviewed"
        else:
            note = self._prompt_review_note("")
            if note is None:
                return
            index[key] = {"note": note, "since": datetime.date.today().isoformat()}
            marked = True
            message = "marked as TO BE REVIEWED" + (f": {note}" if note else "")
        if not self._save_review_index(index):
            return
        self._forget_review_cache()
        self._write_review_flags_to_json(marked, note)
        print(f"[TO BE REVIEWED] {os.path.basename(path)} {message}")
        self.refresh_zoom_viewport()

    # --- The photo's JSON -----------------------------------------------------------------------

    def review_flags_for_json(self, photo_path=None):
        """The LabelMe "flags" of the photo: {"to_be_reviewed": True, "review_note": ...} when marked, else {}."""
        path = photo_path if photo_path is not None else self.cfg.CURRENT_IMAGE_PATH
        if not path or not self.is_photo_to_be_reviewed(path):
            return {}
        flags = {REVIEW_FLAG: True}
        note = self.review_note(path)
        if note:
            flags[REVIEW_NOTE_FLAG] = note
        return flags

    def _write_review_flags_to_json(self, marked, note):
        """Puts the mark into the photo's saved JSON at once (the photo may not be saved again before quitting)."""
        json_path = getattr(self.cfg, "JSON_OUTPUT_PATH", None)
        if not json_path or not os.path.exists(json_path):
            return
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            flags = data.get("flags") if isinstance(data.get("flags"), dict) else {}
            flags.pop(REVIEW_FLAG, None)
            flags.pop(REVIEW_NOTE_FLAG, None)
            if marked:
                flags[REVIEW_FLAG] = True
                if note:
                    flags[REVIEW_NOTE_FLAG] = note
            data["flags"] = flags
            tmp = json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, json_path)
        except (OSError, ValueError) as e:
            print(f"[TO BE REVIEWED] Cannot update the flags of {json_path}: {e}")

    def _import_review_flags_from_jsons(self):
        """Photos marked in their JSON flags but missing from the index (e.g. a JSON copied from another computer)
        are added to the index. Only the start of each JSON is read: LabelMe writes "flags" before "imageData"."""
        folder = self.cfg.OUTPUT_FOLDER
        if not folder or not os.path.isdir(folder):
            return
        index = self._load_review_index()
        added = 0
        for name in os.listdir(folder):
            if not name.lower().endswith(".json") or review_key(name) in index:
                continue
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    head = f.read(4096)
            except OSError:
                continue
            if f'"{REVIEW_FLAG}": true' not in head:
                continue
            note = ""
            m = re.search(r'"%s":\s*"((?:[^"\\]|\\.)*)"' % REVIEW_NOTE_FLAG, head)
            if m:
                try:
                    note = json.loads(f'"{m.group(1)}"')
                except ValueError:
                    note = ""
            index[review_key(name)] = {"note": note, "since": ""}
            added += 1
        if added and self._save_review_index(index):
            self._forget_review_cache()
            print(f"[TO BE REVIEWED] {added} photo(s) marked in their JSON added to {REVIEW_INDEX_NAME}")

    # --- Mode 2 queue ---------------------------------------------------------------------------

    def _prompt_review_filter(self, n_review, n_others):
        """Which segmented photos to load. Default: terminal input(); the GUI asks with a dialog."""
        print(f"\n[MODE 2] {n_review} photo(s) TO BE REVIEWED, {n_others} other labelled photo(s).")
        print("  1) only the photos to be reviewed\n  2) only the other labelled photos\n  3) all of them")
        try:
            answer = input("Type 1, 2 or 3 and press ENTER [3]: ").strip()
        except Exception:
            answer = ""
        return {"1": REVIEW_FILTER_ONLY, "2": REVIEW_FILTER_OTHERS}.get(answer, REVIEW_FILTER_ALL)

    def _apply_review_filter(self, ask):
        """Mode 2: keeps in image_queue the photos of the chosen group. With ask, the operator chooses (only when
        both groups have photos); otherwise the choice made earlier in this pass is applied again."""
        if self.cfg.modalita_scelta != "2" or not self.cfg.image_queue:
            return
        if ask:
            self._import_review_flags_from_jsons()
        marked = [p for p in self.cfg.image_queue if self.is_photo_to_be_reviewed(p)]
        n_review, n_others = len(marked), len(self.cfg.image_queue) - len(marked)
        if ask:
            choice = REVIEW_FILTER_ALL
            if n_review and n_others:
                choice = self._prompt_review_filter(n_review, n_others)
            self.cfg.review_state["filter"] = choice if choice in REVIEW_FILTERS else REVIEW_FILTER_ALL
        choice = self.cfg.review_state.get("filter", REVIEW_FILTER_ALL)
        if choice == REVIEW_FILTER_ONLY:
            kept = marked
        elif choice == REVIEW_FILTER_OTHERS:
            kept = [p for p in self.cfg.image_queue if p not in set(marked)]
        else:
            return
        if not kept:   # e.g. the last photo to be reviewed was unmarked: show them all rather than nothing
            self.cfg.review_state["filter"] = REVIEW_FILTER_ALL
            return
        self.cfg.image_queue[:] = kept
        label = "to be reviewed" if choice == REVIEW_FILTER_ONLY else "not to be reviewed"
        print(f"[MODE 2] Loading only the {len(kept)} photo(s) {label}.")

    # --- HUD --------------------------------------------------------------------------------------

    def _draw_review_label(self, win_out):
        if self.cfg.queue_idle or not self.is_photo_to_be_reviewed():
            return
        note = self.review_note()
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 9 * (45 if large else 30)
        self._put_hud_text(win_out, "TO BE REVIEWED" + (f": {note}" if note else ""), (15, y), (60, 60, 255))
