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
    kept for the reloads of the same pass (e.g. after [Validation]);
  - [Validation] and [Auto split] never move a marked photo (it stays in 'Images' until the mark is removed;
    Auto split then puts it in the set of its building group); [Training / Test / Validation set] ask first.

[To be reviewed by date] (code 160): marks at once every segmented photo whose JSON was last saved on or before a
date (e.g. today: everything labelled so far, to go over it again with a new SAM model). The photos already marked
keep their mark and note; "Remove" takes away only the marks given this way (manual marks stay). The JSON files keep
their saving date (the mark is written in their flags without touching it), so the same date can be used again.
Mode 2 loads only the photos of 'Images': photos already in Training / Validation / Test set are marked too, and
[Reset pipeline training] brings them back.
"""
import datetime
import json
import os
import re

from config import JSON_DIR_NAME, IMAGES_DIR_NAME

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

REVIEW_CODE = 140
REVIEW_BY_DATE_CODE = 160   # v2.0.1 [To be reviewed by date]: every photo whose JSON was saved up to a date
REVIEW_BATCH_KEY = "until"  # index entries of a by-date mark: {"note", "since", "until": "YYYY-MM-DD"}
REVIEW_SETS = ("Training set", "Validation set", "Test set")
REVIEW_INDEX_NAME = "to_be_reviewed.json"
REVIEW_FLAG = "to_be_reviewed"
REVIEW_REJECT_REASON = "marked TO BE REVIEWED"   # Validation / Auto split leave these photos in 'Images'
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
        folder = os.path.join(str(self.cfg.SCRIPT_DIR), JSON_DIR_NAME)
        if not os.path.isdir(folder):
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

    # --- To be reviewed by date ----------------------------------------------------------------------

    @staticmethod
    def parse_review_date(text):
        """'2026-11-08', '08/11/2026', '8-11-2026', '08.11.2026' -> date; '' -> today; None if not a date."""
        text = (text or "").strip()
        if not text:
            return datetime.date.today()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
            try:
                return datetime.datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        return None

    def _segmented_json_folders(self):
        """(folder, set name or None): the JSON folder of the data folder and of each set."""
        root = str(self.cfg.SCRIPT_DIR)
        sets = getattr(self.cfg, "DATASET_SPLIT_DIRS", None) or {}
        names = list(dict.fromkeys(list(sets.values()) + list(REVIEW_SETS)))
        folders = [(os.path.join(root, JSON_DIR_NAME), None)]
        folders += [(os.path.join(root, name, JSON_DIR_NAME), name) for name in names]
        return [(f, name) for f, name in folders if os.path.isdir(f)]

    def _photo_location(self, key):
        """'images' when the photo is in the data folder's 'Images', the set's name when it is in a set's, else None."""
        root = str(self.cfg.SCRIPT_DIR)
        sets = getattr(self.cfg, "DATASET_SPLIT_DIRS", None) or {}
        names = list(dict.fromkeys(list(sets.values()) + list(REVIEW_SETS)))
        cache = self.cfg.review_state.setdefault("_photo_names", {})
        for label, folder in [("images", os.path.join(root, IMAGES_DIR_NAME))] + \
                [(n, os.path.join(root, n, IMAGES_DIR_NAME)) for n in names]:
            if folder not in cache:
                try:
                    cache[folder] = {review_key(f) for f in os.listdir(folder) if f.lower().endswith(PHOTO_EXTENSIONS)}
                except OSError:
                    cache[folder] = set()
            if key in cache[folder]:
                return label
        return None

    def review_by_date_plan(self, until):
        """The segmented photos whose JSON was last saved on or before `until` (a date):
        {"jsons": {key: [json paths]}, "marked": keys already marked, "images": keys whose photo is in 'Images',
         "sets": {set name: [keys]}, "missing": keys without their photo}."""
        limit = datetime.datetime.combine(until, datetime.time.max).timestamp()
        jsons = {}
        for folder, _ in self._segmented_json_folders():
            for name in os.listdir(folder):
                if not name.lower().endswith(".json") or name.startswith("."):
                    continue
                path = os.path.join(folder, name)
                try:
                    if os.path.getmtime(path) > limit:
                        continue
                except OSError:
                    continue
                jsons.setdefault(review_key(name), []).append(path)
        self.cfg.review_state.pop("_photo_names", None)
        index = self._load_review_index()
        plan = {"jsons": jsons, "marked": sorted(k for k in jsons if k in index), "images": [], "sets": {},
                "missing": [], "by_date": sorted(k for k in jsons if REVIEW_BATCH_KEY in (index.get(k) or {}))}
        for key in sorted(jsons):
            where = self._photo_location(key)
            if where == "images":
                plan["images"].append(key)
            elif where:
                plan["sets"].setdefault(where, []).append(key)
            else:
                plan["missing"].append(key)
        self.cfg.review_state.pop("_photo_names", None)
        return plan

    @staticmethod
    def _set_json_review_flags(json_path, marked, note):
        """Writes / removes the mark in a JSON's flags, keeping the file's saving date (so the date filter still
        sees when the photo was really labelled)."""
        try:
            stat = os.stat(json_path)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            flags = data.get("flags") if isinstance(data.get("flags"), dict) else {}
            had = flags.get(REVIEW_FLAG) is True
            if had == marked and (not marked or flags.get(REVIEW_NOTE_FLAG, "") == (note or "")):
                return True
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
            os.utime(json_path, (stat.st_atime, stat.st_mtime))
            return True
        except (OSError, ValueError) as e:
            print(f"[TO BE REVIEWED] Cannot update the flags of {json_path}: {e}")
            return False

    def _prompt_review_by_date(self):
        """{"date": date, "note": str, "remove": bool} or None. Default: terminal input(); the GUI has a dialog."""
        try:
            text = input("[TO BE REVIEWED BY DATE] Mark the photos saved up to (YYYY-MM-DD, ENTER = today): ")
            note = input("Note (ENTER = 're-segment with the new SAM model'): ").strip()
        except Exception:
            return None
        date = self.parse_review_date(text)
        if date is None:
            print(f"[TO BE REVIEWED BY DATE] '{text}' is not a date.")
            return None
        return {"date": date, "note": note or "re-segment with the new SAM model", "remove": False}

    def _confirm_review_by_date(self, message):
        """True = go ahead. Default: terminal input(); the GUI asks with a dialog."""
        print("\n" + message)
        try:
            return input("Type y to go ahead, or ENTER to cancel: ").strip().lower() in ("y", "s")
        except Exception:
            return False

    def _notify_review_by_date(self, message):
        print(message)

    @staticmethod
    def _review_by_date_message(plan, until, remove):
        n = len(plan["jsons"])
        day = until.isoformat()
        if remove:
            return (f"{len(plan['by_date'])} of the {n} photo(s) saved up to {day} were marked TO BE REVIEWED by "
                    f"date: remove their mark?\n(Marks given one photo at a time stay.)")
        new = n - len(plan["marked"])
        lines = [f"{n} segmented photo(s) were saved up to {day}: {new} will be marked TO BE REVIEWED"
                 + (f" ({len(plan['marked'])} are already marked and keep their note)." if plan["marked"] else ".")]
        lines.append(f"- {len(plan['images'])} in 'Images': Mode 2 (Switch Mode) can load only these.")
        for name, keys in sorted(plan["sets"].items()):
            lines.append(f"- {len(keys)} in '{name}': [Reset pipeline training] brings them back to 'Images'.")
        if plan["missing"]:
            lines.append(f"- {len(plan['missing'])} JSON(s) without their photo (marked anyway).")
        lines.append("The JSON files keep their saving date.")
        return "\n".join(lines)

    def mark_review_by_date(self):
        """[To be reviewed by date]: marks (or unmarks) every photo whose JSON was saved up to a date."""
        answer = self._prompt_review_by_date()
        if not answer:
            print("[TO BE REVIEWED BY DATE] Cancelled.")
            return False
        until, note, remove = answer["date"], (answer.get("note") or "").strip(), bool(answer.get("remove"))
        if until > datetime.date.today():
            until = datetime.date.today()
        plan = self.review_by_date_plan(until)
        targets = plan["by_date"] if remove else [k for k in plan["jsons"] if k not in plan["marked"]]
        if not targets:
            self._notify_review_by_date(
                f"To be reviewed by date: no photo saved up to {until.isoformat()} "
                + ("was marked by date." if remove else
                   ("to mark (all already marked)." if plan["jsons"] else "(no segmented photo found)."))
            )
            return False
        if not self._confirm_review_by_date(self._review_by_date_message(plan, until, remove)):
            print("[TO BE REVIEWED BY DATE] Cancelled.")
            return False
        index = dict(self._load_review_index())
        today = datetime.date.today().isoformat()
        for key in targets:
            if remove:
                index.pop(key, None)
            else:
                index[key] = {"note": note, "since": today, REVIEW_BATCH_KEY: until.isoformat()}
        if not self._save_review_index(index):
            return False
        self._forget_review_cache()
        for key in targets:
            for path in plan["jsons"][key]:
                self._set_json_review_flags(path, not remove, note)
        verb = "no longer marked" if remove else "marked TO BE REVIEWED"
        self._notify_review_by_date(
            f"To be reviewed by date: {len(targets)} photo(s) saved up to {until.isoformat()} {verb}."
            + ("" if remove else "\nIn Mode 2 (Switch Mode) choose 'Only the photos to be reviewed'; press "
               "[To be reviewed] on a photo to take its mark away once it is done."))
        print(f"[TO BE REVIEWED BY DATE] {len(targets)} photo(s) {verb} (saved up to {until.isoformat()}).")
        if getattr(self.cfg, "CURRENT_IMAGE_PATH", None) and not getattr(self.cfg, "queue_idle", False):
            try:
                self.refresh_zoom_viewport()
            except Exception:
                pass
        return True

    def _confirm_move_photo_to_be_reviewed(self, message):
        """[Training / Test / Validation set] on a marked photo. Default: terminal input(); the GUI asks."""
        print("\n" + message)
        try:
            return input("Type y to move it anyway, or ENTER to cancel: ").strip().lower() in ("y", "s")
        except Exception:
            return False

    def _review_move_message(self, set_name):
        note = self.review_note()
        return (f"This photo is marked TO BE REVIEWED" + (f" ({note})" if note else "") + ".\n"
                f"Move it to '{set_name}' anyway? It stays marked; press [To be reviewed] first to remove the mark.")

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
            elif n_review:   # every segmented photo is to be reviewed: no question, but start from the first
                choice = REVIEW_FILTER_ONLY
                print(f"[MODE 2] All {n_review} segmented photo(s) are marked TO BE REVIEWED.")
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
        """Red frame round the photo and a red banner on it, plus the status-bar line."""
        if self.cfg.queue_idle or not self.is_photo_to_be_reviewed():
            return
        note = self.review_note()
        text = "TO BE REVIEWED" + (f": {note}" if note else "")
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 9 * (45 if large else 30)
        self._put_hud_text(win_out, text, (15, y), (60, 60, 255))
        if win_out is None or not hasattr(win_out, "shape"):
            return
        import cv2
        h, w = win_out.shape[:2]
        top = 0
        visible_top = getattr(self, "_visible_canvas_top", None)
        if visible_top is not None:
            try:
                top = int(visible_top())
            except Exception:
                top = 0
        red = (40, 40, 230)
        cv2.rectangle(win_out, (2, 2), (w - 3, h - 3), red, 6)
        label = text if len(text) <= 70 else text[:67] + "..."
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0 = max(10, w - tw - 40)
        y0 = top + 14
        cv2.rectangle(win_out, (x0 - 12, y0), (x0 + tw + 12, y0 + th + 18), red, -1)
        cv2.putText(win_out, label, (x0, y0 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
