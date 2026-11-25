"""issue_log.py
v2.4.0 [Issue log] (left sidebar "Dataset" section, code 164) and [Show segmentation issues] (menu bar
"Issues", code 165), mixed into CrackSegmentation.

While the issue log is on, every save ([S], Save+next) in Mode 1 or Mode 2 asks which problems the segmentation
of the photo had -- masonry joints traced as cracks, edges / cornices / roads traced, rubble, crazing, missing
cracks ... -- and writes them to '<data folder>/segmentation_issues.csv', one row per photo and category:

    N, image, category, problem, mode, destination

  - N: progressive number of the row (kept when the row is updated);
  - image: the photo file name;
  - category / problem: a category of ISSUE_CATEGORIES with its standard description, plus the note typed in
    the window (zone, what was done), or "X" (other) with a free description;
  - mode: 1 or 2, the mode the photo was saved in;
  - destination: the set the photo is meant for -- Training set, Validation set, Test set -- or "none" when
    the split is not decided yet (v2.4.0). It is only written down: the photo is not moved. The window proposes
    the set already chosen for the photo, else the set that already holds photos of the same building
    (__BLDGnnn), so a building does not end up in two sets.

Saving the same photo again shows its categories already ticked: the photo's rows are replaced by the new choice
(untick everything to take the photo off the list). "Skip" leaves the list as it is. The table can be shown over
the image from the menu bar (Issues > Show segmentation issues), like the user manual.
"""
import csv
import os
import re

ISSUE_LOG_CODE = 164
ISSUE_LOG_VIEW_CODE = 165
ISSUE_LOG_FILE = "segmentation_issues.csv"
ISSUE_FIELDS = ("N", "image", "category", "problem", "mode", "destination")
# Column names of the first 2.4.0 builds (Italian): still read, written back in English.
LEGACY_FIELDS = {"n": "N", "immagine": "image", "categoria": "category", "problema": "problem", "modalità": "mode",
                 "destinazione": "destination"}
NO_DESTINATION = "none"
# Destination keys (as in cfg.DATASET_SPLIT_DIRS) in the order shown in the window; None = split not decided yet.
DESTINATION_KEYS = ("training", "validation", "test", None)
OTHER_CATEGORY = "X"
# Category letter -> standard description (English since 2.4.0; the Italian wording of the first builds is translated when read).
ISSUE_CATEGORIES = (
    ("A", "masonry joints traced as cracks"),
    ("B", "edges, corners, cornices, roads or streaks traced"),
    ("C", "rubble or collapse traced: area to ignore"),
    ("D", "crazing traced as cracks"),
    ("E", "real cracks not traced"),
    ("F", "cracks traced badly (path, width or gaps)"),
)
CATEGORY_TEXT = dict(ISSUE_CATEGORIES)
# Standard descriptions of the first 2.4.0 builds (Italian) -> English, applied when a file is read.
LEGACY_TEXTS = (
    ("fughe di muratura tracciate come fessure", "masonry joints traced as cracks"),
    ("bordi, spigoli, cornici, strade o striature tracciati", "edges, corners, cornices, roads or streaks traced"),
    ("macerie o crollo tracciati: zona da ignorare", "rubble or collapse traced: area to ignore"),
    ("cavillatura tracciata come fessure", "crazing traced as cracks"),
    ("fessure vere non tracciate", "real cracks not traced"),
    ("fessure tracciate male (percorso, larghezza o interruzioni)", "cracks traced badly (path, width or gaps)"),
)


def issue_log_path(data_dir):
    return os.path.join(str(data_dir), ISSUE_LOG_FILE)


def read_issues(path):
    """The rows of the issue log (list of dicts with ISSUE_FIELDS), [] when the file does not exist."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = []
        for r in csv.DictReader(fh):
            r = {LEGACY_FIELDS.get(k, k): v for k, v in r.items() if k is not None}
            if (r.get("destination") or "").strip() == "nessuna":   # first 2.4.0 builds
                r["destination"] = NO_DESTINATION
            problem = (r.get("problem") or "").strip()
            for old, new in LEGACY_TEXTS:
                if problem.startswith(old):
                    r["problem"] = new + problem[len(old):]
                    break
            rows.append({k: (r.get(k) or "").strip() for k in ISSUE_FIELDS})
        return rows


def write_issues(path, rows):
    """Writes the rows sorted by n (UTF-8 with BOM, so Excel shows the accents)."""
    rows = sorted(rows, key=lambda r: _as_int(r["N"]))
    path = os.path.realpath(path)   # a link to the list elsewhere (e.g. the Documents data folder) stays a link
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=ISSUE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ISSUE_FIELDS})
    os.replace(tmp, path)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def problem_text(category, note="", other=""):
    """'<standard description> -- <note>'; for category X the free description (plus the note)."""
    base = other.strip() if category == OTHER_CATEGORY else CATEGORY_TEXT.get(category, "")
    note = note.strip()
    if base and note:
        return f"{base} -- {note}"
    return base or note


def destination_label(key, split_dirs):
    """'Test set' for 'test' (names from cfg.DATASET_SPLIT_DIRS), 'none' for None."""
    return split_dirs.get(key, NO_DESTINATION) if key else NO_DESTINATION


def destination_key(label, split_dirs):
    """The inverse of destination_label: None for 'none' / empty / unknown."""
    for key, name in split_dirs.items():
        if label == name:
            return key
    return None


def building_sets(data_dir, split_dirs, tag):
    """The set keys whose 'Images' already holds a photo of the building tag (e.g. 'BLDG024')."""
    if not tag:
        return []
    pattern = re.compile(rf"__{re.escape(tag)}(?!\d)", re.IGNORECASE)
    found = []
    for key, name in split_dirs.items():
        folder = os.path.join(str(data_dir), name, "Images")
        if os.path.isdir(folder) and any(pattern.search(f) for f in os.listdir(folder)):
            found.append(key)
    return found


def photo_issues(rows, photo):
    return [r for r in rows if r["image"] == photo]


def set_photo_issues(path, photo, mode, picks, destination=NO_DESTINATION):
    """Replaces the rows of one photo with picks = [(category, problem)], all with the same destination (set name
    or 'none'). A category already logged for the photo keeps its n; new ones get the next numbers.
    Returns the photo's new rows."""
    rows = read_issues(path)
    old = {r["category"]: r for r in photo_issues(rows, photo)}
    kept = [r for r in rows if r["image"] != photo]
    next_n = max([_as_int(r["N"]) for r in rows] + [0]) + 1
    new_rows = []
    for category, problem in picks:
        if category in old:
            n = old[category]["N"]
        else:
            n, next_n = str(next_n), next_n + 1
        new_rows.append({"N": n, "image": photo, "category": category, "problem": problem,
                         "mode": str(mode), "destination": destination or NO_DESTINATION})
    write_issues(path, kept + new_rows)
    return new_rows


def delete_issue_rows(path, numbers):
    """Deletes the rows whose n is in numbers. Returns how many were deleted."""
    numbers = {str(n) for n in numbers}
    rows = read_issues(path)
    kept = [r for r in rows if r["N"] not in numbers]
    if len(kept) != len(rows):
        write_issues(path, kept)
    return len(rows) - len(kept)


class IssueLogMixin:
    """[Issue log] / [Show segmentation issues]. Relies on CrackSegmentation's cfg."""

    def issue_log_file(self):
        return issue_log_path(self.cfg.SCRIPT_DIR)

    def issue_log_on(self):
        return bool(getattr(self.cfg, "issue_log_on", False))

    def toggle_issue_log(self):
        """Code 164: switches the question at every save on / off (for this session)."""
        self.cfg.issue_log_on = not self.issue_log_on()
        if self.cfg.issue_log_on:
            print(f" [ISSUE LOG] On: every save asks the problems of the segmentation -> {self.issue_log_file()}")
        else:
            print(" [ISSUE LOG] Off: saves no longer ask anything.")
        return self.cfg.issue_log_on

    def suggested_destination(self, photo, current_rows):
        """(key or None, hint): the set already written for the photo, else the set holding its building."""
        split_dirs = self.cfg.DATASET_SPLIT_DIRS
        for r in current_rows:
            if r.get("destination"):
                return destination_key(r["destination"], split_dirs), ""
        from building_filter import building_tag
        tag = building_tag(photo)
        found = building_sets(self.cfg.SCRIPT_DIR, split_dirs, tag)
        if not found:
            return None, ""
        names = ", ".join(split_dirs[k] for k in found)
        hint = f"{tag} already has photos in {names}" + (": choose the same set." if len(found) == 1 else
                                                          " -- the building is already split, check it.")
        return found[0], hint

    def _ask_segmentation_issues(self, photo, current_rows):
        """([(category, problem)], destination key or None) -- [] = no problem (the photo leaves the list) --
        or None = skip. A GUI wrapper overrides this with a window; this terminal version asks the letters."""
        for letter, text in ISSUE_CATEGORIES:
            print(f"   {letter}  {text}")
        print(f"   {OTHER_CATEGORY}  other (describe it)")
        now = ",".join(r["category"] for r in current_rows)
        try:
            answer = input(f" [ISSUE LOG] {photo}: categories (e.g. A,C; '-' = none; ENTER = skip)"
                           + (f" [now {now}]" if now else "") + ": ").strip().upper()
        except EOFError:
            return None
        if not answer:
            return None
        if answer == "-":
            return []
        letters = [a.strip() for a in answer.replace(";", ",").split(",") if a.strip()]
        valid = [a for a in letters if a in CATEGORY_TEXT or a == OTHER_CATEGORY]
        other = ""
        if OTHER_CATEGORY in valid:
            try:
                other = input(" [ISSUE LOG] Describe the other problem: ")
            except EOFError:
                other = ""
        try:
            note = input(" [ISSUE LOG] Note (zone, what was done; ENTER = none): ")
        except EOFError:
            note = ""
        picks = []
        for a in dict.fromkeys(valid):
            text = problem_text(a, note, other)
            if text:
                picks.append((a, text))
        suggested, hint = self.suggested_destination(photo, current_rows)
        if hint:
            print(f" [ISSUE LOG] {hint}")
        names = self.cfg.DATASET_SPLIT_DIRS
        try:
            answer = input(" [ISSUE LOG] Destination: t = training, v = validation, s = test, n = not decided"
                           f" (ENTER = {destination_label(suggested, names)}): ").strip().lower()
        except EOFError:
            answer = ""
        destination = {"t": "training", "v": "validation", "s": "test", "n": None}.get(answer, suggested)
        return picks, destination

    def log_segmentation_issues(self):
        """Called after every successful save: asks and writes the problems when the issue log is on."""
        photo_path = getattr(self.cfg, "CURRENT_IMAGE_PATH", None)
        if not self.issue_log_on() or not photo_path:
            return None
        photo = os.path.basename(photo_path)
        path = self.issue_log_file()
        try:
            current = photo_issues(read_issues(path), photo)
            answer = self._ask_segmentation_issues(photo, current)
            if answer is None:
                print(f" [ISSUE LOG] {photo}: skipped, list unchanged.")
                return None
            picks, destination = answer
            rows = set_photo_issues(path, photo, getattr(self.cfg, "modalita_scelta", "") or "", picks,
                                    destination_label(destination, self.cfg.DATASET_SPLIT_DIRS))
        except (OSError, csv.Error) as e:
            print(f" [ISSUE LOG] Could not update {path}: {e}")
            return None
        if rows:
            print(f" [ISSUE LOG] {photo}: " + ", ".join(r["category"] for r in rows)
                  + f", destination {rows[0]['destination']} -> {path}")
        else:
            print(f" [ISSUE LOG] {photo}: no problem, not in the list.")
        return rows

    def _toggle_issue_log_view(self):
        """Code 165. Without a window: prints the table. The GUI shows it over the image, like the manual."""
        path = self.issue_log_file()
        rows = read_issues(path)
        if not rows:
            print(f" [ISSUE LOG] No problem logged yet ({path}).")
            return
        print(f"\n{'N':>4}  {'mode':<4} cat  image  [destination] / problem   ({path})")
        for r in rows:
            print(f"{r['N']:>4}  {r['mode']:<4} {r['category']:<4} {r['image']}  [{r['destination']}]\n"
                  f"{'':>15}{r['problem']}")
