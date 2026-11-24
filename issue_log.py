"""issue_log.py
v2.4.0 [Issue log] (left sidebar "Dataset" section, code 164) and [Show segmentation issues] (menu bar
"Issues", code 165), mixed into CrackSegmentation.

While the issue log is on, every save ([S], Save+next) in Mode 1 or Mode 2 asks which problems the segmentation
of the photo had -- masonry joints traced as cracks, edges / cornices / roads traced, rubble, crazing, missing
cracks ... -- and writes them to '<data folder>/segmentation_issues.csv', one row per photo and category:

    n, immagine, categoria, problema, modalità

  - n: progressive number of the row (kept when the row is updated);
  - immagine: the photo file name;
  - categoria / problema: a category of ISSUE_CATEGORIES with its standard description, plus the note typed in
    the window (zone, what was done), or "X" (other) with a free description;
  - modalità: 1 or 2, the mode the photo was saved in.

Saving the same photo again shows its categories already ticked: the photo's rows are replaced by the new choice
(untick everything to take the photo off the list). "Skip" leaves the list as it is. The table can be shown over
the image from the menu bar (Issues > Show segmentation issues), like the user manual.
"""
import csv
import os

ISSUE_LOG_CODE = 164
ISSUE_LOG_VIEW_CODE = 165
ISSUE_LOG_FILE = "segmentation_issues.csv"
ISSUE_FIELDS = ("n", "immagine", "categoria", "problema", "modalità")
OTHER_CATEGORY = "X"
# Category letter -> standard description (the same wording as the triage of the labelled masks).
ISSUE_CATEGORIES = (
    ("A", "fughe di muratura tracciate come fessure"),
    ("B", "bordi, spigoli, cornici, strade o striature tracciati"),
    ("C", "macerie o crollo tracciati: zona da ignorare"),
    ("D", "cavillatura tracciata come fessure"),
    ("E", "fessure vere non tracciate"),
    ("F", "fessure tracciate male (percorso, larghezza o interruzioni)"),
)
CATEGORY_TEXT = dict(ISSUE_CATEGORIES)


def issue_log_path(data_dir):
    return os.path.join(str(data_dir), ISSUE_LOG_FILE)


def read_issues(path):
    """The rows of the issue log (list of dicts with ISSUE_FIELDS), [] when the file does not exist."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = []
        for r in csv.DictReader(fh):
            rows.append({k: (r.get(k) or "").strip() for k in ISSUE_FIELDS})
        return rows


def write_issues(path, rows):
    """Writes the rows sorted by n (UTF-8 with BOM, so Excel shows the accents)."""
    rows = sorted(rows, key=lambda r: _as_int(r["n"]))
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


def photo_issues(rows, photo):
    return [r for r in rows if r["immagine"] == photo]


def set_photo_issues(path, photo, mode, picks):
    """Replaces the rows of one photo with picks = [(category, problem)]. A category already logged for the photo
    keeps its n; new ones get the next numbers. Returns the photo's new rows."""
    rows = read_issues(path)
    old = {r["categoria"]: r for r in photo_issues(rows, photo)}
    kept = [r for r in rows if r["immagine"] != photo]
    next_n = max([_as_int(r["n"]) for r in rows] + [0]) + 1
    new_rows = []
    for category, problem in picks:
        if category in old:
            n = old[category]["n"]
        else:
            n, next_n = str(next_n), next_n + 1
        new_rows.append({"n": n, "immagine": photo, "categoria": category, "problema": problem,
                         "modalità": str(mode)})
    write_issues(path, kept + new_rows)
    return new_rows


def delete_issue_rows(path, numbers):
    """Deletes the rows whose n is in numbers. Returns how many were deleted."""
    numbers = {str(n) for n in numbers}
    rows = read_issues(path)
    kept = [r for r in rows if r["n"] not in numbers]
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

    def _ask_segmentation_issues(self, photo, current_rows):
        """[(category, problem)], [] = no problem (the photo leaves the list), None = skip.
        A GUI wrapper overrides this with a window; this terminal version asks the letters."""
        for letter, text in ISSUE_CATEGORIES:
            print(f"   {letter}  {text}")
        print(f"   {OTHER_CATEGORY}  other (describe it)")
        now = ",".join(r["categoria"] for r in current_rows)
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
        return picks

    def log_segmentation_issues(self):
        """Called after every successful save: asks and writes the problems when the issue log is on."""
        photo_path = getattr(self.cfg, "CURRENT_IMAGE_PATH", None)
        if not self.issue_log_on() or not photo_path:
            return None
        photo = os.path.basename(photo_path)
        path = self.issue_log_file()
        try:
            current = photo_issues(read_issues(path), photo)
            picks = self._ask_segmentation_issues(photo, current)
            if picks is None:
                print(f" [ISSUE LOG] {photo}: skipped, list unchanged.")
                return None
            rows = set_photo_issues(path, photo, getattr(self.cfg, "modalita_scelta", "") or "", picks)
        except (OSError, csv.Error) as e:
            print(f" [ISSUE LOG] Could not update {path}: {e}")
            return None
        if rows:
            print(f" [ISSUE LOG] {photo}: " + ", ".join(r["categoria"] for r in rows) + f" -> {path}")
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
        print(f"\n{'n':>4}  {'mod.':<4} cat  immagine / problema   ({path})")
        for r in rows:
            print(f"{r['n']:>4}  {r['modalità']:<4} {r['categoria']:<4} {r['immagine']}\n{'':>15}{r['problema']}")
