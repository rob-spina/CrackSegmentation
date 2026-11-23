"""building_filter.py
v2.3.0 [Filter by building] (left sidebar "Dataset" section, Tools menu, code 163), mixed into CrackSegmentation.
Loads only the photos of one building group (the __BLDGnnn tag in the photo name) from 'Images':
  - a window lists the groups found in 'Images', each with its labelled / to-label counts, and three choices:
    labelled photos only (opened as in Mode 2, their JSON loaded), photos still to label only (Mode 1), or all the
    photos of the group (default): each photo then opens in the mode it needs -- Mode 2 when it has a JSON, Mode 1
    otherwise -- and a photo saved in Mode 1 stays in the queue, so the whole building can be walked back and forth;
  - the status bar shows "BUILDING BLDGnnn" while the filter is on; [Switch Mode], [Validation] and the other
    commands that rebuild the queue turn it off.
Photos without a tag belong to no group; photos already moved to Training / Validation / Test set are not in
'Images' and are not listed (as for Mode 1 / Mode 2).
"""
import os
import re

from config import IMAGES_DIR_NAME, JSON_DIR_NAME

BUILDING_FILTER_CODE = 163
BUILDING_FILTER_LABELLED = "labelled"
BUILDING_FILTER_TO_LABEL = "to_label"
BUILDING_FILTER_ALL = "all"
BUILDING_FILTERS = (BUILDING_FILTER_LABELLED, BUILDING_FILTER_TO_LABEL, BUILDING_FILTER_ALL)
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_TAG = re.compile(r"__(BLDG\d+)", re.IGNORECASE)


def building_tag(name):
    """'BLDG024' for 'abc__BLDG024.jpg', None without a tag."""
    m = _TAG.search(os.path.basename(str(name)))
    return m.group(1).upper() if m else None


def _json_bases(json_dir):
    """Lower-case base names with a JSON (a legacy '-seg' suffix dropped)."""
    if not os.path.isdir(json_dir):
        return set()
    bases = set()
    for f in os.listdir(json_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() == ".json":
            bases.add((base[:-4] if base.endswith("-seg") else base).lower())
    return bases


def building_groups_in(data_dir):
    """{tag: {"labelled": [paths], "to_label": [paths]}} of the tagged photos of '<data_dir>/Images', sorted."""
    images = os.path.join(str(data_dir), IMAGES_DIR_NAME)
    done = _json_bases(os.path.join(str(data_dir), JSON_DIR_NAME))
    groups = {}
    if not os.path.isdir(images):
        return groups
    for f in sorted(os.listdir(images)):
        base, ext = os.path.splitext(f)
        tag = building_tag(f)
        if ext.lower() not in PHOTO_EXTENSIONS or f.startswith(".") or base.endswith("-seg") or not tag:
            continue
        kind = "labelled" if base.lower() in done else "to_label"
        groups.setdefault(tag, {"labelled": [], "to_label": []})[kind].append(os.path.join(images, f))
    return dict(sorted(groups.items(), key=lambda kv: (int(re.sub(r"\D", "", kv[0]) or 0), kv[0])))


def filtered_photos(groups, tag, which):
    """The photos of one group for a choice, by name."""
    g = groups.get(tag, {"labelled": [], "to_label": []})
    if which == BUILDING_FILTER_LABELLED:
        return sorted(g["labelled"])
    if which == BUILDING_FILTER_TO_LABEL:
        return sorted(g["to_label"])
    return sorted(g["labelled"] + g["to_label"])


class BuildingFilterMixin:
    """[Filter by building]. Relies on CrackSegmentation's cfg, queue bookkeeping and _settle_unsaved_changes."""

    def _prompt_building_filter(self, groups):
        """(tag, which) or None. A GUI wrapper overrides this with a window; this terminal version asks a number."""
        tags = list(groups)
        for i, t in enumerate(tags, 1):
            print(f"  {i}. {t}  ({len(groups[t]['labelled'])} labelled, {len(groups[t]['to_label'])} to label)")
        try:
            answer = input(f"Building group 1-{len(tags)} [ENTER = cancel]: ").strip()
            if not (answer.isdigit() and 1 <= int(answer) <= len(tags)):
                return None
            kind = input("1 = labelled, 2 = to label, 3 = all [ENTER = all]: ").strip()
        except EOFError:
            return None
        which = {"1": BUILDING_FILTER_LABELLED, "2": BUILDING_FILTER_TO_LABEL}.get(kind, BUILDING_FILTER_ALL)
        return tags[int(answer) - 1], which

    def _notify_building_filter(self, message):
        print(f" [FILTER BY BUILDING] {message}")

    def run_building_filter(self):
        """Code 163. True when a new queue with the photos of the chosen group was loaded."""
        cfg = self.cfg
        if not cfg.queue_idle and cfg.CURRENT_IMAGE_PATH and not self._settle_unsaved_changes("filter by building"):
            return False
        groups = building_groups_in(cfg.SCRIPT_DIR)
        if not groups:
            self._notify_building_filter("No photo of 'Images' has a building tag (__BLDGnnn): use [Assign building] "
                                         "or [Group similar photos] first.")
            return False
        choice = self._prompt_building_filter(groups)
        if not choice:
            return False
        tag, which = choice
        photos = filtered_photos(groups, tag, which if which in BUILDING_FILTERS else BUILDING_FILTER_ALL)
        if not photos:
            kind = "labelled" if which == BUILDING_FILTER_LABELLED else "still to label"
            self._notify_building_filter(f"{tag} has no photo {kind} in 'Images'.")
            return False
        self._load_building_queue(tag, which, photos)
        print(f" [FILTER BY BUILDING] {tag}: {len(photos)} photo(s) ({which}) loaded.")
        return True

    def _load_building_queue(self, tag, which, photos):
        cfg = self.cfg
        cfg.building_filter = {"tag": tag, "which": which, "mixed": which == BUILDING_FILTER_ALL}
        cfg.modalita_scelta = "2" if which == BUILDING_FILTER_LABELLED else "1"
        if which == BUILDING_FILTER_ALL:
            cfg.modalita_scelta = self._mode_for_photo(photos[0])
        cfg.image_queue[:] = list(photos)
        cfg.queue_items_done = 0
        if not cfg.queue_idle:
            cfg.navigate_target = 0
            cfg.queue_item_consumed = False

    def _mode_for_photo(self, photo):
        """'2' when the photo has a JSON in 'JSON files', '1' otherwise."""
        base = os.path.splitext(os.path.basename(photo))[0].lower()
        return "2" if base in _json_bases(os.path.join(str(self.cfg.SCRIPT_DIR), JSON_DIR_NAME)) else "1"

    def building_filter_mixed(self):
        return bool(getattr(self.cfg, "building_filter", {}).get("mixed"))

    def _apply_building_filter_mode(self, photo):
        """Mixed queue (all the photos of a group): the mode each photo needs, set before it opens."""
        if self.building_filter_mixed():
            self.cfg.modalita_scelta = self._mode_for_photo(photo)

    def _clear_building_filter(self):
        self.cfg.building_filter = {}

    def _draw_building_filter_label(self, win_out):
        f = getattr(self.cfg, "building_filter", {})
        if self.cfg.queue_idle or not f.get("tag"):
            return
        kinds = {BUILDING_FILTER_LABELLED: "labelled", BUILDING_FILTER_TO_LABEL: "to label",
                 BUILDING_FILTER_ALL: "all"}
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 10 * (45 if large else 30)
        self._put_hud_text(win_out, f"BUILDING {f['tag']} ({kinds.get(f.get('which'), 'all')}, "
                                    f"{len(self.cfg.image_queue)} photo(s))", (15, y), (255, 200, 80))
