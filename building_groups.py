"""building_groups.py
v2.0.1 [Group similar photos] (left sidebar "Building & scale", Tools menu, code 141), mixed into CrackSegmentation,
and a command-line tool for a whole folder (also on Leonardo):

    python3 building_groups.py --data-dir ~/Desktop/CrackSegmentation            # preview only
    python3 building_groups.py --data-dir ~/Desktop/CrackSegmentation --apply    # and rename the photos

Finds the photos of 'Images' that show the same building, so that [Auto split] / [CV folds] keep each building
in one set (no building both in training and in test). Photos downloaded from a dataset often have meaningless
names and no building information; two photos are put in the same group when:
  - they share enough of the same structure: SIFT features + RANSAC homography, with the thresholds of the
    automatic group of Mode 1 (BUILDING_GROUP_MATCH_MIN_GOOD_MATCHES / _MIN_INLIERS / _MIN_INLIER_RATIO), or
  - both carry GPS in their EXIF and were taken closer than BUILDING_GROUPS_GPS_SAME_SITE_M metres.
The groups are the connected sets of these links (A~B and B~C put A, B, C together): when in doubt two buildings
end up in one group, which can only make the test stricter. Comparing every pair would take hours, so each photo is
compared only with its BUILDING_GROUPS_NEIGHBOURS most similar photos by a quick global descriptor (thumbnail +
colours) and with the photos taken nearby (GPS).

Nothing is renamed before the operator has looked at the result: a preview sheet per group (thumbnails) is written
to '<data>/Building groups preview' with 'building_groups_report.csv'; then the photos get their tag __BLDGnnn
(with their JSON, imagePath and masks). Photos that already have a tag keep it; new photos linked to them join it.
"""
import argparse
import csv
import math
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGES_DIR_NAME, JSON_DIR_NAME, BINARY_DIR_NAME   # noqa: E402

GROUP_PHOTOS_CODE = 141
PREVIEW_DIR_NAME = "Building groups preview"
REPORT_NAME = "building_groups_report.csv"
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
OUTPUT_SUFFIXES = ("-seg", "-crack_mask", "-crack_uncertain_mask", "-detachment_mask", "-ignore_mask")
_TAG = re.compile(r"__BLDG(\d+)", re.IGNORECASE)


class GroupingCancelled(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# Photos
# ----------------------------------------------------------------------------------------------

def building_index(name):
    m = _TAG.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def untagged_base(name):
    return _TAG.sub("", os.path.splitext(os.path.basename(name))[0])


def list_photos(images_dir):
    if not os.path.isdir(images_dir):
        return []
    return sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(PHOTO_EXTENSIONS) and not os.path.splitext(f)[0].endswith("-seg")
                  and not f.startswith("."))


def highest_building_index(data_dir):
    """Highest __BLDGnnn used anywhere in the data folder (Images, the sets, JSON files ...), two levels deep."""
    best = 0
    root_depth = data_dir.rstrip(os.sep).count(os.sep)
    for folder, dirs, files in os.walk(data_dir):
        if folder.count(os.sep) - root_depth >= 2:
            dirs[:] = []
        for f in files:
            idx = building_index(f)
            if idx is not None:
                best = max(best, idx)
    return best


def read_gray(path, max_dim):
    """(grey, colour) of the photo, longer side <= max_dim, upright by its EXIF orientation (cv2.imread turns it,
    like the app shows it); (None, None) when unreadable. The photo is decoded once."""
    colour = cv2.imread(path, cv2.IMREAD_COLOR)
    if colour is None:
        return None, None
    h, w = colour.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        colour = cv2.resize(colour, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY), colour


def read_exif_site(path):
    """(lat, lon, timestamp or None) from the EXIF GPS, or None (no GPS, or Pillow missing)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            gps = ex.get_ifd(0x8825)
            when = ex.get_ifd(0x8769).get(36867) or ex.get(306)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def degrees(v):
        d, m, s = (float(x) for x in v)
        return d + m / 60.0 + s / 3600.0
    try:
        lat, lon = degrees(gps[2]), degrees(gps[4])
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if str(gps.get(1, "N")).upper().startswith("S"):
        lat = -lat
    if str(gps.get(3, "E")).upper().startswith("W"):
        lon = -lon
    stamp = None
    if when:
        try:
            stamp = time.mktime(time.strptime(str(when).strip(), "%Y:%m:%d %H:%M:%S"))
        except ValueError:
            stamp = None
    return lat, lon, stamp


def metres_between(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(min(1.0, h)))


def global_descriptor(gray, colour):
    """Quick 'looks alike' vector: equalised 16x16 thumbnail + HSV colour histogram, unit length."""
    thumb = cv2.resize(cv2.equalizeHist(gray), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb -= thumb.mean()
    thumb /= np.linalg.norm(thumb) + 1e-6
    hist = np.zeros(128, np.float32)
    if colour is not None:
        hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist = np.sqrt(hist / (hist.sum() + 1e-6)).astype(np.float32)
        hist /= np.linalg.norm(hist) + 1e-6
    vec = np.concatenate([thumb, hist])
    return vec / (np.linalg.norm(vec) + 1e-6)


def sift_features(gray, n_features):
    """(points float32 Nx2, descriptors uint8 Nx128): uint8 keeps 650 photos in ~170 MB."""
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.015, edgeThreshold=12)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) < 4:
        return None
    pts = np.float32([k.pt for k in kp])
    return pts, np.clip(np.rint(des), 0, 255).astype(np.uint8)


def match_score(fa, fb):
    """(good matches, inliers, inlier ratio) of two photos, as in the automatic group of Mode 1."""
    if fa is None or fb is None:
        return 0, 0, 0.0
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    try:
        pairs = matcher.knnMatch(fa[1].astype(np.float32), fb[1].astype(np.float32), k=2)
    except cv2.error:
        return 0, 0, 0.0
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    if len(good) < 4:
        return len(good), 0, 0.0
    src = fa[0][[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = fb[0][[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5, maxIters=2000, confidence=0.99)
    if H is None or mask is None:
        return len(good), 0, 0.0
    inliers = int(mask.sum())
    return len(good), inliers, inliers / float(len(good))


# ----------------------------------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------------------------------

class GroupingParams:
    def __init__(self, cfg=None, **override):
        get = (lambda name, default: getattr(cfg, name, default)) if cfg is not None else (lambda n, d: d)
        self.min_good = get("BUILDING_GROUP_MATCH_MIN_GOOD_MATCHES", 12)
        self.min_inliers = get("BUILDING_GROUP_MATCH_MIN_INLIERS", 20)
        self.min_ratio = get("BUILDING_GROUP_MATCH_MIN_INLIER_RATIO", 0.40)
        self.max_dim = get("AUTO_GROUP_SIFT_MAX_DIM", 1200)
        self.neighbours = get("BUILDING_GROUPS_NEIGHBOURS", 10)
        self.features = get("BUILDING_GROUPS_SIFT_FEATURES", 2000)
        self.gps_same_site_m = get("BUILDING_GROUPS_GPS_SAME_SITE_M", 25.0)
        self.gps_candidate_m = get("BUILDING_GROUPS_GPS_CANDIDATE_M", 150.0)
        for k, v in override.items():
            if v is not None:
                setattr(self, k, v)


def _report(progress, **fields):
    if progress is not None:
        progress(**fields)


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise GroupingCancelled()


def candidate_pairs(descs, sites, params):
    """Pairs (i, j), i < j, worth the SIFT check: the most similar photos by global descriptor, plus those taken
    nearby (GPS)."""
    n = len(descs)
    pairs = set()
    if n < 2:
        return []
    mat = np.stack(descs)
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)
    k = min(params.neighbours, n - 1)
    for i in range(n):
        for j in np.argpartition(-sim[i], k - 1)[:k]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    located = [i for i in range(n) if sites[i] is not None]
    for a in range(len(located)):
        for b in range(a + 1, len(located)):
            i, j = located[a], located[b]
            if metres_between(sites[i], sites[j]) <= params.gps_candidate_m:
                pairs.add((i, j))
    return sorted(pairs)


def find_groups(images_dir, photos, params, progress=None, cancelled=None):
    """Links the photos of the same building. Returns {'groups': [[photo, ...], ...] (largest first),
    'links': {photo: (other photo, inliers, metres or None)}, 'pairs': n checked, 'gps': n photos with GPS}.
    progress(phase=..., done=..., total=...) is called as it goes; cancelled() -> True stops (GroupingCancelled)."""
    n = len(photos)
    feats, descs, sites = [None] * n, [np.zeros(384, np.float32)] * n, [None] * n
    start = time.time()
    for i, name in enumerate(photos):
        _check(cancelled)
        path = os.path.join(images_dir, name)
        gray, colour = read_gray(path, params.max_dim)
        if gray is not None:
            feats[i] = sift_features(gray, params.features)
            descs[i] = global_descriptor(gray, colour)
        sites[i] = read_exif_site(path)
        _report(progress, phase="Reading the photos", done=i + 1, total=n, start=start)
    pairs = candidate_pairs(descs, sites, params)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    links = {}
    start = time.time()
    for done, (i, j) in enumerate(pairs, 1):
        _check(cancelled)
        metres = metres_between(sites[i], sites[j]) if sites[i] and sites[j] else None
        same = metres is not None and metres <= params.gps_same_site_m
        inliers = 0
        if not same:
            good, inliers, ratio = match_score(feats[i], feats[j])
            same = good >= params.min_good and inliers >= params.min_inliers and ratio >= params.min_ratio
        if same:
            parent[root(i)] = root(j)
            for a, b in ((i, j), (j, i)):
                if photos[a] not in links or inliers > links[photos[a]][1]:
                    links[photos[a]] = (photos[b], inliers, metres)
        _report(progress, phase="Comparing similar photos", done=done, total=len(pairs), start=start)
    members = {}
    for i, name in enumerate(photos):
        members.setdefault(root(i), []).append(name)
    groups = sorted((sorted(g) for g in members.values()), key=lambda g: (-len(g), g[0]))
    return {"groups": groups, "links": links, "pairs": len(pairs), "gps": sum(1 for s in sites if s)}


def assign_tags(groups, first_free_index):
    """{photo: building index} for the photos to rename. A group with photos already tagged gives its most common
    tag to its untagged photos (tagged photos keep theirs); the other groups get new numbers from first_free_index.
    Also returns {'joined': n untagged photos joining an existing tag, 'mixed': [groups linking several tags]}."""
    plan, joined, mixed = {}, 0, []
    next_index = first_free_index
    for group in groups:
        tags = [building_index(p) for p in group if building_index(p) is not None]
        untagged = [p for p in group if building_index(p) is None]
        if not untagged:
            continue
        if tags:
            counts = {t: tags.count(t) for t in tags}
            target = sorted(counts, key=lambda t: (-counts[t], t))[0]
            if len(counts) > 1:
                mixed.append(sorted(counts))
            joined += len(untagged)
        else:
            target = next_index
            next_index += 1
        for p in untagged:
            plan[p] = target
    return plan, {"joined": joined, "mixed": mixed}


def summary_text(result, plan, info, preview_dir):
    groups = result["groups"]
    multi = [g for g in groups if len(g) > 1]
    singles = len(groups) - len(multi)
    total = sum(len(g) for g in groups)
    lines = [f"{total} photo(s) in {len(groups)} group(s):",
             f"  {len(multi)} group(s) of 2 or more photos (largest: {len(groups[0]) if groups else 0} photos)",
             f"  {singles} photo(s) not similar to any other (a group each)",
             f"  {result['pairs']} pairs of photos compared; {result['gps']} photo(s) with GPS in the EXIF"]
    if groups and total >= 20 and len(groups[0]) > 0.25 * total:
        lines.append(f"\nWARNING: the largest group holds {100 * len(groups[0]) // total}% of the photos: "
                     "it may join different buildings (look at the preview).")
    if info["joined"]:
        lines.append(f"{info['joined']} photo(s) join a building group already tagged.")
    if info["mixed"]:
        lines.append(f"{len(info['mixed'])} group(s) link photos already tagged differently: those keep their tags.")
    lines += ["", f"Preview (one sheet per group): {preview_dir}",
              f"{len(plan)} photo(s) will be renamed with their tag __BLDGnnn (with JSON and masks)."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Preview and report
# ----------------------------------------------------------------------------------------------

def _thumb(path, size=260):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        img = cv2.imread(path)
    if img is None:
        return np.full((size, size, 3), 80, np.uint8)
    h, w = img.shape[:2]
    s = size / float(max(h, w))
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    tile = np.full((size, size, 3), 40, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    return tile


def write_preview(images_dir, groups, plan, out_dir, per_sheet=40, cols=5, progress=None, cancelled=None):
    """One JPEG per group of 2+ photos (thumbnails with their names), plus sheets of the photos alone."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.lower().endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))
    sheets = [(f"group_{n:03d}", g) for n, g in enumerate((g for g in groups if len(g) > 1), 1)]
    singles = [g[0] for g in groups if len(g) == 1]
    sheets += [(f"alone_{n:02d}", singles[i:i + per_sheet])
               for n, i in enumerate(range(0, len(singles), per_sheet), 1)]
    start = time.time()
    for done, (label, photos) in enumerate(sheets, 1):
        _check(cancelled)
        shown = photos[:per_sheet]
        tiles = []
        for p in shown:
            tile = _thumb(os.path.join(images_dir, p))
            tag = plan.get(p, building_index(p))
            text = (p[:22] + ("~" if len(p) > 22 else "")) + (f"  B{tag:03d}" if tag else "")
            cv2.rectangle(tile, (0, tile.shape[0] - 22), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
            cv2.putText(tile, text, (4, tile.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 25))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        sheet = np.vstack(rows)
        header = np.full((40, sheet.shape[1], 3), 255, np.uint8)
        more = f" (first {per_sheet})" if len(photos) > per_sheet else ""
        title = (f"{label}: {len(photos)} photos{more}" if label.startswith("group")
                 else f"{label}: photos not similar to any other")
        cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"{label}_{len(photos)}photos.jpg"), np.vstack([header, sheet]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        _report(progress, phase="Writing the preview", done=done, total=len(sheets), start=start)
    return len(sheets)


def write_report(path, result, plan):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "photo", "new_tag", "linked_to", "inliers", "gps_metres"])
        for n, group in enumerate(result["groups"], 1):
            for p in group:
                link = result["links"].get(p)
                tag = plan.get(p, building_index(p))
                w.writerow([n, p, f"BLDG{tag:03d}" if tag else "", link[0] if link else "",
                            link[1] if link else "", f"{link[2]:.1f}" if link and link[2] is not None else ""])


# ----------------------------------------------------------------------------------------------
# Renaming
# ----------------------------------------------------------------------------------------------

def rename_with_tag(data_dir, photo, index, outputs_dir_name=BINARY_DIR_NAME):
    """Renames an untagged photo of 'Images' to <base>__BLDGnnn<ext>, with its JSON (and imagePath) and its
    overlay/mask files. Returns the new photo name."""
    import json
    images = os.path.join(data_dir, IMAGES_DIR_NAME)
    base, ext = os.path.splitext(photo)
    new_base = f"{base}__BLDG{index:03d}"
    new_photo = new_base + ext
    if os.path.exists(os.path.join(images, new_photo)):
        raise OSError(f"{new_photo} already exists")
    os.rename(os.path.join(images, photo), os.path.join(images, new_photo))
    jsons = os.path.join(data_dir, JSON_DIR_NAME)
    for old_name, new_name in ((f"{base}.json", f"{new_base}.json"), (f"{base}-seg.json", f"{new_base}-seg.json")):
        old_path = os.path.join(jsons, old_name)
        if not os.path.isfile(old_path):
            continue
        try:
            with open(old_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            meta["imagePath"] = new_photo
            with open(old_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        except (OSError, ValueError) as e:
            print(f"[GROUPS] Could not update imagePath in {old_name}: {e}")
        os.rename(old_path, os.path.join(jsons, new_name))
    outputs = os.path.join(data_dir, outputs_dir_name)
    if os.path.isdir(outputs):
        for f in os.listdir(outputs):
            for sfx in OUTPUT_SUFFIXES:
                if f.startswith(base + sfx + "."):
                    os.replace(os.path.join(outputs, f), os.path.join(outputs, new_base + f[len(base):]))
    return new_photo


def apply_plan(data_dir, plan, outputs_dir_name=BINARY_DIR_NAME, progress=None):
    """Renames every photo of the plan. Returns ({old name: new name}, [errors])."""
    renamed, errors = {}, []
    start = time.time()
    for done, (photo, index) in enumerate(sorted(plan.items()), 1):
        try:
            renamed[photo] = rename_with_tag(data_dir, photo, index, outputs_dir_name)
        except OSError as e:
            errors.append(f"{photo}: {e}")
        _report(progress, phase="Renaming the photos", done=done, total=len(plan), start=start)
    return renamed, errors


def progress_text(state):
    """('Comparing similar photos', '1200 of 5400 (22%) - about 3 min left', percent) for a progress display."""
    done, total = state.get("done", 0), state.get("total", 0)
    percent = 100.0 * done / total if total else None
    text = f"{done} of {total}" + (f" ({percent:.0f}%)" if percent is not None else "")
    elapsed = time.time() - state.get("start", time.time())
    if done and total and done < total and elapsed > 2:
        left = elapsed / done * (total - done)
        text += f" - about {left / 60:.0f} min left" if left >= 90 else f" - about {left:.0f} s left"
    return state.get("phase", "..."), text, percent


# ----------------------------------------------------------------------------------------------
# In the app
# ----------------------------------------------------------------------------------------------

class BuildingGroupsMixin:

    def _group_progress_open(self, state):
        """Progress display. Terminal version: a line every 2 s; the GUI opens a window with a progress bar."""
        state["_printed"] = 0.0

    def _group_progress_update(self, state):
        if time.time() - state.get("_printed", 0) < 2 and state.get("done") != state.get("total"):
            return
        state["_printed"] = time.time()
        head, text, _ = progress_text(state)
        print(f" [GROUPS] {head}: {text}")

    def _group_progress_close(self):
        pass

    def _confirm_building_groups(self, message):
        """True = rename the photos. Default: terminal input(); the GUI asks with a dialog."""
        print("\n" + message)
        try:
            return input("Type y to rename the photos, or ENTER to keep them as they are: ").strip().lower() \
                in ("y", "s")
        except Exception:
            return False

    def _notify_building_groups(self, message):
        print(message)

    def _run_with_group_progress(self, work):
        """work(progress, cancelled) in a background thread, with the progress display; GroupingCancelled if stopped."""
        state = {"phase": "Starting...", "done": 0, "total": 0, "start": time.time()}

        def progress(**fields):
            state.update(fields)
        self._group_progress_open(state)
        try:
            return self._run_in_background(lambda cancelled: work(progress, cancelled),
                                           tick=lambda: self._group_progress_update(state))
        finally:
            self._group_progress_close()

    def run_group_similar_photos(self):
        """Code 141 [Group similar photos]. Returns True when photos were renamed (the queue was updated)."""
        cfg = self.cfg
        data_dir = str(cfg.SCRIPT_DIR)
        images_dir = os.path.join(data_dir, IMAGES_DIR_NAME)
        photos = list_photos(images_dir)
        if len(photos) < 2:
            self._notify_building_groups("Group similar photos: fewer than two photos in 'Images'.")
            return False
        if not cfg.queue_idle and cfg.CURRENT_IMAGE_PATH and not self._settle_unsaved_changes("the building groups"):
            print(" [GROUPS] Cancelled: nothing renamed.")
            return False
        params = GroupingParams(cfg)
        preview_dir = os.path.join(data_dir, PREVIEW_DIR_NAME)
        print(f" [GROUPS] Looking for photos of the same building among {len(photos)} photo(s) ... ([Esc] stops)")
        try:
            result = self._run_with_group_progress(
                lambda progress, cancelled: find_groups(images_dir, photos, params, progress, cancelled))
            plan, info = assign_tags(result["groups"], highest_building_index(data_dir) + 1)
            self._run_with_group_progress(
                lambda progress, cancelled: write_preview(images_dir, result["groups"], plan, preview_dir,
                                                          progress=progress, cancelled=cancelled))
        except GroupingCancelled:
            print(" [GROUPS] Stopped: nothing renamed.")
            return False
        write_report(os.path.join(preview_dir, REPORT_NAME), result, plan)
        message = summary_text(result, plan, info, preview_dir)
        print(message)
        if not plan:
            self._notify_building_groups(message + "\n\nEvery photo already has its building tag.")
            return False
        if not self._confirm_building_groups(message + "\n\nRename the photos now?"):
            print(" [GROUPS] Photos left as they are (the preview stays in the folder).")
            return False
        renamed, errors = self._run_with_group_progress(
            lambda progress, cancelled: apply_plan(data_dir, plan, cfg.folder_seg_img, progress))
        self._follow_renamed_photos(images_dir, renamed)
        self._all_segmented_cache = None
        summary = f"{len(renamed)} photo(s) renamed with their building tag." + (
            f"\n{len(errors)} could not be renamed:\n" + "\n".join(errors[:5]) if errors else "")
        print(f" [GROUPS] {summary}")
        self._notify_building_groups(summary + "\n\nNext: Auto split keeps each building in one set.")
        return bool(renamed)

    def _follow_renamed_photos(self, images_dir, renamed):
        """The queue and the photo on screen follow the new names."""
        cfg = self.cfg
        moved = {os.path.join(images_dir, old): os.path.join(images_dir, new) for old, new in renamed.items()}

        def follow(path):
            if not path:
                return path
            return moved.get(path) or moved.get(os.path.abspath(path)) or path
        cfg.image_queue[:] = [follow(p) for p in cfg.image_queue]
        if hasattr(cfg, "historical_processed_queue"):
            cfg.historical_processed_queue[:] = [follow(p) for p in cfg.historical_processed_queue]
        new_current = follow(cfg.CURRENT_IMAGE_PATH)
        if new_current != cfg.CURRENT_IMAGE_PATH:
            old_base = os.path.splitext(os.path.basename(cfg.CURRENT_IMAGE_PATH))[0]
            new_base = os.path.splitext(os.path.basename(new_current))[0]
            cfg.CURRENT_IMAGE_PATH = new_current
            if cfg.JSON_OUTPUT_PATH:
                cfg.JSON_OUTPUT_PATH = os.path.join(os.path.dirname(cfg.JSON_OUTPUT_PATH), f"{new_base}.json")
            if getattr(cfg, "EXPORT_IMAGE_PATH", None):
                cfg.EXPORT_IMAGE_PATH = cfg.EXPORT_IMAGE_PATH.replace(old_base, new_base)


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Groups the photos of 'Images' that show the same building.")
    p.add_argument("--data-dir", required=True, help="the CrackSegmentation data folder (it contains Images/)")
    p.add_argument("--apply", action="store_true", help="rename the photos (default: preview and report only)")
    p.add_argument("--neighbours", type=int, default=None, help="photos compared with each photo (default 10)")
    p.add_argument("--features", type=int, default=None, help="SIFT features per photo (default 2000)")
    p.add_argument("--max-dim", type=int, default=None, help="longer side for SIFT, px (default 1200)")
    p.add_argument("--min-inliers", type=int, default=None, help="RANSAC inliers to link two photos (default 20)")
    args = p.parse_args(argv)
    from config import Config
    cfg = Config()
    params = GroupingParams(cfg, neighbours=args.neighbours, features=args.features, max_dim=args.max_dim,
                            min_inliers=args.min_inliers)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    images_dir = os.path.join(data_dir, IMAGES_DIR_NAME)
    photos = list_photos(images_dir)
    print(f"[GROUPS] {len(photos)} photo(s) in {images_dir}")
    last = [0.0]

    def progress(**state):
        if time.time() - last[0] > 5 or state.get("done") == state.get("total"):
            last[0] = time.time()
            head, text, _ = progress_text(state)
            print(f"[GROUPS] {head}: {text}", flush=True)
    result = find_groups(images_dir, photos, params, progress)
    plan, info = assign_tags(result["groups"], highest_building_index(data_dir) + 1)
    preview_dir = os.path.join(data_dir, PREVIEW_DIR_NAME)
    write_preview(images_dir, result["groups"], plan, preview_dir, progress=progress)
    write_report(os.path.join(preview_dir, REPORT_NAME), result, plan)
    print(summary_text(result, plan, info, preview_dir))
    if args.apply and plan:
        renamed, errors = apply_plan(data_dir, plan, cfg.folder_seg_img, progress)
        print(f"[GROUPS] {len(renamed)} photo(s) renamed." + (f" {len(errors)} error(s):" if errors else ""))
        for e in errors:
            print("  " + e)
    elif plan:
        print("[GROUPS] Nothing renamed: look at the preview, then run again with --apply.")


if __name__ == "__main__":
    main()
