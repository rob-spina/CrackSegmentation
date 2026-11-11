"""auto_ignore.py
v2.0.1 [Auto ignore] (right sidebar, Tools; code 162): ignore regions drawn automatically around what hides or
does not belong to the facade -- trees and plants, grass, fences and site nets, railings, people, vehicles, rubble.

A SegFormer model trained on ADE20K (150 scene classes) labels every pixel of the photo (ignore_infer.py, run in the
SAM Python -- PyTorch + `transformers`); the classes of AUTO_IGNORE_CLASSES are merged, cleaned (small holes closed,
specks dropped, a small margin added) and turned into polygons, saved as ignore regions [I] like the hand-drawn ones:
they can be reshaped or deleted, they go into '<photo>-ignore_mask.png' and are left out of training and scoring.

  - The class map is cached in '<data>/SAM model/ignore maps/<model>/<photo>.png': pressing again, or another class
    list, costs nothing. auto_ignore_batch.py computes the maps of every photo at once and compares the automatic
    regions with the ignore masks already drawn by hand.
  - Pressing again replaces the automatic regions of the photo (after a question); hand-drawn ones are kept.
  - One [U] removes all the regions of one run.
  - Not recognised as a class: ruins and dismantled parts of a building (ADE20K calls them building / wall), and areas
    with too many cracks to trace -- draw those with [I].
"""
import json
import os
import re
import sys
import tempfile
import time

import cv2
import numpy as np


AUTO_IGNORE_CODE = 162
AUTO_IGNORE_SOURCE = "auto_ignore"
IGNORE_MAPS_DIR = "ignore maps"

# ADE20K classes (names as in the model's id2label; any of the ';'-separated synonyms matches).
DEFAULT_AUTO_IGNORE_CLASSES = ("tree", "palm", "plant", "flower", "grass", "field", "fence", "railing", "person",
                               "car", "truck", "van", "bus", "bicycle", "minibike", "rock")


def model_slug(model, long_side):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip("/")) + f"-L{int(long_side)}"


def class_ids(id2label, wanted):
    """{id: name} of the classes in `wanted` (case-insensitive; a label 'plant;flora;plant;life' matches 'plant')."""
    wanted = {w.strip().lower() for w in wanted if w and w.strip()}
    out = {}
    for key, label in (id2label or {}).items():
        names = [n.strip().lower() for n in re.split(r"[;,]", str(label))]
        if wanted.intersection(names):
            out[int(key)] = str(label).split(";")[0].split(",")[0].strip()
    return out


def occluder_mask(classes, ids):
    """Boolean mask of the pixels of a class-id map that belong to one of the ids."""
    if not ids:
        return np.zeros(classes.shape[:2], bool)
    return np.isin(classes, np.fromiter(ids, dtype=np.int32))


def clean_mask(mask, close_px=9, min_area_px=0, margin_px=0):
    """Closes small gaps (branches, fence meshes), fills holes, drops pieces smaller than min_area_px and grows the
    rest by margin_px. uint8 0/255."""
    m = mask.astype(np.uint8) * 255
    if close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px), int(close_px)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    keep = [c for c in contours if cv2.contourArea(c) >= max(1, min_area_px)]
    if keep:
        cv2.drawContours(out, keep, -1, 255, -1)      # filled: holes inside a tree are part of it
    if margin_px > 0 and keep:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(margin_px) + 1, 2 * int(margin_px) + 1))
        out = cv2.dilate(out, k)
    return out


def mask_to_polygons(mask, epsilon_px=3.0, max_regions=20, scale=1.0, size=None, snap_px=2.0):
    """Polygons (lists of (x, y) in photo pixels) around the blobs of a 0/255 mask, largest first. scale = photo px
    per mask px; corners within snap_px (mask px) of an edge are put on it, so a tree cut by the photo edge gives a
    region that reaches the edge."""
    h, w = mask.shape[:2]
    W, H = size if size else (int(round(w * scale)), int(round(h * scale)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max(1, int(max_regions))]
    polys = []
    for c in contours:
        approx = cv2.approxPolyDP(c, float(epsilon_px), True).reshape(-1, 2).astype(np.float64)
        if len(approx) < 3:
            continue
        pts = []
        for x, y in approx:
            x = 0.0 if x <= snap_px else (w - 1.0 if x >= w - 1 - snap_px else x)
            y = 0.0 if y <= snap_px else (h - 1.0 if y >= h - 1 - snap_px else y)
            px = min(W - 1, max(0, int(round(x * scale if x < w - 1 else W - 1))))
            py = min(H - 1, max(0, int(round(y * scale if y < h - 1 else H - 1))))
            if not pts or pts[-1] != (px, py):
                pts.append((px, py))
        if len(pts) >= 3 and cv2.contourArea(np.array(pts, np.float32).reshape(-1, 1, 2)) > 0:
            polys.append(pts)
    return polys


def regions_from_classes(classes, id2label, wanted, photo_size, close_fraction=0.01, min_area_fraction=0.003,
                         margin_px=12, epsilon_fraction=0.003, max_regions=20):
    """(polygons in photo px, {class name: pixel share of the photo}) from a class-id map."""
    ids = class_ids(id2label, wanted)
    h, w = classes.shape[:2]
    W, H = photo_size
    scale = float(W) / w
    found = {}
    total = float(h * w)
    for cid, name in ids.items():
        share = float(np.count_nonzero(classes == cid)) / total
        if share > 0:
            found[name] = found.get(name, 0.0) + share
    mask = clean_mask(occluder_mask(classes, ids), close_px=max(3, int(round(close_fraction * max(h, w)))),
                      min_area_px=min_area_fraction * total, margin_px=max(0, int(round(margin_px / scale))))
    polys = mask_to_polygons(mask, epsilon_px=max(1.0, epsilon_fraction * max(h, w)), max_regions=max_regions,
                             scale=scale, size=(W, H))
    return polys, found


def polygons_mask(polys, size):
    W, H = size
    out = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(out, [np.array(p, np.int32)], 255)
    return out


def read_labels(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("id2label", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


class AutoIgnoreMixin:
    """[Auto ignore]. Relies on CrackSegmentation's cfg, the SAM settings (its Python) and the ignore-region tool."""

    def _auto_ignore_dir(self):
        cfg = self.cfg
        return os.path.join(self._sam_model_dir(), IGNORE_MAPS_DIR,
                            model_slug(cfg.AUTO_IGNORE_MODEL, cfg.AUTO_IGNORE_LONG_SIDE))

    def _auto_ignore_map_path(self, photo=None):
        stem = os.path.splitext(os.path.basename(photo or self.cfg.CURRENT_IMAGE_PATH))[0]
        return os.path.join(self._auto_ignore_dir(), stem + ".png")

    @staticmethod
    def _auto_ignore_script_path():
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "ignore_infer.py")

    def _auto_ignore_python(self):
        settings = self._load_sam_settings() or {}
        return os.path.expanduser(settings.get("python") or self._default_sam_python())

    def _auto_ignore_command(self, image, out, labels_out):
        cfg = self.cfg
        return [self._auto_ignore_python(), "-u", self._auto_ignore_script_path(), "--image", image, "--out", out,
                "--labels-out", labels_out, "--model", cfg.AUTO_IGNORE_MODEL,
                "--long-side", str(cfg.AUTO_IGNORE_LONG_SIDE), "--device", cfg.AUTO_IGNORE_DEVICE]

    def _auto_ignore_class_map(self):
        """(class-id map, id2label) of the photo on screen -- cached, or computed now -- or (None, None)."""
        path = self._auto_ignore_map_path()
        labels_path = os.path.join(self._auto_ignore_dir(), "labels.json")
        classes = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(path) else None
        id2label = read_labels(labels_path)
        if classes is not None and id2label:
            print(" [AUTO IGNORE] Using the class map already computed for this photo.")
            return classes, id2label
        os.makedirs(self._auto_ignore_dir(), exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crackseg_ignore_") as tmp:
            out = os.path.join(tmp, "classes.png")
            print(" [AUTO IGNORE] Recognising trees, fences, people, vehicles ... (Esc cancels)")
            try:
                code = self._run_sam_process(self._auto_ignore_command(self.cfg.CURRENT_IMAGE_PATH, out, labels_path))
            except OSError as e:
                print(f" [AUTO IGNORE] Could not start {self._auto_ignore_python()}: {e}", file=sys.stderr)
                return None, None
            if code is None:
                print(" [AUTO IGNORE] Cancelled: nothing changed.")
                return None, None
            classes = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
            if code != 0 or classes is None:
                print(f" [AUTO IGNORE] The model did not run (exit code {code}); see the lines above. It needs PyTorch "
                      f"and transformers in {self._auto_ignore_python()}: "
                      f"{self._auto_ignore_python()} -m pip install transformers", file=sys.stderr)
                self._notify_auto_ignore(
                    "Auto ignore: the model did not run.\n\nIt needs PyTorch and 'transformers' in the SAM Python:\n"
                    f"{self._auto_ignore_python()} -m pip install transformers\n\n(details in the status lines)")
                return None, None
            cv2.imwrite(path, classes)
        return classes, read_labels(labels_path)

    def _confirm_auto_ignore_replace(self, count):
        try:
            return input(f"Replace the {count} automatic ignore region(s) of this photo? [y/N]: ").strip().lower() \
                in ("y", "yes", "s", "si")
        except EOFError:
            return False

    def _notify_auto_ignore(self, message):
        print(message)

    def run_auto_ignore(self):
        """Code 162: automatic ignore regions on the photo on screen. True when regions were added."""
        cfg = self.cfg
        if getattr(cfg, "queue_idle", False) or not cfg.CURRENT_IMAGE_PATH:
            print(" [AUTO IGNORE] No photo on screen.")
            return False
        old = [r for r in cfg.saved_ignores if r.get('active', True) and r.get('source') == AUTO_IGNORE_SOURCE]
        if old and not self._confirm_auto_ignore_replace(len(old)):
            print(" [AUTO IGNORE] Cancelled: the automatic regions stay as they are.")
            return False
        start = time.time()
        classes, id2label = self._auto_ignore_class_map()
        if classes is None:
            return False
        polys, found = regions_from_classes(
            classes, id2label, cfg.AUTO_IGNORE_CLASSES, (cfg.W_img, cfg.H_img),
            close_fraction=cfg.AUTO_IGNORE_CLOSE_FRACTION, min_area_fraction=cfg.AUTO_IGNORE_MIN_AREA_FRACTION,
            margin_px=cfg.AUTO_IGNORE_MARGIN_PX, epsilon_fraction=cfg.AUTO_IGNORE_EPSILON_FRACTION,
            max_regions=cfg.AUTO_IGNORE_MAX_REGIONS)
        what = ", ".join(f"{n} {100 * s:.0f}%" for n, s in sorted(found.items(), key=lambda kv: -kv[1]) if s >= 0.005)
        if not polys:
            message = ("Auto ignore: nothing to ignore found in this photo (no trees, fences, people, vehicles ...)."
                       + (f"\nSeen, too small: {what}" if what else ""))
            print(" [AUTO IGNORE] " + message.replace("\n", " "))
            self._notify_auto_ignore(message)
            return False
        stamp = str(time.time())
        added = [{'path': p, 'active': True, 'session_id': stamp, 'source': AUTO_IGNORE_SOURCE} for p in polys]
        cfg.saved_ignores[:] = [r for r in cfg.saved_ignores if not any(r is o for o in old)] + added
        cfg.action_history.append(('ignore_batch', (added, old)))
        mask = polygons_mask(polys, (cfg.W_img, cfg.H_img))
        share = 100.0 * np.count_nonzero(mask) / float(mask.size)
        print(f" [AUTO IGNORE] {len(polys)} region(s) added ({share:.0f}% of the photo; {what or 'n/a'}) in "
              f"{time.time() - start:.1f} s. Click inside one with [I] to delete it, drag a corner to reshape it; "
              "[U] removes them all; S saves.")
        self.refresh_zoom_viewport()
        return True

    # --- undo / redo of a whole run (called by the ignore-region undo) ----------------------------

    def _undo_auto_ignore(self, last_action):
        if not (isinstance(last_action, tuple) and last_action and last_action[0] == 'ignore_batch'):
            return False
        added, removed = last_action[1]
        self.cfg.saved_ignores[:] = [r for r in self.cfg.saved_ignores if not any(r is a for a in added)] + removed
        self.cfg.redo_history.append(('ignore_batch', (added, removed)))
        print(f"[UNDO] {len(added)} automatic ignore region(s) removed.")
        return True

    def _redo_auto_ignore(self, action_type, restored):
        if action_type != 'ignore_batch':
            return False
        added, removed = restored
        self.cfg.saved_ignores[:] = [r for r in self.cfg.saved_ignores if not any(r is o for o in removed)] + added
        self.cfg.action_history.append(('ignore_batch', (added, removed)))
        print(f"[REDO] {len(added)} automatic ignore region(s) restored.")
        return True
