"""sam_accept.py
v2.0.0 [SAM click to accept] (right sidebar AI section, Tools menu, code 138), mixed into CrackSegmentation.
Interactive use of the fine-tuned SAM crack model -- only what the operator clicks is saved:
  - pressing the button computes the model's crack map of the photo once (or reuses the one [SAM auto-segment] or
    sam_batch.py already computed) and turns the mode on; nothing is added;
  - v2.0.1 colours: every crack the model sees (a connected region of the map, p >= SAM_ACCEPT_LOW_THRESHOLD) is
    coloured at once by how sure the model is (mean probability of the region):
        orange       >= SAM_ACCEPT_AUTO_MIN_MEAN (0.65): the model is sure
        dark yellow  >= SAM_ACCEPT_DARK_MIN_MEAN (0.45)
        light yellow below: the model is not very sure
    the region under the mouse is outlined in cyan: exactly what a click takes;
  - keys while the mode is on (they leave the mode on):  H  hides / shows every colour;  G  only the yellow ones;
    A  only the orange ones;  L  discards every light-yellow crack,  D  every dark-yellow one (they leave the screen
    and are not proposed again on this photo; nothing is deleted from the photo's JSON);
  - a click on a crack accepts it: the connected part of the map around the click becomes cracks of the app, with
    SAM's exact shape as the width ("source": "sam_click", never uncertain). Ctrl+click on the button (code 142)
    accepts every orange crack still on screen in one go. JSON and binary masks are written after every
    acceptance; [U] undoes; the operator's cracks and the ignore regions are left alone;
  - a photo that still has cracks added automatically by v2.0.0 ("source": "sam", never clicked) asks once whether
    to remove them;
  - the mode stays on from photo to photo (Enter / Q / Go to image) until the button is pressed again; any other
    tool pauses it on this photo.
"""
import os
import time

import cv2
import numpy as np

from crack_relabel import RELABEL_KEEP_KEYS, RELABEL_TARGET_BY_CODE
from review_flags import REVIEW_CODE
from sam_autoseg import mask_to_crack_shapes

SAM_ACCEPT_CODE = 138
SAM_CANDIDATES_CODE = 139   # Shift+click on the button: hide / show every colour (as H)
SAM_ACCEPT_ALL_CODE = 142   # v2.0.1: accept every orange crack (Ctrl+click on the button, Tools menu)
SAM_OVERLAY_CODE = 143      # v2.0.1: hide / show every colour (H, Tools menu)
SAM_DISCARD_LIGHT_CODE = 144   # v2.0.1: discard every light-yellow crack (L, Tools menu)
SAM_DISCARD_DARK_CODE = 145    # v2.0.1: discard every dark-yellow crack (D, Tools menu)
SAM_YELLOW_CODE = 146       # v2.0.1: hide / show the yellow cracks (G, Tools menu)
SAM_ORANGE_CODE = 147       # v2.0.1: hide / show the orange cracks (A, Tools menu)
SAM_EXCLUDE_CODE = 148      # v2.0.1: [Exclude crack] on / off (X): a click removes a crack (also from JSON and PNG)
SAM_ACCEPT_DARK_CODE = 150  # v2.0.1: accept every dark-yellow crack (2)
SAM_ACCEPT_LIGHT_CODE = 151  # v2.0.1: accept every light-yellow crack (3)
SAM_ACCEPT_EVERY_CODE = 152  # v2.0.1: accept every coloured crack (4)
SAM_THRESHOLD_DOWN_CODE = 155   # v2.0.1: lower the map threshold -> more cracks, fewer false negatives ([)
SAM_THRESHOLD_UP_CODE = 156     # v2.0.1: raise it -> fewer cracks, fewer false positives (])
SAM_THRESHOLD_DIAL_CODE = 157   # v2.0.1: the threshold dial window
SAM_RECOMPUTE_MAP_CODE = 158    # v2.0.1: compute the crack map of this photo again, here (e.g. a bad Leonardo map)
SAM_WIDTH_LIMIT_CODE = 159      # v2.0.1: the maximum-width limit on / off, its value kept (0)
SAM_CUT_CODE = 161              # v2.0.1: [Cut map] on / off (K): a line dragged across the map splits it in two
CUT_LINE_COLOR = (255, 0, 255)  # BGR magenta: the cut being drawn

# Classes of a map region, by its mean probability.
CLASS_NONE, CLASS_LIGHT, CLASS_DARK, CLASS_SURE = 0, 1, 2, 3
CLASS_COLORS = {CLASS_LIGHT: (150, 255, 255),   # BGR light yellow
                CLASS_DARK: (0, 215, 255),      # BGR dark yellow (gold)
                CLASS_SURE: (0, 110, 255)}      # BGR orange
CLASS_NAMES = {CLASS_LIGHT: "light-yellow", CLASS_DARK: "dark-yellow", CLASS_SURE: "orange"}
HOVER_OUTLINE = (255, 255, 0)     # BGR cyan: what a click takes

# Keys of the mode itself (only while it is on): they replace these letters' usual commands.
SAM_MODE_KEYS = {ord('h'): SAM_OVERLAY_CODE, ord('H'): SAM_OVERLAY_CODE, ord('g'): SAM_YELLOW_CODE,
                 ord('G'): SAM_YELLOW_CODE, ord('a'): SAM_ORANGE_CODE, ord('A'): SAM_ORANGE_CODE,
                 ord('l'): SAM_DISCARD_LIGHT_CODE, ord('L'): SAM_DISCARD_LIGHT_CODE,
                 ord('d'): SAM_DISCARD_DARK_CODE, ord('D'): SAM_DISCARD_DARK_CODE,
                 ord('1'): SAM_ACCEPT_ALL_CODE, ord('2'): SAM_ACCEPT_DARK_CODE, ord('3'): SAM_ACCEPT_LIGHT_CODE,
                 ord('4'): SAM_ACCEPT_EVERY_CODE, ord('x'): SAM_EXCLUDE_CODE, ord('X'): SAM_EXCLUDE_CODE,
                 ord('['): SAM_THRESHOLD_DOWN_CODE, ord(']'): SAM_THRESHOLD_UP_CODE, ord('0'): SAM_WIDTH_LIMIT_CODE,
                 ord('k'): SAM_CUT_CODE, ord('K'): SAM_CUT_CODE}
SAM_MODE_CODES = frozenset([SAM_CANDIDATES_CODE, SAM_ACCEPT_ALL_CODE, SAM_OVERLAY_CODE, SAM_DISCARD_LIGHT_CODE,
                            SAM_DISCARD_DARK_CODE, SAM_YELLOW_CODE, SAM_ORANGE_CODE, SAM_EXCLUDE_CODE,
                            SAM_ACCEPT_DARK_CODE, SAM_ACCEPT_LIGHT_CODE, SAM_ACCEPT_EVERY_CODE,
                            SAM_THRESHOLD_DOWN_CODE, SAM_THRESHOLD_UP_CODE, SAM_THRESHOLD_DIAL_CODE,
                            SAM_WIDTH_LIMIT_CODE, SAM_CUT_CODE])
EXCLUDE_OUTLINE = (0, 0, 255)     # BGR red: what a click removes in [Exclude crack]
# Keys that move to another photo: they close the mode for this photo but it reopens on the next one.
NAVIGATION_KEYS = frozenset([13, 10, ord('q'), ord('Q'), 29, 31, 4, 5])   # 4/5: next/previous without saving
SAVING_NAVIGATION_KEYS = frozenset([13, 10, ord('q'), ord('Q')])   # these save the photo themselves
SAM_ACCEPT_SOURCE = "sam_click"
# Keys that leave the mode on: view keys, Undo/Redo, Save, the mode's own keys and the button itself.
SAM_ACCEPT_KEEP_KEYS = (RELABEL_KEEP_KEYS | frozenset(RELABEL_TARGET_BY_CODE) | SAM_MODE_CODES
                        | frozenset([ord('u'), ord('U'), ord('r'), ord('R'), ord('s'), ord('S'),
                                     130, 131, 133, SAM_ACCEPT_CODE, REVIEW_CODE, 149,   # 149: Delete image
                                     SAM_RECOMPUTE_MAP_CODE]))


def accept_region(prob_u8, x, y, low_threshold, radius, blocked=None):
    """The crack region accepted by a click at (x, y): the connected component of prob >= low_threshold that
    contains the most probable unblocked pixel within radius of the click. Returns (mask uint8 0/255, seed) or
    (None, reason) with reason 'nothing' (no crack near the click) or 'taken' (already a crack there)."""
    h, w = prob_u8.shape
    low = int(round(low_threshold * 255))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    window = prob_u8[y0:y1, x0:x1].astype(np.int32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    window[(yy - y) ** 2 + (xx - x) ** 2 > radius * radius] = -1
    if blocked is not None:
        free = window.copy()
        free[blocked[y0:y1, x0:x1] > 0] = -1
        if free.max() < low <= window.max():
            return None, "taken"
        window = free
    if window.max() < low:
        return None, "nothing"
    sy, sx = np.unravel_index(int(np.argmax(window)), window.shape)
    seed = (int(sx + x0), int(sy + y0))
    candidate = (prob_u8 >= low).astype(np.uint8)
    if blocked is not None:
        candidate[blocked > 0] = 0
    _, labels = cv2.connectedComponents(candidate, connectivity=8)
    region = (labels == labels[seed[1], seed[0]]).astype(np.uint8) * 255
    return region, seed


class SamAcceptMixin:
    """Relies on SamAutoSegMixin (model settings, crack map, mask -> cracks) and CrackSegmentation's save helpers."""

    # The letters H/G/A/L/D are translated here (OpenCV window: every key arrives as its letter). The GUI translates
    # only the real keystrokes, so that its sidebar buttons (Detachment tool = 'd' ...) keep their meaning.
    _SAM_LETTERS_IN_PROCESS = True

    # --- Mode -------------------------------------------------------------------------------

    def _toggle_sam_accept(self):
        """Code 138: turns the mode on (computing the crack map if needed) or off."""
        state = self.cfg.sam_accept_state
        if state["active"]:
            state["sticky"] = False
            self._save_sam_changes_if_any()
            self._exit_sam_accept()
            return False
        return self._start_sam_accept()   # also resumes it on this photo after a pause

    def _start_sam_accept(self):
        state = self.cfg.sam_accept_state
        cfg = self.cfg
        if not cfg.CURRENT_IMAGE_PATH or cfg.img_original is None or getattr(cfg, "queue_idle", False):
            print(" [SAM CLICK] No photo on screen.")
            return False
        settings = self._usable_sam_settings()
        if not settings:
            return False
        print(f" [SAM CLICK] Crack map of {os.path.basename(cfg.CURRENT_IMAGE_PATH)} ... ([Esc] cancels)")
        prob = self._sam_probability_map(settings)
        if prob is None:
            state["sticky"] = False
            return False
        labels, stats, wide = self._cut_sam_map(prob, cfg.SAM_ACCEPT_LOW_THRESHOLD)
        state.update(active=True, photo=cfg.CURRENT_IMAGE_PATH, prob=prob, labels=labels, stats=stats, hover=0,
                     wide_mask=wide,
                     unsaved=False, exclude=False, window_drag=None, prob_private=False, discarded_mask=None,
                     clicks=0, sticky=bool(cfg.SAM_ACCEPT_STAY_ON), done_labels=set(), discarded=set(),
                     proposed=set(), overlay=None, hover_contour=None, cracks_signature=None,
                     width_probe=None, cut=False, cut_path=None, cut_mask=None)
        self._offer_removal_of_automatic_sam_cracks()
        self._classify_sam_regions()
        counts = self._sam_class_counts()
        print(f" [SAM CLICK] On: {counts[CLASS_SURE]} orange (the model is sure), {counts[CLASS_DARK]} dark-yellow, "
              f"{counts[CLASS_LIGHT]} light-yellow crack(s). Only the cracks you click are saved. "
              "H hides/shows all colours, G the yellow ones, A the orange ones; L discards the light-yellow ones, "
              "D the dark-yellow ones; Ctrl+click on the button accepts every orange one; [U] undoes; "
              "the mode stays on for the next photo; press the button to finish.")
        self.refresh_zoom_viewport()
        return True

    def _sam_changed(self, what):
        """v2.0.1: a change made in the mode. Saved at once only with SAM_ACCEPT_SAVE_EACH_CHANGE (writing the JSON and
        the 4K masks after every click made the mode sluggish); otherwise with the photo (Enter / S), or when the mode
        is turned off or another photo is opened from it."""
        state = self.cfg.sam_accept_state
        if self.cfg.SAM_ACCEPT_SAVE_EACH_CHANGE:
            self._save_annotations_now("SAM CLICK", what)
            return
        if not state.get("unsaved"):
            print(" [SAM CLICK] Not saved yet: Enter / S saves the photo (it is saved anyway when you turn the mode "
                  "off or move to another photo).")
        state["unsaved"] = True

    def _save_sam_changes_if_any(self):
        state = self.cfg.sam_accept_state
        if state.get("unsaved"):
            state["unsaved"] = False
            self._save_annotations_now("SAM CLICK", "the cracks of SAM click to accept")

    def _exit_sam_accept(self, reason=None):
        state = self.cfg.sam_accept_state
        if not state["active"]:
            return
        clicks = state["clicks"]
        state.update(active=False, photo=None, prob=None, labels=None, stats=None, hover=0, clicks=0,
                     done_labels=set(), discarded=set(), proposed=set(), classes=None, overlay=None, exclude=False,
                     window_drag=None,
                     hover_contour=None, cracks_signature=None, width_probe=None, cut=False, cut_path=None,
                     cut_mask=None)
        print(f" [SAM CLICK] Off{': ' + reason if reason else ''} ({clicks} crack(s) accepted).")
        self.refresh_zoom_viewport()

    # --- Classes ------------------------------------------------------------------------------

    def _classify_sam_regions(self):
        """classes[label] = light / dark yellow or orange by the region's mean probability; regions too small, too
        large (not a crack) or already traced get no colour."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        prob, labels = state["prob"], state["labels"]
        n = int(labels.max()) + 1
        classes = np.zeros(n, np.uint8)
        if n > 1:
            # only the crack pixels (label > 0): a few % of a 4K photo, so the threshold dial stays responsive
            inside = labels > 0
            flat = labels[inside]
            counts = np.bincount(flat, minlength=n)
            means = np.bincount(flat, weights=prob[inside].astype(np.float64), minlength=n) / np.maximum(counts, 1) \
                / 255.0
            crack_sized = (counts >= cfg.SAM_MIN_AREA_PX) & (counts <= cfg.SAM_ACCEPT_MAX_AREA_FRACTION * prob.size)
            classes[crack_sized] = CLASS_LIGHT
            classes[crack_sized & (means >= cfg.SAM_ACCEPT_DARK_MIN_MEAN)] = CLASS_DARK
            classes[crack_sized & (means >= cfg.SAM_ACCEPT_AUTO_MIN_MEAN)] = CLASS_SURE
            centre = state.get("centre_mask")
            if centre is not None and centre.any():   # the centre lines of wide lesions: always orange
                on_centre = np.bincount(labels[centre], minlength=n) > 0
                classes[on_centre & (counts >= cfg.SAM_MIN_AREA_PX)] = CLASS_SURE
            classes[0] = CLASS_NONE
        state["classes"] = classes
        self._refresh_sam_done_labels()

    def _cracks_signature(self):
        return tuple(id(c) for c in self.cfg.saved_cracks if c.get('active', True)) + (len(self.cfg.saved_ignores),)

    def _refresh_sam_done_labels(self):
        """done = the regions discarded with L / D plus those already traced (at least half covered by a crack):
        recomputed whenever the cracks change -- accepted, deleted (also by their number), undone, redone."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        labels, classes = state["labels"], state.get("classes")
        done = set(state.get("discarded") or set())
        if classes is not None and len(classes) > 1:
            n = len(classes)
            traced = self._sam_existing_coverage([f for f in cfg.saved_cracks if f.get('active', True)]) > 0
            if traced.any():
                counts = state["stats"][:, cv2.CC_STAT_AREA] if state.get("stats") is not None \
                    else np.bincount(labels.ravel(), minlength=n)
                covered = np.bincount(labels[traced], minlength=n)
                already = (covered >= 0.5 * counts[:n]) & (classes > 0)
                done.update(int(k) for k in np.nonzero(already)[0])
        state["done_labels"] = done
        state["proposed"] = ({int(k) for k in np.nonzero(classes == CLASS_SURE)[0]} - done) if classes is not None \
            else set()
        state["cracks_signature"] = self._cracks_signature()
        state["overlay"] = None
        if state.get("hover") in done:
            state["hover"], state["hover_contour"] = 0, None

    def _sam_labels_of_class(self, cls):
        state = self.cfg.sam_accept_state
        classes = state.get("classes")
        if classes is None:
            return set()
        if state.get("cracks_signature") != self._cracks_signature():   # e.g. after [U]: undone cracks colour again
            self._refresh_sam_done_labels()
        return {int(k) for k in np.nonzero(classes == cls)[0]} - (state.get("done_labels") or set())

    def _sam_class_counts(self):
        return {c: len(self._sam_labels_of_class(c)) for c in (CLASS_LIGHT, CLASS_DARK, CLASS_SURE)}

    # --- Mode keys ----------------------------------------------------------------------------

    def _handle_sam_mode_key(self, key_clean):
        """While the mode is on, H / G / A / L / D are the mode's own commands (and the menu codes always are).
        True when the key was used here."""
        state = self.cfg.sam_accept_state
        code = key_clean
        if state["active"] and self._SAM_LETTERS_IN_PROCESS:
            code = SAM_MODE_KEYS.get(key_clean, key_clean)
        if code not in SAM_MODE_CODES:
            return False
        if not state["active"]:
            print(" [SAM CLICK] Turn SAM click to accept on first.")
            return True
        if code in (SAM_OVERLAY_CODE, SAM_CANDIDATES_CODE):
            self._toggle_sam_colours("all")
        elif code == SAM_YELLOW_CODE:
            self._toggle_sam_colours("yellow")
        elif code == SAM_ORANGE_CODE:
            self._toggle_sam_colours("orange")
        elif code == SAM_DISCARD_LIGHT_CODE:
            self._discard_sam_class(CLASS_LIGHT)
        elif code == SAM_DISCARD_DARK_CODE:
            self._discard_sam_class(CLASS_DARK)
        elif code == SAM_ACCEPT_ALL_CODE:
            self._accept_sam_classes((CLASS_SURE,))
        elif code == SAM_ACCEPT_DARK_CODE:
            self._accept_sam_classes((CLASS_DARK,))
        elif code == SAM_ACCEPT_LIGHT_CODE:
            self._accept_sam_classes((CLASS_LIGHT,))
        elif code == SAM_ACCEPT_EVERY_CODE:
            self._accept_sam_classes((CLASS_SURE, CLASS_DARK, CLASS_LIGHT))
        elif code == SAM_EXCLUDE_CODE:
            self._toggle_sam_exclude()
        elif code == SAM_THRESHOLD_DOWN_CODE:
            self._set_sam_threshold(self.cfg.SAM_ACCEPT_LOW_THRESHOLD - self.cfg.SAM_THRESHOLD_STEP)
        elif code == SAM_THRESHOLD_UP_CODE:
            self._set_sam_threshold(self.cfg.SAM_ACCEPT_LOW_THRESHOLD + self.cfg.SAM_THRESHOLD_STEP)
        elif code == SAM_THRESHOLD_DIAL_CODE:
            self._open_sam_threshold_dial()
        elif code == SAM_WIDTH_LIMIT_CODE:
            self._toggle_sam_width_limit()
        elif code == SAM_CUT_CODE:
            self._toggle_sam_cut()
        return True

    # --- Threshold ------------------------------------------------------------------------------

    def _open_sam_threshold_dial(self):
        """Code 157. Terminal version: the value and the keys; the GUI opens the round dial."""
        print(f" [SAM CLICK] Threshold {self.cfg.SAM_ACCEPT_LOW_THRESHOLD:.2f}: [ lowers it (more cracks, fewer false "
              "negatives), ] raises it (fewer cracks, fewer false positives).")

    def _remember_discarded_pixels(self, labels_gone):
        """The discarded cracks as pixels, so that they stay discarded when the threshold re-cuts the map."""
        state = self.cfg.sam_accept_state
        if not labels_gone:
            return
        mask = np.isin(state["labels"], list(labels_gone))
        state["discarded_mask"] = mask if state.get("discarded_mask") is None else (state["discarded_mask"] | mask)

    def _set_sam_threshold(self, value):
        """Re-cuts the crack map at a new threshold (p >= value is a crack): lower = more and longer cracks (fewer
        false negatives), higher = fewer (fewer false positives). The accepted cracks stay as they are; discarded
        ones stay discarded. Returns the threshold in use."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        value = round(float(min(cfg.SAM_THRESHOLD_MAX, max(cfg.SAM_THRESHOLD_MIN, value))), 3)
        if not state["active"] or state.get("prob") is None:
            cfg.SAM_ACCEPT_LOW_THRESHOLD = value
            return value
        if abs(value - cfg.SAM_ACCEPT_LOW_THRESHOLD) < 1e-6 and state.get("classes") is not None:
            return value
        cfg.SAM_ACCEPT_LOW_THRESHOLD = value
        self._recut_sam_map(f"Threshold {value:.2f}")
        return value

    def _width_limit_px(self):
        cfg = self.cfg
        return float(cfg.SAM_ACCEPT_MAX_WIDTH_PX or 0) if cfg.SAM_WIDTH_LIMIT_ON else 0.0

    def _cut_sam_map(self, prob, threshold):
        """(labels, stats, blocked wide mask): the crack pixels p >= threshold cut into connected regions. Parts wider
        than SAM_ACCEPT_MAX_WIDTH_PX (an open lesion, missing plaster, gaps between stones) are replaced by their
        centre line SAM_WIDE_CENTERLINE_PX wide (shown orange; accepted as a crack of that width) or, with
        SAM_WIDE_AS_CENTERLINE False, dropped. A thin crack touching a wide patch keeps its thin part."""
        cfg = self.cfg
        low = (prob >= int(round(threshold * 255))).astype(np.uint8)
        wide, centre = None, None
        max_w = self._width_limit_px()
        if max_w > 0 and low.any():
            dist = cv2.distanceTransform(low, cv2.DIST_L2, 3)
            core = (dist > max_w / 2.0).astype(np.uint8)
            if core.any():
                # the wide parts: every crack pixel within max_w / 2 of a point deeper than max_w / 2
                near = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3) <= max_w / 2.0 + 1.0
                wide = near & (low > 0)
                low[wide] = 0
                full = wide.copy()
                if cfg.SAM_WIDE_AS_CENTERLINE:
                    centre = self._centre_lines(wide, int(cfg.SAM_WIDE_CENTERLINE_PX))
                    low[centre] = 1
                    wide &= ~centre
        _, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
        self.cfg.sam_accept_state["centre_mask"] = centre
        self.cfg.sam_accept_state["wide_full"] = full if wide is not None else None
        return labels, stats, wide

    def _expand_to_wide(self, region, box=None):
        """v2.0.1: a region touching a centre line becomes the whole wide patch under it (its REAL width -- the PNG
        must show the lesion as it is), unless SAM_WIDE_ACCEPT_FULL is False (the centre line only). box =
        (x0, y0, x1, y1) keeps the patch inside a window. region: bool mask, changed in place and returned."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        centre, full = state.get("centre_mask"), state.get("wide_full")
        if not cfg.SAM_WIDE_ACCEPT_FULL or centre is None or full is None or not (region & centre).any():
            return region
        _, comp = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
        touched = np.unique(comp[region & centre])
        patch = np.isin(comp, touched[touched > 0])
        if box is not None:
            x0, y0, x1, y1 = box
            inside = np.zeros(patch.shape, bool)
            inside[y0:y1, x0:x1] = True
            patch &= inside
        region |= patch
        return region

    @staticmethod
    def _centre_lines(wide, width_px):
        """The medial axis of every wide patch, drawn width_px wide (inside the patch); bool mask."""
        from skimage.morphology import skeletonize
        out = np.zeros(wide.shape, bool)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(wide.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h = (int(v) for v in stats[k, :4])
            sub = comp[y:y + h, x:x + w] == k
            line = skeletonize(np.pad(sub, 1))[1:-1, 1:-1]
            out[y:y + h, x:x + w] |= line
        if width_px > 1 and out.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_px, width_px))
            out = cv2.dilate(out.astype(np.uint8), k).astype(bool) & wide
        return out

    def _toggle_sam_width_limit(self):
        """[0] / the dial: the width limit on / off (the value stays)."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        cfg.SAM_WIDTH_LIMIT_ON = not cfg.SAM_WIDTH_LIMIT_ON
        what = (f"Width limit ON ({cfg.SAM_ACCEPT_MAX_WIDTH_PX} px)" if cfg.SAM_WIDTH_LIMIT_ON
                else "Width limit OFF (wide cracks shown whole)")
        if state["active"] and state.get("prob") is not None:
            self._recut_sam_map(what)
        else:
            print(f" [SAM CLICK] {what}.")

    def _toggle_sam_centre_lines(self):
        """The dial: wider parts shown as their centre line, or not at all."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        cfg.SAM_WIDE_AS_CENTERLINE = not cfg.SAM_WIDE_AS_CENTERLINE
        what = (f"Wide parts: centre line {cfg.SAM_WIDE_CENTERLINE_PX} px" if cfg.SAM_WIDE_AS_CENTERLINE
                else "Wide parts: hidden")
        if state["active"] and state.get("prob") is not None:
            self._recut_sam_map(what)

    def _set_sam_class_threshold(self, which, value):
        """The dial: mean probability from which a crack is orange ('orange') / dark yellow ('dark')."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        value = round(float(min(0.99, max(0.0, value))), 2)
        if which == "orange":
            cfg.SAM_ACCEPT_AUTO_MIN_MEAN = max(value, cfg.SAM_ACCEPT_DARK_MIN_MEAN)
        else:
            cfg.SAM_ACCEPT_DARK_MIN_MEAN = min(value, cfg.SAM_ACCEPT_AUTO_MIN_MEAN)
        if state["active"] and state.get("prob") is not None:
            state["overlay"] = None
            self._classify_sam_regions()
            self.refresh_zoom_viewport()
        return cfg.SAM_ACCEPT_AUTO_MIN_MEAN if which == "orange" else cfg.SAM_ACCEPT_DARK_MIN_MEAN

    def _recut_sam_map(self, what):
        """Cuts the map again with the current threshold and maximum width; accepted cracks stay, discarded ones
        stay discarded."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        labels, stats, wide = self._cut_sam_map(state["prob"], cfg.SAM_ACCEPT_LOW_THRESHOLD)
        state.update(labels=labels, stats=stats, wide_mask=wide, hover=0, hover_contour=None, overlay=None)
        gone = set()
        mask = state.get("discarded_mask")
        if mask is not None and mask.any():
            n = int(labels.max()) + 1
            hit = np.bincount(labels[mask], minlength=n)
            gone = {int(k) for k in np.nonzero(hit >= 0.5 * np.maximum(stats[:n, cv2.CC_STAT_AREA], 1))[0] if k}
        state["discarded"] = gone
        self._classify_sam_regions()
        c = self._sam_class_counts()
        print(f" [SAM CLICK] {what}: {c[CLASS_SURE]} orange, {c[CLASS_DARK]} dark-yellow, "
              f"{c[CLASS_LIGHT]} light-yellow crack(s); the accepted cracks are unchanged.")
        self.refresh_zoom_viewport()

    def _set_sam_max_width(self, value):
        """The widest crack (photo px) still coloured; 0 = no limit. Returns the value in use."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        value = int(max(0, min(cfg.SAM_MAX_WIDTH_LIMIT_PX, round(value))))
        if value == cfg.SAM_ACCEPT_MAX_WIDTH_PX:
            return value
        cfg.SAM_ACCEPT_MAX_WIDTH_PX = value
        cfg.SAM_WIDTH_LIMIT_ON = value > 0 or cfg.SAM_WIDTH_LIMIT_ON
        if state["active"] and state.get("prob") is not None:
            self._recut_sam_map(f"Max width {value} px" if value else "No width limit")
        return value

    def recompute_sam_map_here(self):
        """Code 158: the crack map of the photo on screen computed again by the model on this computer (it replaces
        the cached one, e.g. a map from Leonardo that looks wrong); SAM click to accept reopens on it."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        if not cfg.CURRENT_IMAGE_PATH or getattr(cfg, "queue_idle", False):
            print(" [SAM] No photo on screen.")
            return False
        settings = self._usable_sam_settings()
        if not settings:
            return False
        print(f" [SAM] Computing the crack map of {os.path.basename(cfg.CURRENT_IMAGE_PATH)} here (about 1.5-4 min "
              "for a 4K photo; [Esc] cancels) ...")
        prob = self._sam_run_model_on_photo(settings)
        if prob is None:
            print(" [SAM] Not computed: the previous map stays.")
            return False
        from sam_autoseg import write_cached_probability
        write_cached_probability(self._sam_probability_cache_path(settings), prob)
        print(" [SAM] New crack map saved: it replaces the previous one for this photo.")
        if state["active"]:
            sticky = state.get("sticky")
            state["unsaved"] and self._save_sam_changes_if_any()
            self._exit_sam_accept("new crack map")
            state["sticky"] = sticky
            self._start_sam_accept()
        else:
            self.refresh_zoom_viewport()
        return True

    def sam_threshold_feedback(self):
        """(threshold, {class: regions}, coloured share of the photo) for the dial."""
        state = self.cfg.sam_accept_state
        if not state["active"] or state.get("classes") is None:
            return self.cfg.SAM_ACCEPT_LOW_THRESHOLD, None, 0.0
        colour, mask = self._sam_overlay()
        share = float(cv2.countNonZero(mask)) / mask.size if mask is not None else 0.0
        return self.cfg.SAM_ACCEPT_LOW_THRESHOLD, self._sam_class_counts(), share

    def _toggle_sam_exclude(self):
        state = self.cfg.sam_accept_state
        state["exclude"] = not state.get("exclude")
        if state["exclude"]:
            state["cut"], state["cut_path"] = False, None
        print(" [SAM CLICK] Exclude crack " + (
            "ON: click a crack (accepted ones too) to remove it -- also from the JSON and the PNG masks; a click on a "
            "coloured crack drops it. X or the button again to go back to accepting." if state["exclude"]
            else "off: clicks accept cracks again."))
        self.refresh_zoom_viewport()

    # --- Cut the map ------------------------------------------------------------------------------

    def _toggle_sam_cut(self):
        """Code 161 (K): [Cut map] on / off. While on, a line dragged with the mouse cuts the crack map along it, so
        a crack joined to something else (tree branches, a cable, a stain) becomes two regions and a click takes only
        the crack."""
        state = self.cfg.sam_accept_state
        state["cut"] = not state.get("cut")
        state["cut_path"] = None
        if state["cut"]:
            state["exclude"] = False
        print(" [SAM CLICK] Cut map " + (
            "ON: drag a line across the coloured map where the crack must be separated (e.g. from tree branches); "
            "each line cuts it there. K or the button again to go back to accepting." if state["cut"]
            else "off: clicks accept cracks again."))
        self.refresh_zoom_viewport()

    def _handle_cut_drag(self, event, x, y):
        """Press, drag along the line to cut, release. True when the event was used here."""
        state = self.cfg.sam_accept_state
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["photo"] != self.cfg.CURRENT_IMAGE_PATH:
                self._exit_sam_accept("the photo changed")
                return True
            state["cut_path"] = [(int(x), int(y))]
            return True
        path = state.get("cut_path")
        if path is None:
            return event == cv2.EVENT_MOUSEMOVE and self._update_cut_hover(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            lx, ly = path[-1]
            if abs(x - lx) + abs(y - ly) >= 3:
                path.append((int(x), int(y)))
                self.refresh_zoom_viewport()
            return True
        if event == cv2.EVENT_LBUTTONUP:
            path.append((int(x), int(y)))
            state["cut_path"] = None
            xs, ys = [p[0] for p in path], [p[1] for p in path]
            if max(xs) - min(xs) + max(ys) - min(ys) < 4:
                print(" [SAM CLICK] Cut map: drag a line (press, move, release) across the map to cut it.")
                self.refresh_zoom_viewport()
                return True
            self._apply_sam_cut(path)
            return True
        return False

    def _update_cut_hover(self, x, y):
        """In cut mode the mouse only moves: the width label still follows it."""
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self._update_sam_width_probe(x, y, int(real_x), int(real_y)):
            self.refresh_zoom_viewport()
        return True

    def _apply_sam_cut(self, screen_points):
        """Clears the map along the polyline (screen points) so that it is cut there: at least SAM_CUT_WIDTH_PX photo
        px wide and never thinner than 3 screen px, so that 8-connected regions are really split. The accepted
        cracks are not touched (the eraser cuts those). Returns the number of map pixels cleared."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        prob = state.get("prob")
        if prob is None:
            return 0
        pts = [self.transform_window_to_real_coords(px, py) for px, py in screen_points]
        ax, _ = self.transform_window_to_real_coords(0, 0)
        bx, _ = self.transform_window_to_real_coords(3, 0)
        thick = int(max(int(getattr(cfg, "SAM_CUT_WIDTH_PX", 3)), np.ceil(abs(bx - ax)), 2))
        line = np.zeros(prob.shape[:2], np.uint8)
        poly = np.array([[int(round(px)), int(round(py))] for px, py in pts], np.int32).reshape(-1, 1, 2)
        cv2.polylines(line, [poly], False, 1, thickness=thick, lineType=cv2.LINE_8)
        thr = max(1, int(round(float(cfg.SAM_ACCEPT_LOW_THRESHOLD) * 255)))
        hit = (line > 0) & (prob > 0)
        cleared = int(np.count_nonzero(hit & (prob >= thr)))
        if not hit.any():
            print(" [SAM CLICK] Cut map: the line does not cross the map.")
            self.refresh_zoom_viewport()
            return 0
        if not state.get("prob_private"):
            state["prob"], state["prob_private"] = prob.copy(), True
        state["prob"][hit] = 0
        cuts = state.get("cut_mask")
        state["cut_mask"] = hit if cuts is None else (cuts | hit)
        before = int(state["labels"].max())
        self._recut_sam_map("Cut")
        after = int(state["labels"].max())
        print(f" [SAM CLICK] Cut map: {cleared} map px cleared along the line ({thick} px wide); "
              f"{after - before:+d} region(s). Click the part to keep; K to go back to accepting.")
        return cleared

    def _draw_sam_cut_path(self, win_out):
        state = self.cfg.sam_accept_state
        path = state.get("cut_path")
        if not path or len(path) < 2 or win_out is None or not hasattr(win_out, "shape"):
            return
        pts = np.array(path, np.int32).reshape(-1, 1, 2)
        cv2.polylines(win_out, [pts], False, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(win_out, [pts], False, CUT_LINE_COLOR, 2, cv2.LINE_AA)

    def _toggle_sam_colours(self, which):
        cfg = self.cfg
        if which == "all":
            show = not (cfg.SAM_ACCEPT_SHOW_YELLOW or cfg.SAM_ACCEPT_SHOW_ORANGE)
            cfg.SAM_ACCEPT_SHOW_YELLOW = cfg.SAM_ACCEPT_SHOW_ORANGE = show
            print(" [SAM CLICK] Every SAM colour " + ("shown." if show else "hidden: H shows them again."))
        elif which == "yellow":
            cfg.SAM_ACCEPT_SHOW_YELLOW = not cfg.SAM_ACCEPT_SHOW_YELLOW
            print(" [SAM CLICK] Yellow cracks " + ("shown." if cfg.SAM_ACCEPT_SHOW_YELLOW else "hidden: G shows them."))
        else:
            cfg.SAM_ACCEPT_SHOW_ORANGE = not cfg.SAM_ACCEPT_SHOW_ORANGE
            print(" [SAM CLICK] Orange cracks " + ("shown." if cfg.SAM_ACCEPT_SHOW_ORANGE else "hidden: A shows them."))
        cfg.sam_accept_state["overlay"] = None
        self.refresh_zoom_viewport()

    def _toggle_sam_overlay(self):
        """[H] / code 143: hides / shows every colour."""
        self._toggle_sam_colours("all")

    def _toggle_sam_candidates(self):
        """Code 139 (Shift+click on the button): as H."""
        if not self.cfg.sam_accept_state["active"]:
            print(" [SAM CLICK] Turn SAM click to accept on first.")
            return
        self._toggle_sam_colours("all")

    def _discard_sam_class(self, cls):
        """L / D: every crack of that colour leaves the screen and is no longer proposed on this photo."""
        state = self.cfg.sam_accept_state
        gone = self._sam_labels_of_class(cls)
        if not gone:
            print(f" [SAM CLICK] No {CLASS_NAMES[cls]} crack left on this photo.")
            return 0
        self._remember_discarded_pixels(gone)
        state["discarded"].update(gone)
        state["done_labels"].update(gone)
        state["overlay"] = None
        if state.get("hover") in gone:
            state["hover"], state["hover_contour"] = 0, None
        print(f" [SAM CLICK] {len(gone)} {CLASS_NAMES[cls]} crack(s) discarded (nothing saved for them; turning the "
              "mode off and on again shows them back).")
        self.refresh_zoom_viewport()
        return len(gone)

    # --- Drawing ------------------------------------------------------------------------------

    def _sam_overlay(self):
        """(colour image, mask) of the whole photo, cached: rebuilt only when the cracks, the discarded regions or
        the shown colours change -- so moving the mouse and panning stay fast on 4K photos."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        if state.get("cracks_signature") != self._cracks_signature():
            self._refresh_sam_done_labels()
        key = (cfg.SAM_ACCEPT_SHOW_YELLOW, cfg.SAM_ACCEPT_SHOW_ORANGE)
        cached = state.get("overlay")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        classes = state["classes"].copy()
        done = state.get("done_labels") or set()
        if done:
            classes[list(done)] = CLASS_NONE
        if not cfg.SAM_ACCEPT_SHOW_YELLOW:
            classes[(classes == CLASS_LIGHT) | (classes == CLASS_DARK)] = CLASS_NONE
        if not cfg.SAM_ACCEPT_SHOW_ORANGE:
            classes[classes == CLASS_SURE] = CLASS_NONE
        palette = np.zeros((max(4, len(CLASS_COLORS) + 1), 3), np.uint8)
        for cls, color in CLASS_COLORS.items():
            palette[cls] = color
        cls_map = classes[state["labels"]]
        mask = (cls_map > 0).astype(np.uint8) * 255
        colour = palette[cls_map] if mask.any() else None
        state["overlay"] = (key, colour, mask)
        return colour, mask

    def _hover_contour(self):
        """Outline of the region under the mouse, in photo coordinates (computed once per region)."""
        state = self.cfg.sam_accept_state
        hover = state.get("hover", 0)
        cached = state.get("hover_contour")
        if cached is not None and cached[0] == hover:
            return cached[1]
        contours = []
        stats = state.get("stats")
        if hover and stats is not None and hover < len(stats):
            bx, by, bw, bh = (int(v) for v in stats[hover, :4])
            sub = (state["labels"][by:by + bh, bx:bx + bw] == hover).astype(np.uint8)
            contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(bx, by))
        state["hover_contour"] = (hover, contours)
        return contours

    def _paint_sam_candidates(self, scene, region):
        """The cracks the model sees, coloured by class, in scene (the crop of the photo given by
        region = (y0, y1, x0, x1)), under the cracks drawn afterwards; the one under the mouse outlined in cyan."""
        state = self.cfg.sam_accept_state
        if not state["active"] or state["prob"] is None or state.get("classes") is None:
            return
        y0, y1, x0, x1 = region
        if state["labels"][y0:y1, x0:x1].shape != scene.shape[:2]:
            return
        colour, mask = self._sam_overlay()
        if colour is not None:
            m = mask[y0:y1, x0:x1]
            if cv2.countNonZero(m):   # OpenCV blend + masked copy: ~15 ms on a whole 4K photo
                blended = cv2.addWeighted(scene, 0.3, colour[y0:y1, x0:x1], 0.7, 0)
                cv2.copyTo(blended, m, scene)
        hover = state.get("hover", 0)
        if hover and hover not in (state.get("done_labels") or set()):
            contours = self._hover_contour()
            if contours:
                outline = EXCLUDE_OUTLINE if state.get("exclude") else HOVER_OUTLINE
                cv2.drawContours(scene, contours, -1, outline, 2, offset=(-x0, -y0))

    def _handle_sam_accept_key(self, key_clean):
        """Another command ends the mode (the button itself toggles it); moving to another photo only closes it
        for this one. Never consumes the key."""
        state = self.cfg.sam_accept_state
        if state["active"] and (key_clean in SAVING_NAVIGATION_KEYS or key_clean in (ord('s'), ord('S'))):
            state["unsaved"] = False          # the photo is saved by the key itself
        if not state["active"] or key_clean in SAM_ACCEPT_KEEP_KEYS:
            return False
        sticky = state["sticky"]
        if key_clean in NAVIGATION_KEYS and key_clean not in SAVING_NAVIGATION_KEYS:
            self._save_sam_changes_if_any()   # e.g. Next without saving: the accepted cracks are kept
        if key_clean in NAVIGATION_KEYS:
            self._exit_sam_accept("next photo -- it reopens there" if sticky else None)
        elif sticky:
            self._exit_sam_accept("paused for the tool just chosen -- press the button to come back to it here; "
                                  "the next photo reopens it")
        else:
            self._exit_sam_accept("another command was chosen")
        state["sticky"] = sticky
        return False

    def _resume_sam_accept_for_new_photo(self):
        """Called when a photo is opened: reopens the mode if it was on for the previous photo."""
        state = self.cfg.sam_accept_state
        if state.get("sticky") and not state["active"]:
            self._start_sam_accept()

    def _draw_sam_accept_label(self, win_out):
        state, cfg = self.cfg.sam_accept_state, self.cfg
        if not state["active"]:
            return
        self._draw_exclude_window(win_out)
        self._draw_sam_cut_path(win_out)
        self._draw_sam_width_tag(win_out)
        large = cfg.hud_large_size
        y = (135 if large else 120) + 8 * (45 if large else 30)
        c = self._sam_class_counts()
        hidden = []
        if not cfg.SAM_ACCEPT_SHOW_ORANGE:
            hidden.append("orange")
        if not cfg.SAM_ACCEPT_SHOW_YELLOW:
            hidden.append("yellow")
        extra = f" -- hidden: {', '.join(hidden)}" if hidden else ""
        if state.get("unsaved"):
            extra += " -- NOT SAVED YET (Enter / S)"
        if state.get("exclude"):
            extra = (" -- EXCLUDE: click a crack, or drag a window (10-12000 px) to remove all inside "
                     "(X: back to accepting)" + extra)
        if state.get("cut"):
            extra = " -- CUT MAP: drag a line across the map to split it (K: back to accepting)" + extra
        self._put_hud_text(win_out, f"SAM CLICK TO ACCEPT: {state['clicks']} accepted | orange {c[CLASS_SURE]}, dark "
                                    f"yellow {c[CLASS_DARK]}, light yellow {c[CLASS_LIGHT]} | threshold {cfg.SAM_ACCEPT_LOW_THRESHOLD:.2f} "
                                    f"([ ]) | click or drag a window to accept, all 1/2/3/4, "
                                    f"H/G/A colours, L/D discard, X exclude{extra}", (15, y), (255, 160, 0))

    # --- Accept every orange crack ------------------------------------------------------------

    def _accept_all_sam_proposals(self):
        """Code 142 (Ctrl+click on the button, 1): every orange crack still on screen."""
        return self._accept_sam_classes((CLASS_SURE,))

    def _accept_sam_classes(self, classes):
        """Every crack of these colours still on screen becomes cracks of the app, one Undo step:
        1 / Ctrl+click = orange, 2 = dark yellow, 3 = light yellow, 4 = all of them."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        if not state["active"]:
            print(" [SAM CLICK] Turn SAM click to accept on first.")
            return 0
        names = " / ".join(CLASS_NAMES[c] for c in classes)
        left = sorted(set().union(*(self._sam_labels_of_class(c) for c in classes)))
        if not left:
            print(f" [SAM CLICK] No {names} crack left on this photo.")
            return 0
        prob, labels = state["prob"], state["labels"]
        region = self._expand_to_wide(np.isin(labels, left))
        region &= self._sam_existing_coverage([f for f in cfg.saved_cracks if f.get('active', True)]) == 0
        shapes = mask_to_crack_shapes(region.astype(np.uint8) * 255, cfg.SAM_ACCEPT_MIN_LENGTH_PX, cfg.SAM_MAX_CRACKS)
        new_cracks = [self._sam_crack_from_shape(line, reg, x0, y0, prob) for line, reg, x0, y0 in shapes]
        for crack in new_cracks:
            crack['source'] = SAM_ACCEPT_SOURCE   # accepted by the operator
            crack.pop('uncertain_manual', None)
        state["done_labels"].update(left)
        state["overlay"] = None
        if not new_cracks:
            return 0
        before = self._snapshot_crack_edit_state()
        cfg.saved_cracks.extend(new_cracks)
        self._commit_crack_edit(before)
        state["clicks"] += len(left)
        print(f" [SAM CLICK] {len(left)} {names} crack(s) accepted: {len(new_cracks)} crack line(s) "
              "([U] undoes them; Exclude crack removes single ones).")
        self.recalculate_masks()
        self._sam_changed("the accepted proposals")
        self.refresh_zoom_viewport()
        return len(new_cracks)

    def _confirm_remove_automatic_sam_cracks(self, count):
        """True = remove them. Default: terminal input(); the GUI asks with a dialog."""
        try:
            answer = input(f"[SAM CLICK] This photo has {count} crack(s) added automatically by SAM that you never "
                           "clicked. Remove them? [y/N]: ")
        except Exception:
            return False
        return answer.strip().lower() in ("y", "s")

    def _offer_removal_of_automatic_sam_cracks(self):
        """A photo saved with v2.0.0's automatic part (or with [SAM auto-segment]) keeps cracks of "source": "sam"
        that the operator never confirmed: asked once per photo whether to remove them."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        asked = state.setdefault("asked_cleanup", set())
        photo = os.path.basename(cfg.CURRENT_IMAGE_PATH)
        auto = [c for c in cfg.saved_cracks if c.get('active', True) and c.get('source') == "sam"]
        if not auto or photo in asked:
            return 0
        asked.add(photo)
        if not self._confirm_remove_automatic_sam_cracks(len(auto)):
            return 0
        before = self._snapshot_crack_edit_state()
        cfg.saved_cracks[:] = [c for c in cfg.saved_cracks if not (c.get('active', True) and c.get('source') == "sam")]
        self._commit_crack_edit(before)
        self.recalculate_masks()
        self._sam_changed("the removal of the automatic SAM cracks")
        print(f" [SAM CLICK] {len(auto)} automatic SAM crack(s) removed ([U] brings them back); JSON and masks updated.")
        return len(auto)

    # --- Click ------------------------------------------------------------------------------

    def _handle_sam_accept_mouse(self, event, x, y):
        state = self.cfg.sam_accept_state
        if state.get("cut") and self._handle_cut_drag(event, x, y):
            return
        if self._handle_window_drag(event, x, y):
            return
        if event == cv2.EVENT_MOUSEMOVE:
            if state["photo"] == self.cfg.CURRENT_IMAGE_PATH:
                real_x, real_y = self.transform_window_to_real_coords(x, y)
                changed = self._update_sam_hover(int(real_x), int(real_y))
                changed = self._update_sam_width_probe(x, y, int(real_x), int(real_y)) or changed
                if changed:
                    self.refresh_zoom_viewport()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["photo"] != self.cfg.CURRENT_IMAGE_PATH:
            self._exit_sam_accept("the photo changed")
            return
        self._sam_click(x, y)

    def _sam_click(self, x, y):
        """A click (or a window smaller than SAM_WINDOW_MIN_BOX_PX): exclude, delete by number, or accept."""
        state = self.cfg.sam_accept_state
        if state.get("exclude"):
            self._exclude_at(x, y)
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self.cfg.show_markers and self._delete_crack_by_number(x, y):
            # v2.0.1: with the numbers shown ([Space]), a click on a crack's number deletes it, as with the crack
            # tool; saved at once like every change in this mode, and its region is coloured again.
            self._refresh_sam_done_labels()
            self._sam_changed("the deleted crack")
            print(" [SAM CLICK] Crack deleted ([U] brings it back).")
            self.refresh_zoom_viewport()
            return
        if self._accept_sam_crack_at(int(real_x), int(real_y)) and state.get("hover"):
            state["done_labels"].add(state["hover"])
        state["hover"], state["hover_contour"] = 0, None
        self.refresh_zoom_viewport()

    # --- Windows: accept / exclude everything inside ---------------------------------------------

    def _window_drag_rect(self, x, y):
        """The window from the press point to (x, y), each side at most SAM_WINDOW_MAX_BOX_PX screen pixels."""
        x0, y0 = self.cfg.sam_accept_state["window_drag"][:2]
        limit = int(self.cfg.SAM_WINDOW_MAX_BOX_PX)
        x1 = x0 + max(-limit, min(limit, x - x0))
        y1 = y0 + max(-limit, min(limit, y - y0))
        return x0, y0, x1, y1

    def _handle_window_drag(self, event, x, y):
        """Press and drag a window (10x10 to 12000x12000 screen px): on release, with [Exclude crack] everything inside
        is removed, otherwise the coloured cracks inside it are accepted (only the shown colours, only the part
        inside). A press without a real drag is a click. True when the event was used here."""
        state = self.cfg.sam_accept_state
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["photo"] != self.cfg.CURRENT_IMAGE_PATH:
                return False
            state["window_drag"] = (x, y, x, y)
            return True
        if state.get("window_drag") is None:
            return False
        if event == cv2.EVENT_MOUSEMOVE:
            state["window_drag"] = self._window_drag_rect(x, y)
            self.refresh_zoom_viewport()
            return True
        if event == cv2.EVENT_LBUTTONUP:
            x0, y0, x1, y1 = self._window_drag_rect(x, y)
            state["window_drag"] = None
            small = int(self.cfg.SAM_WINDOW_MIN_BOX_PX)
            if abs(x1 - x0) < small or abs(y1 - y0) < small:
                self._sam_click(x0, y0)            # no real window: a click
            elif state.get("exclude"):
                self._exclude_in_window(x0, y0, x1, y1)
            else:
                self._accept_in_window(x0, y0, x1, y1)
            return True
        return False

    def _window_to_photo(self, wx0, wy0, wx1, wy1):
        """(x0, y0, x1, y1) on the photo of a screen window, clipped; None when empty."""
        ax, ay = self.transform_window_to_real_coords(min(wx0, wx1), min(wy0, wy1))
        bx, by = self.transform_window_to_real_coords(max(wx0, wx1), max(wy0, wy1))
        h, w = self.cfg.sam_accept_state["labels"].shape
        rx0, ry0 = max(0, int(np.floor(ax))), max(0, int(np.floor(ay)))
        rx1, ry1 = min(w, int(np.ceil(bx)) + 1), min(h, int(np.ceil(by)) + 1)
        return (rx0, ry0, rx1, ry1) if rx0 < rx1 and ry0 < ry1 else None

    def _accept_in_window(self, wx0, wy0, wx1, wy1):
        """The coloured cracks inside the window (shown colours only, the part inside) become cracks, one Undo."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        box = self._window_to_photo(wx0, wy0, wx1, wy1)
        if box is None:
            return 0
        rx0, ry0, rx1, ry1 = box
        shown = []
        if cfg.SAM_ACCEPT_SHOW_ORANGE:
            shown.append(CLASS_SURE)
        if cfg.SAM_ACCEPT_SHOW_YELLOW:
            shown += [CLASS_DARK, CLASS_LIGHT]
        wanted = sorted(set().union(*(self._sam_labels_of_class(c) for c in shown))) if shown else []
        sub_labels = state["labels"][ry0:ry1, rx0:rx1]
        sub = np.isin(sub_labels, wanted) if wanted else np.zeros(sub_labels.shape, bool)
        if not sub.any():
            print(" [SAM CLICK] No coloured crack inside the window.")
            self.refresh_zoom_viewport()
            return 0
        region = np.zeros(state["labels"].shape, bool)
        region[ry0:ry1, rx0:rx1] = sub
        self._expand_to_wide(region, (rx0, ry0, rx1, ry1))
        region &= self._sam_existing_coverage([f for f in cfg.saved_cracks if f.get('active', True)]) == 0
        prob = state["prob"]
        shapes = mask_to_crack_shapes(region.astype(np.uint8) * 255, cfg.SAM_ACCEPT_MIN_LENGTH_PX, cfg.SAM_MAX_CRACKS)
        new_cracks = [self._sam_crack_from_shape(line, reg, x0, y0, prob) for line, reg, x0, y0 in shapes]
        for crack in new_cracks:
            crack['source'] = SAM_ACCEPT_SOURCE
            crack.pop('uncertain_manual', None)
        if not new_cracks:
            print(" [SAM CLICK] The coloured parts inside the window are too short to become cracks.")
            self.refresh_zoom_viewport()
            return 0
        before = self._snapshot_crack_edit_state()
        cfg.saved_cracks.extend(new_cracks)
        self._commit_crack_edit(before)
        state["clicks"] += len(new_cracks)
        # the accepted part is no longer coloured (whole regions may stay coloured outside the window)
        state["overlay"], state["hover"], state["hover_contour"] = None, 0, None
        if not state.get("prob_private"):
            state["prob"], state["prob_private"] = state["prob"].copy(), True
        state["labels"][region] = 0
        state["prob"][region] = 0
        print(f" [SAM CLICK] Window: {len(new_cracks)} crack line(s) accepted ([U] undoes them).")
        self.recalculate_masks()
        self._refresh_sam_done_labels()
        self._sam_changed("the cracks accepted with a window")
        self.refresh_zoom_viewport()
        return len(new_cracks)

    def _exclude_in_window(self, wx0, wy0, wx1, wy1):
        """Removes what is inside the window (screen corners): accepted cracks are cut there (saved at once, one
        Undo); the map of every colour is blanked there, so those parts are neither coloured nor accepted."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        box = self._window_to_photo(wx0, wy0, wx1, wy1)
        if box is None:
            return
        rx0, ry0, rx1, ry1 = box
        # 1. the accepted cracks: the points inside go, the rest stays in pieces (Eraser rules)
        from crack_cleanup import erased_runs
        before = self._snapshot_crack_edit_state()
        cracks, erased = cfg.saved_cracks, []
        for idx in range(len(cracks) - 1, -1, -1):
            crack = cracks[idx]
            if not crack.get('active', True) or not crack.get('path') or not self._crack_shown(crack):
                continue
            pts = np.asarray(crack['path'], dtype=np.float64).reshape(-1, 2)
            inside = (pts[:, 0] >= rx0) & (pts[:, 0] < rx1) & (pts[:, 1] >= ry0) & (pts[:, 1] < ry1)
            if inside.any():
                cracks[idx:idx + 1] = self._pieces_left_by_eraser(crack, inside)
                erased.extend(erased_runs(crack['path'], inside))
        # 2. the coloured cracks: their map is blanked inside the window
        coloured = int(np.count_nonzero(state["labels"][ry0:ry1, rx0:rx1]))
        if coloured:
            if not state.get("prob_private"):
                state["prob"], state["prob_private"] = state["prob"].copy(), True
            state["prob"][ry0:ry1, rx0:rx1] = 0
            state["labels"][ry0:ry1, rx0:rx1] = 0
            state["overlay"], state["hover"], state["hover_contour"] = None, 0, None
        if erased:
            cfg.cut_exclusions.extend(erased)
            self._commit_crack_edit(before)
            self.recalculate_masks()
            self._refresh_sam_done_labels()
            state["unsaved"] = False
            self._save_annotations_now("SAM CLICK", "the excluded window")
            print(f" [SAM CLICK] Window: {sum(len(r) for r in erased)} crack point(s) removed -- JSON and PNG masks "
                  "rewritten ([U] brings them back, then Enter / S saves again).")
        if coloured:
            print(f" [SAM CLICK] Window: the coloured cracks inside it are gone ({coloured} px).")
        if not erased and not coloured:
            print(" [SAM CLICK] Nothing inside the window.")
        self.refresh_zoom_viewport()

    def _draw_exclude_window(self, win_out):
        """The window being dragged: red = exclude, cyan = accept."""
        state = self.cfg.sam_accept_state
        drag = state.get("window_drag")
        if drag is None or win_out is None or not hasattr(win_out, "shape"):
            return
        x0, y0, x1, y1 = (int(v) for v in drag)
        color = EXCLUDE_OUTLINE if state.get("exclude") else HOVER_OUTLINE
        cv2.rectangle(win_out, (x0, y0), (x1, y1), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.rectangle(win_out, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(win_out, f"{abs(x1 - x0)}x{abs(y1 - y0)}", (min(x0, x1), max(12, min(y0, y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _crack_under_click(self, wx, wy, tolerance=10):
        """Index of the active crack whose line (or fill) passes within `tolerance` SCREEN pixels of the click,
        the nearest one; None if none."""
        rx, ry = self.transform_window_to_real_coords(wx, wy)
        rx2, _ = self.transform_window_to_real_coords(wx + tolerance, wy)
        tol = max(2.0, abs(rx2 - rx))
        best, best_d = None, None
        for idx, f in enumerate(self.cfg.saved_cracks):
            if not f.get('active', True) or not self._crack_shown(f) or not f.get('path'):
                continue
            pts = np.asarray(self._densify_path(f['path']) if len(f['path']) > 1 else f['path'], np.float32)
            d = float(np.min(np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)))
            half = 0.5 * max([float(seg.get('width_px', 0) or 0) for seg in (f.get('width_segments') or [])] or [0.0])
            d = max(0.0, d - half)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = idx, d
        return best

    def _exclude_at(self, wx, wy):
        """[Exclude crack]: removes the crack under the click (saved at once: JSON and PNG masks) or, on a
        coloured crack not accepted, drops it from the screen."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        idx = None
        if cfg.show_markers and self._delete_crack_by_number(wx, wy):
            idx = -1
        else:
            idx = self._crack_under_click(wx, wy)
            if idx is not None:
                removed = cfg.saved_cracks.pop(idx)
                cfg.action_history.append(('crack_delete', removed))
                self.recalculate_masks()
        if idx is not None:
            self._refresh_sam_done_labels()
            state["unsaved"] = False
            self._save_annotations_now("SAM CLICK", "the excluded crack")
            print(" [SAM CLICK] Crack excluded: removed from the JSON and the PNG masks ([U] brings it back, then "
                  "Enter / S saves it again).")
        elif state.get("hover") and state["hover"] not in state["done_labels"]:
            self._remember_discarded_pixels({state["hover"]})
            state["discarded"].add(state["hover"])
            state["done_labels"].add(state["hover"])
            state["overlay"] = None
            print(" [SAM CLICK] Coloured crack dropped (it was not saved).")
        else:
            print(" [SAM CLICK] No crack here.")
        state["hover"], state["hover_contour"] = 0, None
        self.refresh_zoom_viewport()

    def _delete_crack_by_number(self, wx, wy, tolerance=22):
        """v2.0.1: deletes the crack whose number box (start) or end dot is under the click, measured on the SCREEN
        (the boxes are 22 px on screen whatever the zoom; measured on the photo, a 4K photo seen whole needed a
        click within a few screen pixels). The nearest one wins. True when one was deleted."""
        best, best_d = None, None
        for idx, f in enumerate(self.cfg.saved_cracks):
            if not f.get('active', True) or not self._crack_shown(f) or self._crack_number_hidden(f):
                continue
            for key in ('start', 'end'):
                if key not in f:
                    continue
                px, py = self.transform_real_to_window_coords(int(f[key][0]), int(f[key][1]))
                d = max(abs(px - wx), abs(py - wy))
                if d <= tolerance and (best_d is None or d < best_d):
                    best, best_d = idx, d
        if best is None:
            return False
        removed = self.cfg.saved_cracks.pop(best)
        self.cfg.action_history.append(('crack_delete', removed))
        self.recalculate_masks()
        return True

    def _update_sam_hover(self, x, y):
        """Picks the map region a click at (x, y) would take, for the yellow preview. True when it changed."""
        state, cfg = self.cfg.sam_accept_state, self.cfg
        prob, labels = state["prob"], state.get("labels")
        if prob is None or labels is None:
            return False
        h, w = prob.shape
        r = cfg.SAM_ACCEPT_CLICK_RADIUS_PX
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        label = 0
        if x0 < x1 and y0 < y1:
            window = labels[y0:y1, x0:x1]
            if window.any():
                yy, xx = np.ogrid[y0:y1, x0:x1]
                done = state.get("done_labels") or set()
                near = ((yy - y) ** 2 + (xx - x) ** 2 <= r * r) & (window > 0)
                if done:
                    near &= ~np.isin(window, list(done))
                if near.any():
                    weights = np.where(near, prob[y0:y1, x0:x1].astype(np.int32) + 1, 0)
                    sy, sx = np.unravel_index(int(np.argmax(weights)), weights.shape)
                    label = int(window[sy, sx])
        if label == state.get("hover", 0):
            return False
        state["hover"] = label
        return True

    # --- Width under the mouse --------------------------------------------------------------------

    def _sam_width_at(self, x, y):
        """(width, capped) of the crack of the map under photo point (x, y) at the current threshold, in photo px:
        twice the depth of the crack (distance transform) at its centre near the mouse; capped = wider than could be
        measured. None when no crack pixel is within SAM_ACCEPT_CLICK_RADIUS_PX. Hidden wide parts count too (the
        width limit does not change what is measured)."""
        state, cfg = self.cfg.sam_accept_state, self.cfg
        prob = state.get("prob")
        if prob is None:
            return None
        h, w = prob.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        thr = max(1, int(round(float(cfg.SAM_ACCEPT_LOW_THRESHOLD) * 255)))
        r = max(3, int(cfg.SAM_ACCEPT_CLICK_RADIUS_PX))
        near = prob[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] >= thr
        if not near.any():
            return None
        limit = float(cfg.SAM_ACCEPT_MAX_WIDTH_PX or 40)
        half = int(max(160, 4 * limit))                       # crop: up to ~2 x half px measurable
        x0, y0, x1, y1 = max(0, x - half), max(0, y - half), min(w, x + half + 1), min(h, y + half + 1)
        crop = (prob[y0:y1, x0:x1] >= thr).astype(np.uint8)
        # the crack touching the mouse only (not a parallel crack or a patch nearby)
        _, comp = cv2.connectedComponents(crop, connectivity=8)
        cx, cy = x - x0, y - y0
        wy0, wy1, wx0, wx1 = max(0, cy - r), cy + r + 1, max(0, cx - r), cx + r + 1
        win = comp[wy0:wy1, wx0:wx1]
        yy, xx = np.nonzero(win)
        if not len(yy):
            return None
        k = int(np.argmin((yy + wy0 - cy) ** 2 + (xx + wx0 - cx) ** 2))
        lab = int(win[yy[k], xx[k]])
        mine = (comp == lab).astype(np.uint8)
        dist = cv2.distanceTransform(mine, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # widen the search until it contains the centre of the crack under the mouse
        s, depth = r, 0.0
        for _ in range(8):
            depth = float(dist[max(0, cy - s):cy + s + 1, max(0, cx - s):cx + s + 1].max())
            if depth + 1 <= s:
                break
            s = int(depth) + r
        width = max(1, int(round(2 * depth - 1)))
        capped = depth >= half - 1
        return width, capped

    def _update_sam_width_probe(self, sx, sy, x, y):
        """Keeps state["width_probe"] = (screen x, screen y, width, capped) for the label next to the mouse. True
        when the label must be drawn again (another value, or the mouse moved away from it)."""
        state, cfg = self.cfg.sam_accept_state, self.cfg
        old = state.get("width_probe")
        if not getattr(cfg, "SAM_SHOW_WIDTH", True):
            state["width_probe"] = None
            return old is not None
        try:
            found = self._sam_width_at(x, y)
        except Exception:
            found = None
        if found is None:
            state["width_probe"] = None
            return old is not None
        width, capped = found
        if old is not None and old[2] == width and old[3] == capped and abs(old[0] - sx) + abs(old[1] - sy) < 40:
            return False
        state["width_probe"] = (int(sx), int(sy), width, capped)
        return True

    def sam_width_text(self, width, capped):
        """('width 23 px', False) or ('width 152 px > 40: not a crack -- [I] ignore', True)."""
        limit = float(self.cfg.SAM_ACCEPT_MAX_WIDTH_PX or 0)
        value = f"{'>' if capped else ''}{width} px"
        if limit and width > limit:
            return (f"width {value} > {limit:g}: not a crack -- cover it with [I] ignore", True)
        return (f"width {value}", False)

    def _draw_sam_width_tag(self, win_out):
        """The width label next to the mouse (white; red with a hint above the width limit)."""
        state = self.cfg.sam_accept_state
        probe = state.get("width_probe")
        if not probe or win_out is None or not hasattr(win_out, "shape") or state.get("window_drag") is not None:
            return
        sx, sy, width, capped = probe
        text, wide = self.sam_width_text(width, capped)
        scale, thick = 0.55, 1
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        H, W = win_out.shape[:2]
        x = min(max(0, sx + 16), max(0, W - tw - 8))
        y = min(max(th + 6, sy - 14), H - 6)
        bg = (0, 0, 170) if wide else (30, 30, 30)
        cv2.rectangle(win_out, (x - 4, y - th - 5), (x + tw + 4, y + base + 2), bg, -1)
        cv2.putText(win_out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _accept_sam_crack_at(self, x, y):
        """Adds the cracks of the map region under (x, y). True when something was added."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        prob = state["prob"]
        keep = [f for f in cfg.saved_cracks if f.get('active', True)]
        blocked = self._sam_existing_coverage(keep)
        wide = state.get("wide_mask")
        if wide is not None:   # v2.0.1: the parts wider than SAM_ACCEPT_MAX_WIDTH_PX are never taken
            blocked = wide if blocked is None else (blocked | wide)
        region, seed = accept_region(prob, x, y, cfg.SAM_ACCEPT_LOW_THRESHOLD, cfg.SAM_ACCEPT_CLICK_RADIUS_PX, blocked)
        if region is None:
            if seed == "taken":
                print(" [SAM CLICK] That crack is already there.")
            else:
                print(f" [SAM CLICK] The model sees no crack within {cfg.SAM_ACCEPT_CLICK_RADIUS_PX} px of the click: "
                      "click closer, or trace it by hand.")
            return False
        region = (self._expand_to_wide(region > 0) & (self._sam_existing_coverage(keep) == 0)).astype(np.uint8) * 255
        area = int((region > 0).sum())
        if area > cfg.SAM_ACCEPT_MAX_AREA_FRACTION * prob.size:
            print(f" [SAM CLICK] The region under the click covers {100.0 * area / prob.size:.0f}% of the photo: "
                  "not a crack. Raise SAM_ACCEPT_LOW_THRESHOLD or trace it by hand.")
            return False
        shapes = mask_to_crack_shapes(region, cfg.SAM_ACCEPT_MIN_LENGTH_PX, cfg.SAM_MAX_CRACKS)
        if not shapes:
            print(" [SAM CLICK] The crack under the click is too short to keep: trace it by hand.")
            return False
        new_cracks = [self._sam_crack_from_shape(line, reg, x0, y0, prob) for line, reg, x0, y0 in shapes]
        for crack in new_cracks:
            crack['source'] = SAM_ACCEPT_SOURCE
            crack.pop('uncertain_manual', None)   # the operator confirmed it
        before = self._snapshot_crack_edit_state()
        cfg.saved_cracks.extend(new_cracks)
        self._commit_crack_edit(before)
        state["clicks"] += 1
        length = sum(len(c['path']) for c in new_cracks)
        print(f" [SAM CLICK] Accepted: {len(new_cracks)} crack line(s), {length} px long.")
        self.recalculate_masks()
        self._sam_changed("the last click")
        return True
