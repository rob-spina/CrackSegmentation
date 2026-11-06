"""sam_accept.py
v2.0.0 [SAM click to accept] (right sidebar AI section, Tools menu, code 138), mixed into CrackSegmentation.
Interactive use of the fine-tuned SAM crack model, with no false positives to delete:
  - pressing the button computes the model's crack map of the photo once (or reuses the one [SAM auto-segment] or
    sam_batch.py already computed) and turns the mode on; nothing is added yet;
  - every click near a crack accepts it: the connected part of the map around the click (down to the low threshold
    SAM_ACCEPT_LOW_THRESHOLD, so faint stretches come too) becomes cracks of the app, with SAM's exact shape as the
    width ("source": "sam_click"). The operator confirmed them, so they are never saved as uncertain;
  - JSON and binary masks are written after every click; [U] undoes the last click; the operator's cracks and the
    ignore regions are left alone;
  - automatic part: on every photo the mode opens on, the cracks the model is sure of (mean probability of the
    region >= SAM_ACCEPT_AUTO_MIN_MEAN) are added at once ("source": "sam", one Undo step); the others are left to
    the operator. Skipped on a photo that already has SAM cracks (it was done before);
  - moving the mouse over a remaining crack shows its probability map: dark yellow where the model is surer,
    light yellow where it is less sure -- exactly what a click would accept. Click to accept it, or trace / edit it
    with the usual tools (the mode pauses and comes back on the next photo). Shift+click on the button (code 139)
    shows / hides the map of every remaining crack at once;
  - the mode stays on from photo to photo (Enter / Q / Go to image) until the button is pressed again.
"""
import os
import time

import cv2
import numpy as np

from crack_relabel import RELABEL_KEEP_KEYS, RELABEL_TARGET_BY_CODE
from review_flags import REVIEW_CODE
from sam_autoseg import mask_to_crack_shapes

SAM_ACCEPT_CODE = 138
SAM_CANDIDATES_CODE = 139   # show / hide the cracks the model sees (Shift+click on the button, Tools menu)
CANDIDATE_COLOR = (0, 165, 255)   # BGR orange, stronger where the model is surer (Shift+click: every crack)
HOVER_LIGHT = (170, 255, 255)     # BGR light yellow: the model is not very sure
HOVER_DARK = (0, 175, 235)        # BGR dark yellow: the model is sure
# Keys that move to another photo: they close the mode for this photo but it reopens on the next one.
NAVIGATION_KEYS = frozenset([13, 10, ord('q'), ord('Q'), 29, 31])
SAM_ACCEPT_SOURCE = "sam_click"
# Keys that leave the mode on: view keys, Undo/Redo, Save, the overlay toggles and the button itself.
SAM_ACCEPT_KEEP_KEYS = (RELABEL_KEEP_KEYS | frozenset(RELABEL_TARGET_BY_CODE)
                        | frozenset([ord('u'), ord('U'), ord('r'), ord('R'), ord('s'), ord('S'),
                                     130, 131, 133, SAM_ACCEPT_CODE, SAM_CANDIDATES_CODE, REVIEW_CODE]))


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

    # --- Mode -------------------------------------------------------------------------------

    def _toggle_sam_accept(self):
        """Code 138: turns the mode on (computing the crack map if needed) or off."""
        state = self.cfg.sam_accept_state
        if state["active"]:
            state["sticky"] = False
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
        low = (prob >= int(round(cfg.SAM_ACCEPT_LOW_THRESHOLD * 255))).astype(np.uint8)
        _, labels = cv2.connectedComponents(low, connectivity=8)   # for the hover preview
        state.update(active=True, photo=cfg.CURRENT_IMAGE_PATH, prob=prob, labels=labels, hover=0, clicks=0,
                     sticky=bool(cfg.SAM_ACCEPT_STAY_ON), done_labels=set())
        if cfg.SAM_ACCEPT_AUTO and not any(str(c.get('source', '')).startswith('sam') for c in cfg.saved_cracks):
            self._auto_accept_confident_cracks()
        print(" [SAM CLICK] On: move the mouse over a crack to see its map (dark yellow = the model is sure), click "
              "the REAL ones to accept them, or trace them with the usual tools. Shift+click on the button shows every "
              "remaining crack; [U] undoes; the mode stays on for the next photo; press the button to finish.")
        self.refresh_zoom_viewport()
        return True

    def _exit_sam_accept(self, reason=None):
        state = self.cfg.sam_accept_state
        if not state["active"]:
            return
        clicks = state["clicks"]
        state.update(active=False, photo=None, prob=None, labels=None, hover=0, clicks=0, done_labels=set())
        print(f" [SAM CLICK] Off{': ' + reason if reason else ''} ({clicks} crack(s) accepted).")
        self.refresh_zoom_viewport()

    def _toggle_sam_candidates(self):
        """Code 139: hides / shows the orange tint of the cracks the model sees."""
        cfg = self.cfg
        cfg.SAM_ACCEPT_SHOW_CANDIDATES = not cfg.SAM_ACCEPT_SHOW_CANDIDATES
        shown = "shown" if cfg.SAM_ACCEPT_SHOW_CANDIDATES else "hidden"
        if not cfg.sam_accept_state["active"]:
            print(f" [SAM CLICK] Model's cracks will be {shown} in SAM click to accept.")
        else:
            print(f" [SAM CLICK] Model's cracks {shown}.")
        self.refresh_zoom_viewport()

    def _paint_sam_candidates(self, scene, region):
        """Orange tint over the cracks the model sees (p >= SAM_ACCEPT_LOW_THRESHOLD), in scene (the crop of the
        photo given by region = (y0, y1, x0, x1)), under the cracks drawn afterwards."""
        state, cfg = self.cfg.sam_accept_state, self.cfg
        if not state["active"] or state["prob"] is None:
            return
        y0, y1, x0, x1 = region
        prob = state["prob"][y0:y1, x0:x1]
        if prob.shape != scene.shape[:2]:
            return
        labels = state.get("labels")
        done = state.get("done_labels") or set()
        if cfg.SAM_ACCEPT_SHOW_CANDIDATES:
            mask = prob >= int(round(cfg.SAM_ACCEPT_LOW_THRESHOLD * 255))
            if done and labels is not None:
                mask &= ~np.isin(labels[y0:y1, x0:x1], list(done))
            if mask.any():
                alpha = (0.25 + 0.5 * prob[mask].astype(np.float32) / 255.0)[:, None]
                color = np.array(CANDIDATE_COLOR, np.float32)
                scene[mask] = (scene[mask].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        hover = state.get("hover", 0)
        if hover and labels is not None:
            mask = labels[y0:y1, x0:x1] == hover
            if mask.any():
                low = cfg.SAM_ACCEPT_LOW_THRESHOLD * 255
                t = np.clip((prob[mask].astype(np.float32) - low) / max(255 - low, 1), 0, 1)[:, None]
                color = (np.array(HOVER_LIGHT, np.float32) * (1 - t) + np.array(HOVER_DARK, np.float32) * t)
                scene[mask] = (scene[mask].astype(np.float32) * 0.2 + color * 0.8).astype(np.uint8)

    def _handle_sam_accept_key(self, key_clean):
        """Another command ends the mode (the button itself toggles it); moving to another photo only closes it
        for this one. Never consumes the key."""
        state = self.cfg.sam_accept_state
        if not state["active"] or key_clean in SAM_ACCEPT_KEEP_KEYS:
            return False
        sticky = state["sticky"]
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
        state = self.cfg.sam_accept_state
        if not state["active"]:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 8 * (45 if large else 30)
        self._put_hud_text(win_out, f"SAM CLICK TO ACCEPT: click the real cracks ({state['clicks']} accepted)",
                           (15, y), (255, 160, 0))

    # --- Automatic part ---------------------------------------------------------------------

    def _auto_accept_confident_cracks(self):
        """Adds, as one Undo step, every map region whose mean probability reaches SAM_ACCEPT_AUTO_MIN_MEAN."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        prob, labels = state["prob"], state["labels"]
        n = int(labels.max()) + 1
        if n <= 1:
            return 0
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=n)
        sums = np.bincount(flat, weights=prob.ravel().astype(np.float64), minlength=n)
        means = sums / np.maximum(counts, 1) / 255.0
        ok = (means >= cfg.SAM_ACCEPT_AUTO_MIN_MEAN) & (counts >= cfg.SAM_MIN_AREA_PX) \
            & (counts <= cfg.SAM_ACCEPT_MAX_AREA_FRACTION * prob.size)
        ok[0] = False
        chosen = np.nonzero(ok)[0]
        if not len(chosen):
            print(" [SAM CLICK] No crack the model is sure of: they are all left to you.")
            return 0
        region = np.isin(labels, chosen)
        region &= self._sam_existing_coverage([f for f in cfg.saved_cracks if f.get('active', True)]) == 0
        shapes = mask_to_crack_shapes(region.astype(np.uint8) * 255, cfg.SAM_MIN_CRACK_LENGTH_PX, cfg.SAM_MAX_CRACKS)
        new_cracks = [self._sam_crack_from_shape(line, reg, x0, y0, prob) for line, reg, x0, y0 in shapes]
        for crack in new_cracks:
            crack.pop('uncertain_manual', None)   # sure by construction
        if not new_cracks:
            return 0
        before = self._snapshot_crack_edit_state()
        cfg.saved_cracks.extend(new_cracks)
        self._commit_crack_edit(before)
        state["done_labels"].update(int(k) for k in chosen)
        left = int((counts[1:] >= cfg.SAM_MIN_AREA_PX).sum()) - len(chosen)
        print(f" [SAM CLICK] {len(new_cracks)} crack line(s) the model is sure of added ([U] removes them); "
              f"{max(left, 0)} less sure region(s) left to you: move the mouse over them.")
        self.recalculate_masks()
        self._save_annotations_now("SAM CLICK", "the cracks the model is sure of")
        return len(new_cracks)

    # --- Click ------------------------------------------------------------------------------

    def _handle_sam_accept_mouse(self, event, x, y):
        state = self.cfg.sam_accept_state
        if event == cv2.EVENT_MOUSEMOVE:
            if state["photo"] == self.cfg.CURRENT_IMAGE_PATH:
                real_x, real_y = self.transform_window_to_real_coords(x, y)
                if self._update_sam_hover(int(real_x), int(real_y)):
                    self.refresh_zoom_viewport()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["photo"] != self.cfg.CURRENT_IMAGE_PATH:
            self._exit_sam_accept("the photo changed")
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self._accept_sam_crack_at(int(real_x), int(real_y)) and state.get("hover"):
            state["done_labels"].add(state["hover"])
        state["hover"] = 0
        self.refresh_zoom_viewport()

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

    def _accept_sam_crack_at(self, x, y):
        """Adds the cracks of the map region under (x, y). True when something was added."""
        cfg, state = self.cfg, self.cfg.sam_accept_state
        prob = state["prob"]
        keep = [f for f in cfg.saved_cracks if f.get('active', True)]
        blocked = self._sam_existing_coverage(keep)
        region, seed = accept_region(prob, x, y, cfg.SAM_ACCEPT_LOW_THRESHOLD, cfg.SAM_ACCEPT_CLICK_RADIUS_PX, blocked)
        if region is None:
            if seed == "taken":
                print(" [SAM CLICK] That crack is already there.")
            else:
                print(f" [SAM CLICK] The model sees no crack within {cfg.SAM_ACCEPT_CLICK_RADIUS_PX} px of the click: "
                      "click closer, or trace it by hand.")
            return False
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
        self._save_annotations_now("SAM CLICK", "the last click")
        return True
