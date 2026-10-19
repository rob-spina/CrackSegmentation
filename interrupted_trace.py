"""interrupted_trace.py
v1.2.0 Interrupted crack, mixed into CrackSegmentation (smart_segmentation.py).
When the last point of a Crack-tool trace is clicked, the route is checked against the photo (crack mask
cfg.binary_mask, and darkness against the wall around): if it crosses a stretch with no crack (a gap of at least
CRACK_GAP_MIN_PX), the crack is not continuous. An on-screen message says so and the trace is stored as SEPARATE
cracks, one per continuous piece, without the gap: each is its own entry in the JSON and the gap stays black in
the binary PNG masks.
They are written by [S] Save / Save+next (in Uncertain mode right away, as every uncertain trace).
"""
import time

import cv2
import numpy as np


INTERRUPTED_CRACK_BANNER_BGR = (0, 140, 255)


class InterruptedTraceMixin:
    """Splits a just-traced crack at its gaps. Relies on CrackSegmentation's cfg and crack helpers."""

    # --- Gap detection ------------------------------------------------------------------

    def _trace_points_on_crack(self, path):
        """Bool per path point: True where a crack lies under it -- a crack-mask pixel within CRACK_GAP_TOLERANCE_PX,
        or a pixel clearly darker than the wall around it (the mask is hollow in the middle of wide cracks)."""
        pts = np.asarray(path, dtype=np.int64).reshape(-1, 2)
        h, w = self.cfg.binary_mask.shape[:2]
        xs, ys = np.clip(pts[:, 0], 0, w - 1), np.clip(pts[:, 1], 0, h - 1)
        pad = max(0, int(self.cfg.CRACK_GAP_TOLERANCE_PX)) + int(self.cfg.CRACK_GAP_BG_BLOCK_PX)
        x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        x1, y1 = min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)
        on = self._mask_hits_near(self.cfg.binary_mask[y0:y1, x0:x1])
        dark = self._darker_than_wall((x0, y0, x1, y1))
        if dark is not None:
            on = on | dark
        return on[ys - y0, xs - x0]

    def _mask_hits_near(self, crop):
        """Crack-mask crop widened by CRACK_GAP_TOLERANCE_PX (a route a pixel off the mask still counts)."""
        hits = (crop > 0).astype(np.uint8)
        r = max(0, int(self.cfg.CRACK_GAP_TOLERANCE_PX))
        if r:
            hits = cv2.dilate(hits, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)))
        return hits > 0

    def _darker_than_wall(self, roi):
        """Pixels of the roi at least CRACK_GAP_DARKNESS grey levels below the mean of a CRACK_GAP_BG_BLOCK_PX
        window around them; None without the photo (or if its size differs from the masks)."""
        img = self.cfg.img_original
        if img is None or img.shape[:2] != self.cfg.binary_mask.shape[:2]:
            return None
        x0, y0, x1, y1 = roi
        crop = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = gray.astype(np.float32)
        block = max(3, int(self.cfg.CRACK_GAP_BG_BLOCK_PX) | 1)
        wall = cv2.blur(gray, (block, block))
        return cv2.GaussianBlur(gray, (5, 5), 0) < wall - float(self.cfg.CRACK_GAP_DARKNESS)

    @staticmethod
    def _off_runs(on):
        """(first, last) index pairs of the runs of False in on."""
        runs, start = [], None
        for i, flag in enumerate(on):
            if not flag and start is None:
                start = i
            elif flag and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(on) - 1))
        return runs

    def _merge_close_runs(self, runs):
        """Joins gap runs split only by a few stray crack pixels (plaster texture inside a real gap)."""
        # The tolerance widens a stray speck by CRACK_GAP_TOLERANCE_PX on each side.
        limit = self.cfg.CRACK_GAP_MERGE_POINTS + 2 * max(0, int(self.cfg.CRACK_GAP_TOLERANCE_PX))
        merged = []
        for run in runs:
            if merged and run[0] - merged[-1][1] - 1 <= limit:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        return merged

    def _find_trace_gaps(self, path):
        """(first, last) indices of the path stretches with no crack under them, at least CRACK_GAP_MIN_PX long
        (measured from the last crack point before to the first one after). Stretches at the two ends are not
        gaps: the crack simply starts or ends there."""
        if self.cfg.binary_mask is None or not path or len(path) < 3:
            return []
        runs = self._merge_close_runs(self._off_runs(self._trace_points_on_crack(path)))
        gaps = []
        for first, last in runs:
            if first == 0 or last == len(path) - 1:
                continue
            if self._path_length(path[first - 1:last + 2]) >= self.cfg.CRACK_GAP_MIN_PX:
                gaps.append((first, last))
        return gaps

    def _split_path_at_gaps(self, path, gaps):
        """The continuous pieces of path between the gaps (pieces shorter than CUT_JOIN_MIN_PIECE_POINTS dropped)."""
        pieces, start = [], 0
        for first, last in gaps:
            pieces.append(path[start:first])
            start = last + 1
        pieces.append(path[start:])
        return [list(p) for p in pieces if len(p) >= self.CUT_JOIN_MIN_PIECE_POINTS]

    # --- Splitting ----------------------------------------------------------------------

    def _interrupted_trace_pieces(self, computed_path):
        """The pieces a new trace must be stored as: [path] when it is continuous (or splitting is off)."""
        if not self.cfg.SPLIT_INTERRUPTED_CRACKS:
            return [computed_path]
        gaps = self._find_trace_gaps(computed_path)
        pieces = self._split_path_at_gaps(computed_path, gaps) if gaps else []
        return pieces if len(pieces) >= 2 else [computed_path]

    def _new_crack_from_path(self, path, suffix=""):
        crack = {
            'start': (int(path[0][0]), int(path[0][1])),
            'end': (int(path[-1][0]), int(path[-1][1])),
            'path': path,
            'active': True,
            'session_id': str(time.time()) + suffix,
        }
        self._mark_new_crack_if_uncertain(crack)   # v1.1.0 [Uncertain crack]
        return crack

    def _store_interrupted_trace(self, pieces):
        """Appends one crack per piece; a single [U] removes them all (crack snapshot, as CUT/JOIN)."""
        before = self._snapshot_crack_edit_state()
        cracks = [self._new_crack_from_path(p, suffix=f"-{i + 1}") for i, p in enumerate(pieces)]
        self.cfg.saved_cracks.extend(cracks)
        self._commit_crack_edit(before)
        self._announce_interrupted_crack(len(pieces))
        return cracks

    # --- Message ------------------------------------------------------------------------

    def _announce_interrupted_crack(self, n_pieces):
        save_hint = "saved now (Uncertain mode)" if self.cfg.uncertain_trace_mode else "[S] saves them"
        self.cfg.interrupted_crack_message = (f"INTERRUPTED CRACK: saved as {n_pieces} separate cracks "
                                              f"({save_hint}, [U] undoes)")
        self.cfg.interrupted_crack_triggered = True
        self.cfg.banner_first_seen.pop('interrupted_crack_triggered', None)
        print(f" [CRACK] The traced crack is not continuous: it has been split into {n_pieces} separate cracks "
              f"without the gap{'s' if n_pieces > 2 else ''}. Each one gets its own entry in the JSON and the gap "
              f"stays out of the PNG masks; {save_hint}, [U] undoes.")

    def _draw_interrupted_crack_banner(self, win_out, hm):
        """Orange banner for INTERRUPTED_CRACK_BANNER_SECONDS after a trace was split ([.] hides it)."""
        if not self.cfg.interrupted_crack_triggered:
            return
        first_seen = self.cfg.banner_first_seen.setdefault('interrupted_crack_triggered', time.time())
        if time.time() - first_seen > self.cfg.INTERRUPTED_CRACK_BANNER_SECONDS:
            self.cfg.interrupted_crack_triggered = False
            self.cfg.banner_first_seen.pop('interrupted_crack_triggered', None)
            return
        cv2.rectangle(win_out, (15, 250), (1185, 310), INTERRUPTED_CRACK_BANNER_BGR, -1)
        cv2.putText(win_out, self.cfg.interrupted_crack_message, (35, 288), cv2.FONT_HERSHEY_SIMPLEX,
                    hm['f_scale_btm'], (255, 255, 255), hm['f_thick_btm'], cv2.LINE_AA)
