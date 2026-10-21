"""crack_labels.py
v1.3.0 [Crack labels] (right sidebar, Inspect cracks section, View menu, code 11), mixed into CrackSegmentation
(smart_segmentation.py). Press it once: while the cursor rests on a crack number or along a crack line, a small
temporary window next to the crack shows its data (length, width, reliability, confidence, views, uncertain /
ignored status ...: the same values the [J] glass panel lists). The window disappears as soon as the cursor moves
away from the crack. Press the button again to switch the labels off.
"""
import cv2
import numpy as np


CRACK_LABELS_CODE = 11
CRACK_LABEL_TITLE_BGR = (0, 230, 255)     # yellow title (same as the Crack report)
CRACK_LABEL_TEXT_BGR = (235, 235, 235)
CRACK_LABEL_WARN_BGR = (90, 90, 255)      # red-ish lines: uncertain / ignored
CRACK_LABEL_BG_BGR = (35, 30, 25)
CRACK_LABEL_BORDER_BGR = (0, 230, 255)
CRACK_LABELS_HUD_BGR = (0, 230, 255)


class CrackLabelsMixin:
    """Hover labels of the cracks. Relies on CrackSegmentation's cfg, quality scores and coordinate helpers."""

    # --- Mode ---------------------------------------------------------------------------

    def _toggle_crack_labels(self):
        """Code 11 [Crack labels]: on = hovering a crack number or line shows its data; second press = off."""
        cfg = self.cfg
        cfg.crack_labels_mode = not cfg.crack_labels_mode
        cfg.crack_label_hover = None
        if cfg.crack_labels_mode:
            print(" [CRACK LABELS] On: rest the cursor on a crack number or along a crack to read its data. "
                  "Press Crack labels again to switch them off.")
        else:
            print(" [CRACK LABELS] Off.")
        self.refresh_zoom_viewport()

    def _reset_crack_labels_for_new_image(self):
        """New photo: no crack under the cursor and no cached values (the toggle stays as it is)."""
        self.cfg.crack_label_hover = None
        self.cfg.crack_label_cache = {}

    def _clear_crack_label_hover(self):
        """The cursor left the photo: the label window disappears."""
        self.cfg.crack_label_hover = None

    # --- Hover --------------------------------------------------------------------------

    def _crack_label_tolerance(self):
        """Hover tolerance converted from window pixels to image pixels at the current zoom."""
        view_w = max(1, self.cfg.zoom_box[2] - self.cfg.zoom_box[0])
        return max(2.0, self.cfg.CRACK_LABEL_HOVER_WIN_PX * view_w / 1200.0)

    def _crack_number_marker_under(self, x, y):
        """Index of the crack whose number marker (start square, window space) holds (x, y), else None."""
        half = self.cfg.CRACK_LABEL_MARKER_WIN_PX
        for idx, f in enumerate(self.cfg.saved_cracks):
            if 'start' not in f or not f.get('active', True) or not f.get('path') or not self._crack_shown(f):
                continue
            wx, wy = self.transform_real_to_window_coords(int(f['start'][0]), int(f['start'][1]))
            if abs(x - wx) <= half and abs(y - wy) <= half:
                return idx
        return None

    def _crack_line_under(self, real_x, real_y):
        """(crack_idx, nearest point) of the active crack line within hover tolerance, else (None, None)."""
        best = (None, None, self._crack_label_tolerance())
        for idx, f in enumerate(self.cfg.saved_cracks):
            if not f.get('active', True) or not f.get('path') or not self._crack_shown(f):
                continue
            pts = np.asarray(f['path'], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            d = np.hypot(pts[:, 0] - real_x, pts[:, 1] - real_y)
            p_idx = int(np.argmin(d))
            if d[p_idx] <= best[2]:
                best = (idx, (int(pts[p_idx, 0]), int(pts[p_idx, 1])), float(d[p_idx]))
        return best[0], best[1]

    def _update_crack_label_hover(self, x, y):
        """Mouse move with the labels on: remembers which crack is under the cursor (None = window hidden).
        The window is anchored to the number marker or to the nearest point of the line."""
        cfg = self.cfg
        if not cfg.crack_labels_mode:
            cfg.crack_label_hover = None
            return
        idx = self._crack_number_marker_under(x, y) if cfg.show_markers else None
        if idx is not None:
            start = cfg.saved_cracks[idx]['start']
            cfg.crack_label_hover = {"idx": idx, "anchor": (int(start[0]), int(start[1]))}
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        idx, point = self._crack_line_under(real_x, real_y)
        cfg.crack_label_hover = None if idx is None else {"idx": idx, "anchor": point}

    def _hovered_crack(self):
        """(number, crack) under the cursor, or (None, None) when there is none (or it no longer exists)."""
        hover = self.cfg.crack_label_hover
        if not self.cfg.crack_labels_mode or not hover:
            return None, None
        idx = hover["idx"]
        if not (0 <= idx < len(self.cfg.saved_cracks)):
            return None, None
        crack = self.cfg.saved_cracks[idx]
        if not crack.get('active', True) or not crack.get('path'):
            return None, None
        return idx + 1, crack

    # --- Values -------------------------------------------------------------------------

    def _crack_width_px(self, crack):
        """(mean, max) width in px of the crack's exported band (same mask as <photo>-crack_mask.png), or (None, None).
        mean = band area / centre-line length; max = twice the largest distance from the band edge."""
        path = crack.get('path') or []
        length = self._path_length(path)
        if len(path) < 2 or length <= 0 or self.cfg.H_img <= 0 or self.cfg.W_img <= 0:
            return None, None
        mask = self._build_variable_width_crack_mask([dict(crack, active=True)], self.cfg.H_img, self.cfg.W_img,
                                                     self.cfg.CRACK_MASK_DILATION_PX)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None, None
        y0, y1, x0, x1 = max(0, ys.min() - 1), ys.max() + 2, max(0, xs.min() - 1), xs.max() + 2
        crop = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=np.uint8)
        crop[1:-1, 1:-1] = mask[y0:y1, x0:x1]
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        return float(len(xs)) / length, max(1.0, 2.0 * float(dist.max()) - 1.0)

    def _crack_ignored_share(self, path):
        """Share (0-1) of the crack's points inside an active ignore region; 0.0 when there is none."""
        regions = [np.array([[float(p[0]), float(p[1])] for p in r['path']], dtype=np.float32).reshape(-1, 1, 2)
                   for r in getattr(self.cfg, 'saved_ignores', [])
                   if r.get('active', True) and len(r.get('path', [])) >= 3]
        if not regions or not path:
            return 0.0
        step = max(1, len(path) // 200)   # at most ~200 checks per crack
        sample = path[::step]
        inside = sum(1 for p in sample
                     if any(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0 for poly in regions))
        return inside / float(len(sample))

    def _crack_label_values(self, crack):
        """Values shown in the label, cached until the crack, its widths or the ignore regions change."""
        ignores = tuple(tuple(map(tuple, r.get('path', []))) for r in getattr(self.cfg, 'saved_ignores', [])
                        if r.get('active', True))
        key = (self._crack_path_key(crack['path']), repr(crack.get('width_segments')), ignores,
               bool(crack.get('uncertain_manual')), self.cfg.PIXEL_TO_CM_SCALE, repr(self._crack_multiview(crack['path'])))
        cache = self.cfg.crack_label_cache
        if key not in cache:
            if len(cache) > 200:
                cache.clear()
            cache[key] = self._compute_crack_label_values(crack)
        return cache[key]

    def _compute_crack_label_values(self, crack):
        reliability, confidence, auto_uncertain = self._crack_quality(crack['path'])
        return {
            "length_px": self._path_length(crack['path']),
            "width_px": self._crack_width_px(crack),
            "reliability": reliability,
            "confidence": confidence,
            "auto_uncertain": auto_uncertain,
            "below_threshold": self._warn_uncertain_reasons(reliability, confidence),
            "multiview": self._crack_multiview(crack['path']),
            "ignored_share": self._crack_ignored_share(crack['path']),
        }

    def _format_width(self, px):
        cm = px * self.cfg.PIXEL_TO_CM_SCALE
        return f"{cm * 10:.1f} mm ({px:.1f} px)"

    def _crack_label_status_lines(self, crack, v):
        """(text, colour) lines about the uncertain / ignored status."""
        lines = []
        if self._is_manual_uncertain(crack):
            lines.append(("Status: UNCERTAIN (saved as uncertain)", CRACK_LABEL_WARN_BGR))
        elif v["auto_uncertain"]:
            lines.append(("Status: UNCERTAIN [?] (automatic rule)", CRACK_LABEL_WARN_BGR))
        else:
            lines.append(("Status: normal crack", CRACK_LABEL_TEXT_BGR))
        if v["below_threshold"] and not self._is_manual_uncertain(crack):
            lines.append(("Below threshold: " + ", ".join(v["below_threshold"]), CRACK_LABEL_WARN_BGR))
        share = v["ignored_share"]
        if share >= 0.999:
            lines.append(("IGNORED: traced inside an ignore region", CRACK_LABEL_WARN_BGR))
        elif share > 0:
            lines.append((f"Partly ignored: {share * 100:.0f}% inside an ignore region", CRACK_LABEL_WARN_BGR))
        return lines

    def _crack_label_measure_lines(self, v):
        """Length, width, scores and multi-view lines."""
        txt = CRACK_LABEL_TEXT_BGR
        lines = [(f"Length: {v['length_px'] * self.cfg.PIXEL_TO_CM_SCALE:.1f} cm ({v['length_px']:.0f} px)", txt)]
        mean_w, max_w = v["width_px"]
        lines.append((f"Width: mean {self._format_width(mean_w)}, max {self._format_width(max_w)}"
                      if mean_w is not None else "Width: not available", txt))
        rel, conf = v["reliability"], v["confidence"]
        lines.append(("Reliability: " + ("n/a" if rel is None else f"{rel:.0f}%")
                      + " | Confidence: " + ("n/a" if conf is None else f"{conf:.2f}"), txt))
        mv = v["multiview"]
        views = (f"confirmed in {mv['views_confirmed']}/{mv['views_checked']} other photo(s)"
                 if mv and mv.get('views_checked') else "not checked yet (computed at save)")
        lines.append(("Multi-view: " + views, txt))
        return lines

    def _crack_label_summary_line(self, crack, v):
        level, _ = self._crack_report_verdict(v["reliability"], v["confidence"], self._crack_entry_uncertain(crack))
        extra = [f"{len(crack['path'])} points"]
        if crack.get('width_segments'):
            extra.append(f"{len(crack['width_segments'])} width-edited stretch(es)")
        if crack.get('links'):
            extra.append("linked")
        return f"Overall reliability: {level} | " + ", ".join(extra), CRACK_LABEL_TEXT_BGR

    def build_crack_label(self, number, crack):
        """(title, [(text, colour)]) of the hover window for one crack."""
        v = self._crack_label_values(crack)
        lines = self._crack_label_measure_lines(v) + self._crack_label_status_lines(crack, v)
        lines.append(self._crack_label_summary_line(crack, v))
        return f"CRACK {number}", lines

    # --- Drawing ------------------------------------------------------------------------

    def _draw_crack_labels_hud(self, win_out):
        """HUD line shown while the labels are on, below the uncertain-mode line."""
        if not self.cfg.crack_labels_mode:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 3 * (45 if large else 30)
        self._put_hud_text(win_out, "CRACK LABELS: hover a crack number or line", (15, y), CRACK_LABELS_HUD_BGR)

    def _crack_label_box(self, win_out, anchor_xy, width, height):
        """Top-left corner of the window: right of / below the anchor, flipped to stay inside the visible frame."""
        ax, ay = anchor_xy
        frame_h, frame_w = win_out.shape[:2]
        top_limit = self._visible_canvas_top()
        x = ax + 18 if ax + 18 + width <= frame_w - 4 else ax - 18 - width
        y = ay + 18 if ay + 18 + height <= frame_h - 4 else ay - 18 - height
        return max(4, min(x, frame_w - 4 - width)), max(top_limit + 4, min(y, frame_h - 4 - height))

    def _draw_crack_label(self, win_out):
        """Draws the hover window of the crack under the cursor (nothing when there is none)."""
        number, crack = self._hovered_crack()
        if crack is None:
            return
        try:
            title, lines = self.build_crack_label(number, crack)
        except Exception as e:
            title, lines = f"CRACK {number}", [(f"(data not available: {e})", CRACK_LABEL_TEXT_BGR)]
        self._paint_crack_label(win_out, title, lines)

    def _paint_crack_label(self, win_out, title, lines):
        """Translucent window with title and lines, next to the hover anchor (small dot on the crack)."""
        scale, thick, line_h = (0.8, 2, 32) if self.cfg.hud_large_size else (0.5, 1, 21)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [title] + [t for t, _ in lines]
        width = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in texts) + 20
        height = line_h * len(texts) + 12
        ax, ay = self.transform_real_to_window_coords(*self.cfg.crack_label_hover["anchor"])
        x, y = self._crack_label_box(win_out, (ax, ay), width, height)
        overlay = win_out.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), CRACK_LABEL_BG_BGR, -1)
        cv2.addWeighted(overlay, 0.82, win_out, 0.18, 0, dst=win_out)
        cv2.rectangle(win_out, (x, y), (x + width, y + height), CRACK_LABEL_BORDER_BGR, 1, cv2.LINE_AA)
        cv2.circle(win_out, (ax, ay), 5, CRACK_LABEL_BORDER_BGR, 2, cv2.LINE_AA)
        cv2.putText(win_out, title, (x + 10, y + line_h), font, scale, CRACK_LABEL_TITLE_BGR, thick + 1, cv2.LINE_AA)
        for i, (text, colour) in enumerate(lines, start=2):
            cv2.putText(win_out, text, (x + 10, y + line_h * i), font, scale, colour, thick, cv2.LINE_AA)
