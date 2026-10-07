"""inspection_tools.py
v1.0.6 inspection aids mixed into CrackSegmentation (smart_segmentation.py):
  - HWAV (Highlight Windows Already Viewed): areas already shown in a zoomed view are tinted green.
  - Zoom window: drag a rectangle to zoom straight onto a single crack; pressed again it resets the zoom.
  - Crack report: click a crack to read a written report on its reliability in the [J] glass panel.
All three are sidebar buttons (codes 14, 15, 16 in process_keypress()).
"""
import os

import cv2
import numpy as np


HWAV_GLASS_BGR = (90, 215, 110)     # green glass tint
HWAV_EDGE_BGR = (140, 255, 170)     # brighter border of the viewed area
HWAV_GLASS_ALPHA = 0.30
ZOOM_WINDOW_BGR = (255, 255, 0)     # cyan dashed rectangle
CRACK_REPORT_BGR = (0, 230, 255)    # yellow highlight of the reported crack


class InspectionToolsMixin:
    """HWAV, Zoom window and Crack report. Relies on CrackSegmentation's cfg, zoom and coordinate helpers."""

    # --- shared ---------------------------------------------------------------------

    def _reset_inspection_tools_for_new_image(self):
        """New photo: forgets the viewed areas, any drag in progress and the reported crack (toggles stay as they are)."""
        self.cfg.viewed_mask = None
        self.cfg.zoom_window_state["drag_start"] = None
        self.cfg.zoom_window_state["drag_end"] = None
        self.cfg.zoom_window_state["press_window_xy"] = None
        self.cfg.crack_report_state["idx"] = None

    def _put_hud_text(self, win_out, text, org, color, scale=None, thick=None):
        """HUD text with a black shadow, sized like the other tool indicators."""
        large = self.cfg.hud_large_size
        scale = scale if scale is not None else (1.2 if large else 0.65)
        thick = thick if thick is not None else (3 if large else 2)
        x, y = org
        cv2.putText(win_out, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(win_out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_inspection_tools(self, win_out):
        """Window-space drawing of the three tools, called from the HUD stack."""
        self._draw_viewed_windows_label(win_out)
        self._draw_zoom_window_indicator(win_out)
        self._draw_reported_crack_highlight(win_out)

    # --- HWAV: highlight windows already viewed ------------------------------------------

    def _ensure_viewed_mask(self):
        """The low-resolution viewed-area map of the current photo, created on first use. None without a photo."""
        cfg = self.cfg
        if cfg.W_img <= 0 or cfg.H_img <= 0:
            return None
        scale = min(1.0, cfg.VIEWED_MASK_MAX_DIM / float(max(cfg.W_img, cfg.H_img)))
        shape = (max(1, int(round(cfg.H_img * scale))), max(1, int(round(cfg.W_img * scale))))
        if cfg.viewed_mask is None or cfg.viewed_mask.shape != shape:
            cfg.viewed_mask = np.zeros(shape, dtype=np.uint8)
            cfg.viewed_mask_scale = scale
        return cfg.viewed_mask

    def _box_to_mask_slice(self, box, mask):
        """(y0, y1, x0, x1) of an image-pixel box on the viewed-area map, clamped to it."""
        s = self.cfg.viewed_mask_scale
        h, w = mask.shape[:2]
        x0, y0, x1, y1 = box
        mx0, my0 = max(0, int(x0 * s)), max(0, int(y0 * s))
        mx1, my1 = min(w, int(np.ceil(x1 * s))), min(h, int(np.ceil(y1 * s)))
        return my0, my1, mx0, mx1

    def _is_full_photo_box(self, box):
        x0, y0, x1, y1 = box
        return x0 <= 0 and y0 <= 0 and x1 >= self.cfg.W_img and y1 >= self.cfg.H_img

    def _record_viewed_window(self, box):
        """Marks one zoomed view (image-pixel box) as already seen. The whole-photo view is not a window and is skipped."""
        try:
            box = [int(v) for v in box]
        except (TypeError, ValueError):
            return
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] or self._is_full_photo_box(box):
            return
        mask = self._ensure_viewed_mask()
        if mask is None:
            return
        y0, y1, x0, x1 = self._box_to_mask_slice(box, mask)
        mask[y0:y1, x0:x1] = 255

    def _note_viewport_change(self, old_box):
        """Called by refresh_zoom_viewport(): the view just left is recorded as seen."""
        if list(old_box) != list(self.cfg.zoom_box):
            self._record_viewed_window(old_box)

    def viewed_fraction(self):
        """Share (0-1) of the photo already shown in a zoomed view."""
        mask = self.cfg.viewed_mask
        if mask is None or mask.size == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / mask.size

    def _toggle_viewed_windows(self):
        """Code 14 [HWAV]: first press shows the viewed areas in green glass; second press removes the colors
        and clears the record, so a new inspection pass starts from scratch."""
        cfg = self.cfg
        if cfg.show_viewed_windows:
            cfg.show_viewed_windows = False
            cfg.viewed_mask = None
            print(" [HWAV] Highlight removed: the record of viewed windows is cleared (a new inspection starts now).")
        else:
            cfg.show_viewed_windows = True
            print(f" [HWAV] Windows already viewed while zoomed are now tinted green "
                  f"({100.0 * self.viewed_fraction():.0f}% of the photo so far). Press HWAV again to clear.")
        self.refresh_zoom_viewport()

    def _paint_viewed_windows(self, win_out):
        """Tints the already-viewed part of the current view with green glass and outlines it (window space)."""
        cfg = self.cfg
        mask = cfg.viewed_mask
        if not cfg.show_viewed_windows or mask is None:
            return
        y0, y1, x0, x1 = self._box_to_mask_slice(cfg.zoom_box, mask)
        crop = mask[y0:y1, x0:x1]
        if crop.size == 0 or not crop.any():
            return
        h, w = win_out.shape[:2]
        seen = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        tinted = win_out.copy()
        tinted[seen > 0] = HWAV_GLASS_BGR
        cv2.addWeighted(tinted, HWAV_GLASS_ALPHA, win_out, 1.0 - HWAV_GLASS_ALPHA, 0, dst=win_out)
        contours, _ = cv2.findContours(seen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(win_out, contours, -1, HWAV_EDGE_BGR, 2, cv2.LINE_AA)

    def _draw_viewed_windows_label(self, win_out):
        if not self.cfg.show_viewed_windows:
            return
        y = win_out.shape[0] - (70 if self.cfg.hud_large_size else 50)
        self._put_hud_text(win_out, f"HWAV: {100.0 * self.viewed_fraction():.0f}% of the photo viewed", (15, y), HWAV_EDGE_BGR)

    # --- Zoom window ---------------------------------------------------------------------

    def _toggle_zoom_window(self):
        """Code 15 [Zoom window]: on = drag rectangles to zoom onto them; off (second press) = zoom reset."""
        state = self.cfg.zoom_window_state
        state["drag_start"], state["drag_end"] = None, None
        if state["active"]:
            state["active"] = False
            self._zoom_reset()
            print(" [ZOOM WINDOW] Tool closed: zoom reset to the whole photo.")
            return
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._reset_cut_join_state()
        self._stop_crack_report(hide_panel=True)
        state["active"] = True
        print(" [ZOOM WINDOW] Drag a rectangle around a crack to zoom onto it (repeat to zoom further); "
              "plain clicks still trace with the current tool. "
              "Press Zoom window again to reset the zoom.")
        self.refresh_zoom_viewport()

    def _handle_zoom_window_mouse(self, event, x, y):
        """Press + drag + release = zoom onto the dragged rectangle (corners kept in image pixels).
        A click without dragging, and mouse moves with no button held, are NOT consumed: the caller
        hands them on to the current tool, so cracks can be traced inside the zoomed view.
        Returns the event to hand on, or None when the zoom tool used it."""
        state = self.cfg.zoom_window_state
        if event == cv2.EVENT_LBUTTONDOWN:
            corner = self.transform_window_to_real_coords(x, y)
            state["drag_start"], state["drag_end"] = corner, corner
            state["press_window_xy"] = (x, y)
            return None
        if event == cv2.EVENT_MOUSEMOVE:
            if state["drag_start"] is None:
                return event
            if self._is_zoom_window_drag(state, x, y):
                state["drag_end"] = self.transform_window_to_real_coords(x, y)
            return None
        if event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            start, end = state["drag_start"], self.transform_window_to_real_coords(x, y)
            is_drag = self._is_zoom_window_drag(state, x, y)
            state["drag_start"], state["drag_end"], state["press_window_xy"] = None, None, None
            if not is_drag:
                return cv2.EVENT_LBUTTONDOWN   # a plain click: the current tool gets it
            self._zoom_to_rect(start, end)
            return None
        return event

    def _is_zoom_window_drag(self, state, x, y):
        """True once the pointer has moved ZOOM_WINDOW_MIN_DRAG_PX (window px) from where it was pressed."""
        press = state.get("press_window_xy")
        if press is None:
            return True
        return max(abs(x - press[0]), abs(y - press[1])) >= self.cfg.ZOOM_WINDOW_MIN_DRAG_PX

    def _min_drag_image_px(self):
        """ZOOM_WINDOW_MIN_DRAG_PX (window px) converted to image pixels at the current zoom."""
        view_w = max(1, self.cfg.zoom_box[2] - self.cfg.zoom_box[0])
        return self.cfg.ZOOM_WINDOW_MIN_DRAG_PX * view_w / 1200.0

    def _zoom_to_rect(self, p1, p2):
        """Zooms onto the image rectangle p1-p2, grown to the photo's aspect ratio so nothing is stretched.
        Returns False (zoom unchanged) for a rectangle too small to be a deliberate drag."""
        cfg = self.cfg
        x0, x1 = sorted((p1[0], p2[0]))
        y0, y1 = sorted((p1[1], p2[1]))
        min_side = self._min_drag_image_px()
        if x1 - x0 < min_side or y1 - y0 < min_side:
            print(" [ZOOM WINDOW] Rectangle too small: press and drag to draw it.")
            return False
        w, h = float(x1 - x0), float(y1 - y0)
        aspect = cfg.W_img / float(cfg.H_img)
        if w / h < aspect:
            w = h * aspect
        cfg.zoom_factor = min(cfg.ZOOM_WINDOW_MAX_FACTOR, max(1.0, cfg.W_img / w))
        cfg.zoom_center = [int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))]
        self.refresh_zoom_viewport()
        print(f" [ZOOM WINDOW] Zoom {cfg.zoom_factor:.1f}x on the selected window.")
        return True

    @staticmethod
    def _draw_dashed_line(img, p, q, color, thick=2, dash=10, gap=6):
        length = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if length < 1:
            return
        ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
        t = 0.0
        while t < length:
            t_end = min(length, t + dash)
            a = (int(round(p[0] + ux * t)), int(round(p[1] + uy * t)))
            b = (int(round(p[0] + ux * t_end)), int(round(p[1] + uy * t_end)))
            cv2.line(img, a, b, color, thick, cv2.LINE_AA)
            t += dash + gap

    def _draw_dashed_rect(self, img, p1, p2, color):
        (x0, y0), (x1, y1) = p1, p2
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            self._draw_dashed_line(img, corners[i], corners[(i + 1) % 4], color)

    def _draw_zoom_window_indicator(self, win_out):
        state = self.cfg.zoom_window_state
        if not state["active"]:
            return
        y = 135 if self.cfg.hud_large_size else 120
        self._put_hud_text(win_out, "ZOOM WINDOW: drag = zoom, click = trace", (15, y), ZOOM_WINDOW_BGR)
        if state["drag_start"] is not None and state["drag_end"] is not None:
            p1 = self.transform_real_to_window_coords(*state["drag_start"])
            p2 = self.transform_real_to_window_coords(*state["drag_end"])
            self._draw_dashed_rect(win_out, p1, p2, ZOOM_WINDOW_BGR)

    # --- Crack report ----------------------------------------------------------------------

    def _stop_crack_report(self, hide_panel=True):
        state = self.cfg.crack_report_state
        was_active = state["active"]
        state["active"], state["idx"] = False, None
        if was_active and hide_panel:
            self.cfg.show_info_overlay = False

    def _toggle_crack_report(self):
        """Code 16 [Crack report]: on = click a crack to show its report in the [J] glass (opened automatically);
        second press = report closed and the glass hidden."""
        if self.cfg.crack_report_state["active"]:
            self._stop_crack_report(hide_panel=True)
            print(" [CRACK REPORT] Closed.")
        else:
            self._reset_translate_state()
            self._reset_width_edit_state()
            self._reset_cut_join_state()
            zoom_state = self.cfg.zoom_window_state
            zoom_state["active"], zoom_state["drag_start"], zoom_state["drag_end"] = False, None, None
            self.cfg.crack_report_state["active"] = True
            print(" [CRACK REPORT] Click a crack to read its reliability report. Press Crack report again to close it.")
        self.refresh_zoom_viewport()

    def _handle_crack_report_mouse(self, event, x, y):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        crack_idx, _ = self._pick_active_crack_point(real_x, real_y)
        if crack_idx is None:
            print(" [CRACK REPORT] No crack near the click: click ON a crack line.")
            return
        self.cfg.crack_report_state["idx"] = crack_idx
        self.cfg.show_info_overlay = True
        print(f" [CRACK REPORT] Report of crack {crack_idx + 1}.")
        self.refresh_zoom_viewport()

    def _reported_crack(self):
        """(number, crack) shown in the report, or (None, None) when none is selected or it no longer exists."""
        state = self.cfg.crack_report_state
        idx = state["idx"]
        if not state["active"] or idx is None:
            return None, None
        if not (0 <= idx < len(self.cfg.saved_cracks)):
            return idx + 1, None
        crack = self.cfg.saved_cracks[idx]
        if not crack.get('active', True) or not crack.get('path'):
            return idx + 1, None
        return idx + 1, crack

    def _crack_report_verdict(self, reliability, confidence, uncertain):
        """(level, sentence) summing up the scores; HIGH uses the same thresholds as [Validation]."""
        cfg = self.cfg
        if uncertain:
            return "LOW", ("weak evidence in the photo: check the trace, or keep it only as an ignore region "
                           "(it goes to the uncertain mask).")
        rel_ok = reliability is not None and (cfg.VALIDATION_MIN_MEAN_RELIABILITY_PCT is None
                                              or reliability >= cfg.VALIDATION_MIN_MEAN_RELIABILITY_PCT)
        conf_ok = confidence is not None and (cfg.VALIDATION_MIN_MEAN_CONFIDENCE is None
                                              or confidence >= cfg.VALIDATION_MIN_MEAN_CONFIDENCE)
        if rel_ok and conf_ok:
            return "HIGH", "the trace is well supported by the image: suitable for training."
        return "MEDIUM", "partly supported: review the stretches where the line leaves the dark fracture."

    @staticmethod
    def _describe_reliability(reliability):
        if reliability is None:
            return "Reliability: not available (no texture map for this photo)."
        return (f"Reliability: {reliability:.0f}% of the traced line lies on dark crack-like texture "
                f"found in the photo.")

    @staticmethod
    def _describe_confidence(confidence):
        if confidence is None:
            return "Confidence: not available (line too short or too close to the photo border)."
        if confidence >= 0.85:
            strength = "stands out clearly"
        elif confidence >= 0.65:
            strength = "stands out moderately"
        else:
            strength = "barely stands out"
        return (f"Confidence: {confidence:.2f} -- the line {strength} as a dark ridge against the wall around it "
                f"(0.5 = like the wall, 1.0 = clearly a line).")

    @staticmethod
    def _describe_multiview(multiview):
        if not multiview or not multiview.get('views_checked'):
            return "Multi-view: not checked yet (computed at save, needs other photos of the same building group)."
        return (f"Multi-view: confirmed in {multiview['views_confirmed']} of {multiview['views_checked']} "
                f"other photo(s) of the building group.")

    def _describe_uncertainty(self, reliability, confidence, multiview, uncertain):
        if not uncertain:
            return "Uncertainty: none -- every enabled threshold is met."
        reasons = []
        rel_thr, conf_thr = self.cfg.CRACK_UNCERTAIN_RELIABILITY_PCT, self.cfg.CRACK_UNCERTAIN_CONFIDENCE
        if rel_thr is not None and reliability is not None and reliability < rel_thr:
            reasons.append(f"reliability below {rel_thr:.0f}%")
        if conf_thr is not None and confidence is not None and confidence < conf_thr:
            reasons.append(f"confidence below {conf_thr:.2f}")
        if self._is_multiview_unconfirmed(multiview):
            reasons.append("too few confirming views")
        return "Uncertainty: UNCERTAIN [?] (" + (", ".join(reasons) or "threshold not met") + ")."

    def build_crack_report(self):
        """(title, lines) of the report for the selected crack, or (None, []) when no crack is selected."""
        number, crack = self._reported_crack()
        if number is None:
            return None, []
        title = f"CRACK {number} REPORT"
        if crack is None:
            return title, ["This crack no longer exists (deleted, cut or merged). Click another crack."]
        path = crack['path']
        reliability, confidence, uncertain = self._crack_quality(path)
        multiview = self._crack_multiview(path)
        length_px = self._path_length(path)
        level, sentence = self._crack_report_verdict(reliability, confidence, uncertain)
        lines = [
            f"Length: {length_px * self.cfg.PIXEL_TO_CM_SCALE:.1f} cm ({length_px:.0f} px, {len(path)} points)"
            f" | Photo: {os.path.basename(self.cfg.CURRENT_IMAGE_PATH or '')}",
            self._describe_reliability(reliability),
            self._describe_confidence(confidence),
            self._describe_multiview(multiview),
            self._describe_uncertainty(reliability, confidence, multiview, uncertain),
            f"Overall reliability: {level} -- {sentence}",
        ]
        return title, lines

    def _wrap_hud_text(self, text, max_width, hm):
        """Splits text into lines no wider than max_width pixels (word wrap)."""
        lines, current = [], ""
        for word in text.split(" "):
            candidate = f"{current} {word}" if current else word
            if current and self._hud_text_width(candidate, hm) > max_width:
                lines.append(current)
                candidate = word
            current = candidate
        if current:
            lines.append(current)
        return lines

    def _draw_crack_report_panel(self, win_out, hm):
        """Draws the report in the [J] glass in place of its usual content. Returns False when there is no report to show."""
        title, body = self.build_crack_report()
        if title is None:
            return False
        line_h, y_off = hm['line_h'], hm.get('y_off', 0)
        max_w = win_out.shape[1] - 8 - 30
        wrapped = [w for line in body for w in self._wrap_hud_text(line, max_w, hm)]
        panel_bottom = 6 + y_off + line_h * (len(wrapped) + 2)
        panel_right = self._info_panel_right_edge(win_out, [title] + wrapped, hm)
        glass = win_out.copy()
        cv2.rectangle(glass, (8, 6 + y_off), (panel_right, panel_bottom), (35, 30, 25), -1)
        cv2.addWeighted(glass, 0.55, win_out, 0.45, 0, win_out)
        cv2.putText(win_out, title, (15, hm['y_pos1']), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], CRACK_REPORT_BGR, hm['f_thick'], cv2.LINE_AA)
        for i, line in enumerate(wrapped):
            y = hm['y_pos1'] + line_h * (i + 1)
            cv2.putText(win_out, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (235, 235, 235), hm['f_thick'], cv2.LINE_AA)
        return True

    def _is_crack_report_shown(self):
        return self.cfg.show_info_overlay and self.cfg.crack_report_state["active"] \
            and self.cfg.crack_report_state["idx"] is not None

    def _draw_reported_crack_highlight(self, win_out):
        """Prompt while waiting for a click; afterwards outlines the reported crack so it is clear which one the report is about."""
        state = self.cfg.crack_report_state
        if not state["active"]:
            return
        if state["idx"] is None:
            y = 135 if self.cfg.hud_large_size else 120
            self._put_hud_text(win_out, "CRACK REPORT: click a crack", (15, y), CRACK_REPORT_BGR)
            return
        if not self._is_crack_report_shown():
            return
        _, crack = self._reported_crack()
        if crack is None:
            return
        pts = np.array([self.transform_real_to_window_coords(int(p[0]), int(p[1])) for p in crack['path']], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(win_out, [pts], False, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(win_out, [pts], False, CRACK_REPORT_BGR, 3, cv2.LINE_AA)
