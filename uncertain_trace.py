"""uncertain_trace.py
v1.1.0 [Uncertain crack] (right sidebar, Tools menu, code 7), mixed into CrackSegmentation (smart_segmentation.py).
Press the button BEFORE tracing: every crack traced while it is on is saved as uncertain by the operator
('uncertain_manual'), whatever its reliability/confidence scores. It is drawn in red, written in the JSON with
label CRACK_UNCERTAIN_LABEL and flags {"uncertain": true, "uncertain_manual": true}, and goes into
<photo>-crack_uncertain_mask.png. JSON and masks are updated as soon as the trace is completed.
Press the button again to go back to normal tracing.
"""
import cv2
import numpy as np


UNCERTAIN_TRACE_CODE = 7
# BGR colour of the cracks marked uncertain by the operator (on screen and in the -seg overlay).
UNCERTAIN_MANUAL_BGR = (0, 0, 255)
UNCERTAIN_TRACE_HUD_BGR = (0, 0, 255)


class UncertainTraceMixin:
    """Manual 'uncertain' marking of new cracks. Relies on CrackSegmentation's cfg and save helpers."""

    # --- Mode ---------------------------------------------------------------------------

    def _toggle_uncertain_trace_mode(self):
        """Code 7 [Uncertain crack]: on = the next cracks are traced as uncertain (red); second press = normal."""
        cfg = self.cfg
        cfg.uncertain_trace_mode = not cfg.uncertain_trace_mode
        if cfg.uncertain_trace_mode:
            hint = "" if cfg.current_tool == 'crack' else " Select the Crack tool [C] to trace."
            print(" [UNCERTAIN] Uncertain mode ON: the cracks you trace now are saved as uncertain (red); "
                  "JSON and masks are updated when each trace is completed. Press Uncertain crack again "
                  "to go back to normal tracing." + hint)
        else:
            print(" [UNCERTAIN] Uncertain mode OFF: new cracks are traced as normal cracks.")
        self.refresh_zoom_viewport()

    def _draw_uncertain_trace_label(self, win_out):
        """HUD line shown while uncertain mode is on, below the crack filter line."""
        if not self.cfg.uncertain_trace_mode:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 2 * (45 if large else 30)
        self._put_hud_text(win_out, "UNCERTAIN MODE: new cracks are uncertain (red)", (15, y), UNCERTAIN_TRACE_HUD_BGR)

    # --- Crack state --------------------------------------------------------------------

    @staticmethod
    def _is_manual_uncertain(crack):
        return bool(crack and crack.get('uncertain_manual'))

    def _crack_entry_uncertain(self, crack):
        """True when the crack is uncertain: marked by the operator, or by the CRACK_UNCERTAIN_* score rule."""
        if self._is_manual_uncertain(crack):
            return True
        return self._crack_quality(crack.get('path'))[2]

    def _mark_new_crack_if_uncertain(self, crack):
        """Called on a crack just traced: tags it when uncertain mode is on. Returns whether it was tagged."""
        if not self.cfg.uncertain_trace_mode:
            return False
        crack['uncertain_manual'] = True
        return True

    # --- Display ------------------------------------------------------------------------

    def _crack_display_bgr(self, crack):
        """Fill colour of a crack on screen: red when marked uncertain by the operator, blue otherwise."""
        return UNCERTAIN_MANUAL_BGR if self._is_manual_uncertain(crack) else (255, 0, 0)

    def _manual_uncertain_mask(self, height, width, region=None, shown_only=True):
        """Mask (cropped to region=(y0, y1, x0, x1), if given) of the centre lines of the active cracks marked
        uncertain by the operator (only those shown by the number filter, if shown_only); None if there are none."""
        cracks = [f for f in self.cfg.saved_cracks if f.get('active', True) and self._is_manual_uncertain(f)
                  and (not shown_only or self._crack_shown(f))]
        if not cracks:
            return None
        y0, y1, x0, x1 = region if region is not None else (0, height, 0, width)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for f in cracks:
            pts = np.asarray(f.get('path') or [], dtype=np.int64)
            if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            keep = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            mask[ys[keep] - y0, xs[keep] - x0] = 255
        return mask

    def _paint_red_uncertain_overlay(self, scene, k, h_actual, w_active, region=None):
        """Paints the operator-uncertain cracks red over the blue fill (same dilation, so red replaces blue)."""
        if not self.cfg.show_crack_overlay:
            return
        mask = self._manual_uncertain_mask(h_actual, w_active, region)
        if mask is None:
            return
        scene[cv2.dilate(mask, k) == 255] = UNCERTAIN_MANUAL_BGR

    def _paint_red_uncertain_on_export(self, segmented_img, k):
        """-seg overlay export: operator-uncertain cracks in red instead of blue."""
        mask = self._manual_uncertain_mask(self.cfg.H_img, self.cfg.W_img, shown_only=False)
        if mask is not None:
            segmented_img[cv2.dilate(mask, k) == 255] = UNCERTAIN_MANUAL_BGR

    def _temp_trace_bgr(self):
        """Colour of the trace in progress: red in uncertain mode, orange otherwise."""
        return UNCERTAIN_MANUAL_BGR if self.cfg.uncertain_trace_mode else (0, 165, 255)

    # --- Saving -------------------------------------------------------------------------

    def _save_after_uncertain_trace(self):
        """JSON and binary masks written right after an uncertain crack is traced."""
        return self._save_annotations_now("UNCERTAIN", "the trace")
