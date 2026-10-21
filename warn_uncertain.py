"""warn_uncertain.py
v1.3.0 [Warn uncertain] (right sidebar, Tools menu, code 12), mixed into CrackSegmentation (smart_segmentation.py).
Press it BEFORE tracing: when a trace is completed, the new crack's scores are checked against
WARN_UNCERTAIN_RELIABILITY_PCT / WARN_UNCERTAIN_CONFIDENCE (config.py). If it is below a limit, a window
proposes to save it as uncertain: Yes = saved as uncertain (red, JSON and masks updated right away, as with
[Uncertain crack]); No = it stays a normal crack. Above the limits no window appears.
Press the button again to switch the check off. Cracks are never marked uncertain automatically any more
(CRACK_UNCERTAIN_AUTO = False): the operator decides.
"""

WARN_UNCERTAIN_CODE = 12
WARN_UNCERTAIN_HUD_BGR = (0, 140, 255)


class WarnUncertainMixin:
    """Proposal to save a weak new crack as uncertain. Relies on CrackSegmentation's cfg, scores and save helpers."""

    # --- Mode ---------------------------------------------------------------------------

    def _toggle_warn_uncertain(self):
        """Code 12 [Warn uncertain]: on = every completed trace is checked; second press = off."""
        cfg = self.cfg
        cfg.warn_uncertain_mode = not cfg.warn_uncertain_mode
        if cfg.warn_uncertain_mode:
            print(" [WARN UNCERTAIN] On: after each trace, a crack below the reliability/confidence limits "
                  "is proposed as uncertain (you confirm or refuse). Press Warn uncertain again to switch it off.")
        else:
            print(" [WARN UNCERTAIN] Off.")
        self.refresh_zoom_viewport()

    def _draw_warn_uncertain_label(self, win_out):
        """HUD line shown while the check is on, below the Crack labels line."""
        if not self.cfg.warn_uncertain_mode:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 4 * (45 if large else 30)
        self._put_hud_text(win_out, "WARN UNCERTAIN: weak new cracks are proposed as uncertain", (15, y),
                           WARN_UNCERTAIN_HUD_BGR)

    # --- Rule ---------------------------------------------------------------------------

    def _warn_uncertain_reasons(self, reliability, confidence):
        """Limits the scores do not reach (empty list = the crack looks fine). A score that cannot be computed is skipped."""
        cfg = self.cfg
        reasons = []
        rel_thr, conf_thr = cfg.WARN_UNCERTAIN_RELIABILITY_PCT, cfg.WARN_UNCERTAIN_CONFIDENCE
        if rel_thr is not None and reliability is not None and reliability < rel_thr:
            reasons.append(f"reliability {reliability:.0f}% < {rel_thr:.0f}%")
        if conf_thr is not None and confidence is not None and confidence < conf_thr:
            reasons.append(f"confidence {confidence:.2f} < {conf_thr:.2f}")
        return reasons

    def _confirm_warn_uncertain(self, number, reasons):
        """Asks whether to save the crack as uncertain. A GUI wrapper overrides this with a dialog."""
        try:
            answer = input(f"Crack {number} is below the limit ({'; '.join(reasons)}). Save it as uncertain? [y/N]: ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes", "s", "si")

    # --- After a trace ------------------------------------------------------------------

    def _warn_uncertain_new_cracks(self, new_cracks):
        """Checks the cracks just traced; marks as uncertain those the operator confirms. Returns whether any was marked."""
        if not self.cfg.warn_uncertain_mode:
            return False
        marked = False
        for crack in new_cracks:
            if self._is_manual_uncertain(crack) or not crack.get('path'):
                continue
            reliability, confidence, _ = self._crack_quality(crack['path'])
            reasons = self._warn_uncertain_reasons(reliability, confidence)
            if reasons and self._ask_warn_uncertain(crack, reasons):
                crack['uncertain_manual'] = True
                marked = True
        return marked

    def _ask_warn_uncertain(self, crack, reasons):
        number = next((i for i, f in enumerate(self.cfg.saved_cracks, start=1) if f is crack), "?")
        accepted = bool(self._confirm_warn_uncertain(number, reasons))
        verdict = "saved as uncertain" if accepted else "kept as a normal crack"
        print(f" [WARN UNCERTAIN] Crack {number} ({'; '.join(reasons)}): {verdict}.")
        return accepted
