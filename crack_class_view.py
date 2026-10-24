"""crack_class_view.py
v1.4.0 [Hide uncertain overlay] (code 130) and [Hide normal overlay] (code 131), left sidebar View section and View
menu, mixed into CrackSegmentation (smart_segmentation.py). Each hides the on-screen fill of one class only:
uncertain (red) or normal (blue) cracks. [N] still hides both. Markers, numbers and exported files are unchanged.
Also v1.4.0: with the Crack tool and no trace in progress, a click ON a crack line hides / shows that crack's markers
(numbered start square and end dot); hidden markers cannot delete the crack. Shift+click or Ctrl/Cmd+click starts a
trace there instead. On screen only, never exported.
"""

HIDE_UNCERTAIN_OVERLAY_CODE = 130
HIDE_NORMAL_OVERLAY_CODE = 131


class CrackClassViewMixin:
    """Per-class visibility of the crack fill. Relies on CrackSegmentation's cfg and mask helpers."""

    def _crack_class_shown(self, crack):
        """True when the fill of this crack's class (uncertain = red, normal = blue) is visible."""
        if self._is_manual_uncertain(crack):
            return self.cfg.show_uncertain_overlay
        return self.cfg.show_normal_overlay

    def _both_crack_classes_shown(self):
        return self.cfg.show_uncertain_overlay and self.cfg.show_normal_overlay

    @staticmethod
    def _crack_number_hidden(crack):
        return bool(crack.get('number_hidden'))

    def _toggle_crack_number_at(self, real_x, real_y):
        """Crack tool click ON a crack line: hides / shows its markers and number. True when a crack was there."""
        idx, _ = self._pick_active_crack_point(real_x, real_y)
        if idx is None:
            return False
        crack = self.cfg.saved_cracks[idx]
        if crack.get('number_hidden'):
            crack.pop('number_hidden')
            print(f" [MARKERS] Crack {idx + 1}: markers shown again.")
        else:
            crack['number_hidden'] = True
            print(f" [MARKERS] Crack {idx + 1}: markers hidden (click the crack again to show them; "
                  "Shift+click or Cmd/Ctrl+click starts a trace on it).")
        self.refresh_zoom_viewport()
        return True

    def _toggle_uncertain_overlay(self):
        """Code 130: hides / shows only the red fill of the uncertain cracks."""
        self.cfg.show_uncertain_overlay = not self.cfg.show_uncertain_overlay
        state = "visible" if self.cfg.show_uncertain_overlay else "hidden"
        print(f" [UNCERTAIN OVERLAY] Red fill of the uncertain cracks is now {state}.")
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _toggle_normal_overlay(self):
        """Code 131: hides / shows only the blue fill of the normal cracks."""
        self.cfg.show_normal_overlay = not self.cfg.show_normal_overlay
        state = "visible" if self.cfg.show_normal_overlay else "hidden"
        print(f" [NORMAL OVERLAY] Blue fill of the normal cracks is now {state}.")
        self.recalculate_masks()
        self.refresh_zoom_viewport()
