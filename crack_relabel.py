"""crack_relabel.py
v1.3.0 Make uncertain / Make normal, mixed into CrackSegmentation (smart_segmentation.py).
Changes the class of cracks ALREADY traced:
  - Ctrl+C (Cmd+C on macOS), or Ctrl/Cmd+click on the sidebar [Crack tool] button, or the Tools menu (code 128):
    the next clicks on a normal (blue) crack -- on its line or on its number -- make it uncertain (red);
  - Ctrl+U (Cmd+U), or Ctrl/Cmd+click on the sidebar [Uncertain crack] button, or the Tools menu (code 129):
    the next clicks on an uncertain (red) crack make it normal (blue).
One crack per press: the click on a crack ends the mode, so the next clicks work as usual again (delete by marker,
trace ...). A click away from any crack keeps it waiting; the same combination again cancels it (the other one
switches to the opposite change), and so does choosing any other command (Cut, Join, a tool, Save ...). JSON and binary masks, -crack_uncertain_mask.png included, are updated after the
change; [U] undoes it.
"""
import cv2


MARK_UNCERTAIN_CODE = 128   # C1 control range: never produced by a typed character (Italian accents are >= 160)
MARK_NORMAL_CODE = 129
RELABEL_TARGET_BY_CODE = {MARK_UNCERTAIN_CODE: 'uncertain', MARK_NORMAL_CODE: 'normal'}
# Keys that only move or change the view: they leave a pending relabel waiting. Any other key or command
# (Cut, Join, a tool, Save ...) cancels it.
RELABEL_KEEP_KEYS = frozenset([0, 1, 2, 3, 81, 82, 83, 84, ord('+'), ord('='), ord('-'), ord('o'), ord('O'),
                               ord('0'), ord('z'), ord('Z'), ord('h'), ord('H'), ord('n'), ord('N'), ord('j'),
                               ord('J'), ord(' '), ord('.'), ord('f'), ord('F'), 14, 15, 11, 133])


class CrackRelabelMixin:
    """Click-to-relabel of existing cracks. Relies on CrackSegmentation's cfg, crack pickers and save helpers."""

    # --- Mode ---------------------------------------------------------------------------

    def _toggle_crack_relabel(self, target):
        """Codes 128 / 129: on = the next clicked crack becomes 'uncertain' / 'normal'; same code again = cancel."""
        cfg = self.cfg
        cfg.crack_relabel_mode = None if cfg.crack_relabel_mode == target else target
        if cfg.crack_relabel_mode == 'uncertain':
            print(" [MAKE UNCERTAIN] Click a blue crack (its line or its number) to make it uncertain (red). "
                  "Ctrl/Cmd+C again to cancel.")
        elif cfg.crack_relabel_mode == 'normal':
            print(" [MAKE NORMAL] Click a red crack (its line or its number) to make it a normal crack (blue). "
                  "Ctrl/Cmd+U again to cancel.")
        else:
            print(" [RELABEL] Off: clicks trace again.")
        self.refresh_zoom_viewport()

    def _cancel_crack_relabel_for_key(self, key_clean):
        """Another command was chosen (Cut, Join, a tool ...) while a relabel was pending: it is cancelled, so
        the next click goes to that command. View keys (zoom, pan, markers ...) leave it waiting."""
        if self.cfg.crack_relabel_mode is None or key_clean in RELABEL_TARGET_BY_CODE or key_clean in RELABEL_KEEP_KEYS:
            return
        self.cfg.crack_relabel_mode = None
        print(" [RELABEL] Cancelled: another command was chosen.")

    def _draw_crack_relabel_label(self, win_out):
        """Mode line (status bar in the GUI) while a relabel mode is on."""
        mode = self.cfg.crack_relabel_mode
        if mode is None:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 5 * (45 if large else 30)
        if mode == 'uncertain':
            text, color = "MAKE UNCERTAIN: click a blue crack or its number (Ctrl/Cmd+C = cancel)", (0, 0, 255)
        else:
            text, color = "MAKE NORMAL: click a red crack or its number (Ctrl/Cmd+U = cancel)", (255, 0, 0)
        self._put_hud_text(win_out, text, (15, y), color)

    # --- Click --------------------------------------------------------------------------

    def _crack_under_click(self, x, y):
        """Index of the crack clicked at window (x, y): its number marker (when markers are shown) first, then the
        nearest point of an active line within the pick tolerance. None if no crack is there."""
        if self.cfg.show_markers:
            idx = self._crack_number_marker_under(x, y)
            if idx is not None:
                return idx
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        idx, _ = self._pick_active_crack_point(real_x, real_y)
        return idx

    def _handle_crack_relabel_mouse(self, event, x, y):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = self._crack_under_click(x, y)
        if idx is None:
            key = "C" if self.cfg.crack_relabel_mode == 'uncertain' else "U"
            print(f" [RELABEL] No crack there: click ON a crack line or on its number (Ctrl/Cmd+{key} cancels).")
            return
        # One crack per press: from the next click the tools work as usual again (e.g. deleting by its marker).
        target, self.cfg.crack_relabel_mode = self.cfg.crack_relabel_mode, None
        if self._relabel_crack(idx, target):
            self.recalculate_masks()
            self._save_annotations_now("RELABEL", "the change")
        self.refresh_zoom_viewport()

    def _relabel_crack(self, idx, target):
        """Makes crack idx uncertain (operator mark, red) or normal (blue). False when it already is."""
        crack = self.cfg.saved_cracks[idx]
        is_uncertain = self._is_manual_uncertain(crack)
        if (target == 'uncertain') == is_uncertain:
            state = "already uncertain (red)" if is_uncertain else "already a normal crack (blue)"
            print(f" [RELABEL] Crack {idx + 1} is {state}: nothing changed.")
            return False
        before = self._snapshot_crack_edit_state()
        if target == 'uncertain':
            crack['uncertain_manual'] = True
        else:
            crack.pop('uncertain_manual', None)
            crack.pop('uncertain', None)
        self._commit_crack_edit(before)
        verdict = "uncertain (red)" if target == 'uncertain' else "a normal crack (blue)"
        print(f" [RELABEL] Crack {idx + 1} is now {verdict}.")
        return True
