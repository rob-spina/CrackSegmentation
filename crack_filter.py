"""crack_filter.py
v1.0.7 Show cracks by number, mixed into CrackSegmentation (smart_segmentation.py).
The sidebar button (code 19 in process_keypress()) asks for one or more crack numbers separated by ';'
(e.g. 3;7;12) and then shows only those cracks: blue fill, markers and the [J] list. Hidden cracks cannot
be clicked by the tools either. An empty answer shows every crack again.
Display only: the JSON, the binary masks and the -seg preview always contain every crack.
"""
import numpy as np


CRACK_FILTER_CODE = 19
CRACK_FILTER_HUD_BGR = (0, 255, 255)


def parse_crack_numbers(text):
    """Crack numbers typed as '3; 7;12' (';' or ',' between them) -> [3, 7, 12], sorted and unique.
    Raises ValueError on anything that is not a whole number from 1 up."""
    numbers = set()
    for part in str(text).replace(',', ';').split(';'):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(part)
        numbers.add(n)
    return sorted(numbers)


class CrackFilterMixin:
    """Show cracks by number. Relies on CrackSegmentation's cfg, _crack_path_key and mask helpers."""

    def _prompt_crack_numbers(self, current_text):
        """Asks for the crack numbers to show. Returns the typed text, or None if cancelled.
        A GUI wrapper overrides this with a graphical dialog."""
        print("\n[PROMPT] Switch to the terminal to enter the crack numbers...")
        try:
            return input(f"Crack numbers separated by ';' (empty = all) [{current_text}]: ")
        except EOFError:
            return None

    def _reset_crack_filter(self):
        self.cfg.crack_filter = None
        self.cfg.filtered_blue_mask = None

    def _crack_shown(self, crack):
        """False only for a crack hidden by the number filter. A crack edited after filtering
        (new path) counts as a new crack and is shown."""
        flt = self.cfg.crack_filter
        if flt is None or not crack.get('path'):
            return True
        return self._crack_path_key(crack['path']) not in flt["hidden_keys"]

    def _crack_filter_button(self):
        """Code 19 [Show cracks by number]: asks for the numbers, then shows only those cracks."""
        flt = self.cfg.crack_filter
        current = "; ".join(str(n) for n in flt["numbers"]) if flt else ""
        text = self._prompt_crack_numbers(current)
        if text is None:
            return
        try:
            numbers = parse_crack_numbers(text)
        except ValueError:
            print(f" [CRACK FILTER] '{text}' is not a list of crack numbers: type e.g. 3;7;12.")
            return
        if not numbers:
            self._reset_crack_filter()
            print(" [CRACK FILTER] Off: every crack is shown.")
        else:
            self._apply_crack_filter(numbers)
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _apply_crack_filter(self, numbers):
        cracks = self.cfg.saved_cracks
        valid = [n for n in numbers if n <= len(cracks)]
        missing = [n for n in numbers if n > len(cracks)]
        if not valid:
            print(f" [CRACK FILTER] No crack with those numbers: this photo has {len(cracks)} crack(s). Filter unchanged.")
            return
        hidden = {self._crack_path_key(c['path']) for i, c in enumerate(cracks, start=1)
                  if i not in valid and c.get('path')}
        self.cfg.crack_filter = {"numbers": valid, "hidden_keys": hidden}
        print(f" [CRACK FILTER] Showing only crack(s) {', '.join(map(str, valid))}"
              + (f" (no crack {', '.join(map(str, missing))} in this photo)" if missing else "")
              + ". Press the button and leave the field empty to show every crack.")

    def _refresh_filtered_crack_mask(self):
        """Display-only blue mask of the cracks the filter shows (None with no filter); rebuilt with the masks."""
        if self.cfg.crack_filter is None or self.cfg.blue_visual_mask is None:
            self.cfg.filtered_blue_mask = None
            return
        mask = np.zeros_like(self.cfg.blue_visual_mask)
        self._stamp_points_on_mask(mask, [c for c in self.cfg.saved_cracks if self._crack_shown(c)])
        self.cfg.filtered_blue_mask = mask

    def _display_blue_mask(self):
        """The blue mask drawn on screen: filtered when the number filter is on."""
        mask = self.cfg.filtered_blue_mask
        if self.cfg.crack_filter is not None and mask is not None and mask.shape == self.cfg.blue_visual_mask.shape:
            return mask
        return self.cfg.blue_visual_mask

    def _draw_crack_filter_label(self, win_out):
        flt = self.cfg.crack_filter
        if flt is None:
            return
        y = (135 if self.cfg.hud_large_size else 120) + (45 if self.cfg.hud_large_size else 30)
        shown = ", ".join(map(str, flt["numbers"]))
        self._put_hud_text(win_out, f"FILTER: cracks {shown}", (15, y), CRACK_FILTER_HUD_BGR)
