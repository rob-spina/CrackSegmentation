"""crack_cleanup.py
v1.0.8 crack clean-up tools, mixed into CrackSegmentation (smart_segmentation.py):
  - Eraser (sidebar, code 21): a square window follows the mouse; press and drag it over a crack to erase
    the part it touches. A crack crossed in the middle is split in two. [ / ] shrink / enlarge the window.
  - Remove uncertain (sidebar, code 22): deletes every crack on screen marked [?] (uncertain);
    cracks hidden by Show cracks by number are kept.
Both write the JSON and the binary masks right away when the change is done; [U] undoes it on screen
and [S] saves again.
"""
import cv2
import numpy as np


ERASER_CODE = 21
REMOVE_UNCERTAIN_CODE = 22
ERASER_BGR = (60, 60, 255)            # red window
ERASER_ACTIVE_BGR = (0, 0, 255)       # brighter while erasing


def kept_runs(inside):
    """(first, last) index ranges of the consecutive False entries of a boolean array (the points kept)."""
    runs, start = [], None
    for i, gone in enumerate(inside):
        if not gone and start is None:
            start = i
        elif gone and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(inside) - 1))
    return runs


def erased_runs(path, inside):
    """The erased points of path, one list per consecutive run (used to keep JOIN off the erased stretch)."""
    out, current = [], []
    for pt, gone in zip(path, inside):
        if gone:
            current.append((int(pt[0]), int(pt[1])))
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def sweep_centres(p, q, step):
    """Window centres from p to q (p excluded unless p == q) no more than step apart, so a fast drag
    leaves no unerased gap between two mouse events."""
    if p is None or tuple(p) == tuple(q):
        return [tuple(q)]
    dist = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    n = max(1, int(np.ceil(dist / max(1.0, step))))
    return [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(1, n + 1)]


class CrackCleanupMixin:
    """Eraser and Remove uncertain. Relies on CrackSegmentation's cfg, CUT helpers and save helpers."""

    # --- Eraser ---------------------------------------------------------------------------

    def _reset_eraser_state(self):
        st = self.cfg.eraser_state
        st["active"], st["pos"], st["stroke"], st["erased"], st["changed"] = False, None, None, [], False

    def _toggle_eraser(self):
        """Code 21 [Eraser]: on = drag the window over a crack to erase that part; second press = off."""
        if self.cfg.eraser_state["active"]:
            self._reset_eraser_state()
            print(" [ERASER] Tool closed.")
        else:
            self._close_other_tools_for_eraser()
            self.cfg.eraser_state["active"] = True
            print(" [ERASER] Press and drag the red window over a crack to erase that part "
                  "([ / ] smaller / larger window). JSON and masks are updated on release; "
                  "[U] undoes. Press Eraser again to close it.")
        self.refresh_zoom_viewport()

    def _close_other_tools_for_eraser(self):
        """Only one click-priority tool at a time; a crack trace in progress is dropped."""
        self._reset_cut_join_state()   # also closes the eraser itself, so it runs before activating it
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._stop_crack_report(hide_panel=True)
        if self.cfg.alt_route_state["active"]:
            self._reset_alt_route_state()
        self.cfg.portion_state["active"], self.cfg.portion_state["drag"] = False, None
        self.cfg.edit_mode = False
        self.cfg.temp_start = None
        self.cfg.temp_path.clear()
        self.cfg.trace_waypoints, self.cfg.trace_partial_path = [], []

    def _eraser_half_side(self):
        """Half the window side, converted from window pixels to image pixels at the current zoom."""
        view_w = max(1, self.cfg.zoom_box[2] - self.cfg.zoom_box[0])
        return max(1.0, self.cfg.ERASER_SIZE_WIN_PX / 2.0 * view_w / 1200.0)

    def _resize_eraser(self, step_sign):
        cfg = self.cfg
        size = cfg.ERASER_SIZE_WIN_PX + step_sign * cfg.ERASER_SIZE_STEP_WIN_PX
        cfg.ERASER_SIZE_WIN_PX = int(min(cfg.ERASER_MAX_SIZE_WIN_PX, max(cfg.ERASER_MIN_SIZE_WIN_PX, size)))
        print(f" [ERASER] Window size: {cfg.ERASER_SIZE_WIN_PX} px.")

    def _dispatch_eraser_keys(self, key_clean):
        """[ / ] resize the eraser window while the tool is on. True if the key was used."""
        if not self.cfg.eraser_state["active"] or key_clean not in (ord('['), ord(']')):
            return False
        self._resize_eraser(-1 if key_clean == ord('[') else 1)
        return True

    def _handle_eraser_mouse(self, event, x, y):
        """Press = start erasing, drag = erase along the way, release = commit (Undo + files)."""
        st = self.cfg.eraser_state
        pos = self.transform_window_to_real_coords(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            st["stroke"], st["erased"], st["changed"] = self._snapshot_crack_edit_state(), [], False
            st["pos"] = pos
            self._erase_along(None, pos)
        elif event == cv2.EVENT_MOUSEMOVE:
            previous, st["pos"] = st["pos"], pos
            if st["stroke"] is not None:
                self._erase_along(previous, pos)
        elif event == cv2.EVENT_LBUTTONUP and st["stroke"] is not None:
            st["pos"] = pos
            self._erase_along(None, pos)
            self._finish_eraser_stroke()

    def _erase_along(self, previous, pos):
        half = self._eraser_half_side()
        changed = False
        for cx, cy in sweep_centres(previous, pos, half):
            changed = self._erase_in_box(cx, cy, half) or changed
        if changed:
            self.cfg.eraser_state["changed"] = True
            self.recalculate_masks()
            self.refresh_zoom_viewport()

    def _erase_in_box(self, cx, cy, half):
        """Removes the points of every shown, active crack inside the square (cx +- half, cy +- half).
        A crack keeps its untouched pieces (shorter ones than CUT_JOIN_MIN_PIECE_POINTS are dropped)."""
        cracks = self.cfg.saved_cracks
        changed = False
        for idx in range(len(cracks) - 1, -1, -1):
            crack = cracks[idx]
            if not crack.get('active', True) or not crack.get('path') or not self._crack_shown(crack):
                continue
            pts = np.asarray(crack['path'], dtype=np.float64).reshape(-1, 2)
            inside = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
            if not inside.any():
                continue
            cracks[idx:idx + 1] = self._pieces_left_by_eraser(crack, inside)
            self.cfg.eraser_state["erased"].extend(erased_runs(crack['path'], inside))
            changed = True
        return changed

    def _pieces_left_by_eraser(self, crack, inside):
        path = list(crack['path'])
        pieces = []
        for first_i, last_i in kept_runs(inside):
            if last_i - first_i + 1 < self.CUT_JOIN_MIN_PIECE_POINTS:
                continue
            segs, _ = self._remap_width_segments(crack.get('width_segments'), first_i, last_i)
            pieces.append(self._crack_with_path(crack, path[first_i:last_i + 1], segs))
        return pieces

    def _finish_eraser_stroke(self):
        st = self.cfg.eraser_state
        before, changed, erased = st["stroke"], st["changed"], st["erased"]
        st["stroke"], st["erased"], st["changed"] = None, [], False
        if not changed:
            print(" [ERASER] No crack under the window: nothing erased.")
            return
        self.cfg.cut_exclusions.extend(erased)
        self._commit_crack_edit(before)
        n_points = sum(len(run) for run in erased)
        print(f" [ERASER] {n_points} crack point(s) erased; {len(self.cfg.saved_cracks)} crack(s) now in the photo.")
        self.recalculate_masks()
        self._save_annotations_now("ERASER", "the erasing")
        self.refresh_zoom_viewport()

    def _draw_eraser(self, win_out):
        st = self.cfg.eraser_state
        if not st["active"]:
            return
        y = 135 if self.cfg.hud_large_size else 120
        self._put_hud_text(win_out, f"ERASER: drag over a crack to erase it  [ / ] size ({self.cfg.ERASER_SIZE_WIN_PX} px)",
                           (15, y), ERASER_BGR)
        if st["pos"] is None:
            return
        half = self._eraser_half_side()
        cx, cy = st["pos"]
        p1 = self.transform_real_to_window_coords(cx - half, cy - half)
        p2 = self.transform_real_to_window_coords(cx + half, cy + half)
        erasing = st["stroke"] is not None
        color = ERASER_ACTIVE_BGR if erasing else ERASER_BGR
        cv2.rectangle(win_out, p1, p2, (0, 0, 0), 5 if erasing else 4, cv2.LINE_AA)
        cv2.rectangle(win_out, p1, p2, color, 3 if erasing else 2, cv2.LINE_AA)

    # --- Remove uncertain -----------------------------------------------------------------

    def _uncertain_crack_indices(self):
        """Indices of the active cracks on screen marked [?] (same rule as the [J] list and the JSON).
        Cracks hidden by Show cracks by number are left alone."""
        return [i for i, f in enumerate(self.cfg.saved_cracks)
                if f.get('active', True) and f.get('path') and self._crack_shown(f)
                and self._crack_quality(f.get('path'))[2]]

    def _confirm_remove_uncertain(self, count):
        """Asks before deleting. A GUI wrapper overrides this with a dialog."""
        try:
            answer = input(f"Remove the {count} uncertain crack(s) [?] shown on screen? [y/N]: ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes", "s", "si")

    def _remove_uncertain_cracks(self):
        """Code 22 [Remove uncertain]: deletes every [?] crack on screen, then updates JSON and masks."""
        indices = self._uncertain_crack_indices()
        if not indices:
            print(" [REMOVE UNCERTAIN] No uncertain crack [?] on screen.")
            return False
        numbers = ", ".join(str(i + 1) for i in indices)
        if not self._confirm_remove_uncertain(len(indices)):
            print(" [REMOVE UNCERTAIN] Cancelled: nothing removed.")
            return False
        before = self._snapshot_crack_edit_state()
        gone = set(indices)
        self.cfg.saved_cracks[:] = [f for i, f in enumerate(self.cfg.saved_cracks) if i not in gone]
        self._commit_crack_edit(before)
        print(f" [REMOVE UNCERTAIN] {len(indices)} uncertain crack(s) removed (were #{numbers}).")
        self.recalculate_masks()
        self._save_annotations_now("REMOVE UNCERTAIN", "the removal")
        self.refresh_zoom_viewport()
        return True
