"""crack_extremes.py
v1.4.0 [Edit extremes] (right sidebar, Edit cracks section, Tools menu, key [4], code 132), mixed into
CrackSegmentation (smart_segmentation.py). Press it once: the crack numbers disappear and every crack shows its two
ends as large red dots. Click (or press and drag) a red dot: the other dots disappear; then click the point the
crack should reach: the crack is lengthened along the crack edge from that end to the clicked point (A*, like a
normal trace). Clicking ON the crack itself instead shortens it to that point; clicking the same dot again drops it
and all the dots come back. Clicks away from the dots trace new cracks as usual (no delete-by-marker while the mode
is on). [U] undoes each change. Press it again: the numbers come back and JSON and binary masks are written.
"""
import cv2
import numpy as np

from crack_relabel import RELABEL_KEEP_KEYS, RELABEL_TARGET_BY_CODE


CRACK_EXTREMES_CODE = 132
CRACK_EXTREMES_KEY = ord('4')
EXTREME_DOT_BGR = (0, 0, 255)
EXTREME_DOT_RING_BGR = (255, 255, 255)
EXTREME_GRAB_BGR = (0, 255, 255)
EXTREME_PREVIEW_BGR = (0, 165, 255)
CRACK_EXTREMES_HUD_BGR = (0, 0, 255)
# Keys that leave the mode on: view keys, Undo, the visibility toggles and Ctrl/Cmd+C / Ctrl/Cmd+U (the relabel
# click is served first, then the red dots are back). Any other command closes it (saved).
EXTREMES_KEEP_KEYS = (RELABEL_KEEP_KEYS | frozenset(RELABEL_TARGET_BY_CODE)
                      | frozenset([ord('u'), ord('U'), CRACK_EXTREMES_CODE, CRACK_EXTREMES_KEY, 130, 131]))
# Navigation that leaves the photo WITHOUT saving: the mode closes without writing the files.
EXTREMES_NO_SAVE_KEYS = frozenset([4, 5, 6, 29])


class CrackExtremesMixin:
    """Lengthening / shortening of traced cracks from their ends. Relies on CrackSegmentation's cfg, A*,
    _stitch, _crack_with_path, Undo and save helpers."""

    # --- Mode ---------------------------------------------------------------------------

    def _toggle_crack_extremes(self):
        """Code 132 / key [4] [Edit extremes]: on = red end dots, drag them to lengthen; second press = save and off."""
        state = self.cfg.crack_extremes_state
        if state["active"]:
            self._close_crack_extremes(save=True)
        else:
            self._open_crack_extremes()
        self.refresh_zoom_viewport()

    def _open_crack_extremes(self):
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._reset_cut_join_state()
        cfg = self.cfg
        cfg.edit_mode = False
        cfg.current_tool, cfg.temp_start, cfg.temp_path = 'crack', None, []
        cfg.trace_waypoints, cfg.trace_partial_path = [], []
        cfg.crack_extremes_state = {"active": True, "grab": None, "cursor": None, "changed": False}
        print(" [EDIT EXTREMES] On: click a red end dot, then click where the crack should reach "
              "(on the crack itself = shorten). Press Edit extremes again to save.")

    def _close_crack_extremes(self, save=True):
        """Mode off: numbers back; with save=True and any change, JSON and masks are written."""
        state = self.cfg.crack_extremes_state
        changed = state["changed"]
        self.cfg.crack_extremes_state = {"active": False, "grab": None, "cursor": None, "changed": False}
        self.recalculate_masks()
        if changed and save:
            self._save_annotations_now("EDIT EXTREMES", "the change")
        elif changed:
            print(" [EDIT EXTREMES] Off: the changes are on screen but were not written.")
        else:
            print(" [EDIT EXTREMES] Off: nothing changed.")

    def _close_crack_extremes_for_key(self, key_clean):
        """Another command was chosen while the mode is on: it closes (saving, unless leaving without saving)."""
        if not self.cfg.crack_extremes_state["active"] or key_clean in EXTREMES_KEEP_KEYS:
            return
        self._close_crack_extremes(save=key_clean not in EXTREMES_NO_SAVE_KEYS)

    def _reset_crack_extremes_for_new_image(self):
        self.cfg.crack_extremes_state = {"active": False, "grab": None, "cursor": None, "changed": False}

    def _crack_extremes_active(self):
        return self.cfg.crack_extremes_state["active"]

    # --- Picking ------------------------------------------------------------------------

    def _crack_extreme_under(self, x, y):
        """(crack_idx, 'start' | 'end') of the end dot nearest to window (x, y) within the pick radius, else None."""
        best, best_d = None, float(self.cfg.EXTREME_PICK_WIN_PX)
        for idx, f in enumerate(self.cfg.saved_cracks):
            path = f.get('path')
            if not f.get('active', True) or not path or len(path) < 2 or not self._crack_shown(f):
                continue
            for which, pt in (('start', path[0]), ('end', path[-1])):
                wx, wy = self.transform_real_to_window_coords(int(pt[0]), int(pt[1]))
                d = float(np.hypot(wx - x, wy - y))
                if d <= best_d:
                    best, best_d = (idx, which), d
        return best

    def _grabbed_extreme_point(self):
        """Image point of the grabbed end, or None (also when the crack no longer exists)."""
        grab = self.cfg.crack_extremes_state["grab"]
        if grab is None or not (0 <= grab[0] < len(self.cfg.saved_cracks)):
            return None
        path = self.cfg.saved_cracks[grab[0]].get('path') or []
        if len(path) < 2:
            return None
        return path[0] if grab[1] == 'start' else path[-1]

    # --- Mouse --------------------------------------------------------------------------

    def _handle_crack_extremes_mouse(self, event, x, y, flags=0):
        """A red dot (or the target of a picked one) is handled here; any other click traces a new crack as usual."""
        state = self.cfg.crack_extremes_state
        if state["grab"] is None and self._extremes_click_traces(event, x, y):
            # Ctrl flag: no delete-by-marker while the large dots are on screen.
            self._handle_crack_tool_mouse(event, x, y, flags | cv2.EVENT_FLAG_CTRLKEY)
            return
        if event == cv2.EVENT_MOUSEMOVE:
            state["cursor"] = (x, y)
            if state["grab"] is not None:
                self.refresh_zoom_viewport()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return   # releasing the button after a drag waits for the click on the target point
        if state["grab"] is None:
            self._grab_crack_extreme(x, y)
        else:
            self._drop_crack_extreme(x, y)
        self.refresh_zoom_viewport()

    def _extremes_owns_event(self, event, x, y):
        """True when the event is about a red end dot (one is picked, or the press lands on one): it goes to
        Edit extremes ahead of Zoom window, which would otherwise turn the drag into a zoom rectangle."""
        if self.cfg.crack_extremes_state["grab"] is not None:
            return True
        return (event == cv2.EVENT_LBUTTONDOWN and self.cfg.temp_start is None
                and self._crack_extreme_under(x, y) is not None)

    def _extremes_click_traces(self, event, x, y):
        """True when the event belongs to crack tracing: a trace is in progress, or a click lands off every dot."""
        if self.cfg.temp_start is not None:
            return True
        return event == cv2.EVENT_LBUTTONDOWN and self._crack_extreme_under(x, y) is None

    def _grab_crack_extreme(self, x, y):
        hit = self._crack_extreme_under(x, y)
        if hit is None:
            print(" [EDIT EXTREMES] Click a red dot at the start or end of a crack.")
            return
        self.cfg.crack_extremes_state["grab"] = hit
        self.cfg.crack_extremes_state["cursor"] = (x, y)
        print(f" [EDIT EXTREMES] {hit[1].capitalize()} of crack {hit[0] + 1} picked: click the point it should reach "
              "(the same dot again = drop it).")

    def _drop_crack_extreme(self, x, y):
        """Second click: same dot = release it; elsewhere = new end of the crack."""
        state = self.cfg.crack_extremes_state
        idx, which = state["grab"]
        anchor = self._grabbed_extreme_point()
        if anchor is None:
            state["grab"] = None
            return
        ax, ay = self.transform_real_to_window_coords(int(anchor[0]), int(anchor[1]))
        if np.hypot(ax - x, ay - y) <= self.cfg.EXTREME_PICK_WIN_PX:
            state["grab"] = None
            print(" [EDIT EXTREMES] Dot released: nothing changed.")
            return
        target = self.transform_window_to_real_coords(x, y)
        target = (int(target[0]), int(target[1]))
        if self._move_crack_extreme(idx, which, target):
            state["grab"], state["changed"] = None, True
            self.recalculate_masks()

    # --- Path change --------------------------------------------------------------------

    def _move_crack_extreme(self, idx, which, target):
        """New end of crack idx at target: shortened when target lies on the crack, lengthened otherwise.
        Returns True when the crack changed (recorded for Undo)."""
        crack = self.cfg.saved_cracks[idx]
        path = [(int(p[0]), int(p[1])) for p in crack['path']]
        p_idx = self._nearest_own_point(path, target)
        if p_idx is not None:
            result = self._shortened_crack_path(path, which, p_idx, crack.get('width_segments'))
        else:
            result = self._lengthened_crack_path(path, which, target, crack.get('width_segments'))
        if result is None:
            return False
        new_path, segments, note = result
        before = self._snapshot_crack_edit_state()
        self.cfg.saved_cracks[idx] = self._crack_with_path(crack, new_path, segments)
        self._commit_crack_edit(before)
        print(f" [EDIT EXTREMES] Crack {idx + 1}: {note} ({len(path)} -> {len(new_path)} points). [U] undoes it.")
        return True

    def _nearest_own_point(self, path, target):
        """Index of the crack's own point nearest to target when within the pick tolerance, else None."""
        pts = np.asarray(path, dtype=np.float64)
        d = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1])
        p_idx = int(np.argmin(d))
        return p_idx if d[p_idx] <= self._cut_join_pick_tolerance() else None

    def _shortened_crack_path(self, path, which, p_idx, segments):
        """(path, width tracts, note) cut at point p_idx on the grabbed side, or None if too little is left."""
        first, last = (p_idx, len(path) - 1) if which == 'start' else (0, p_idx)
        if last - first < 1 or (first, last) == (0, len(path) - 1):
            print(" [EDIT EXTREMES] That would leave no crack: click farther from the other end, or off the crack "
                  "to lengthen it.")
            return None
        kept, dropped = self._remap_width_segments(segments, first, last)
        note = f"{which} moved back along the crack" + (f", {dropped} width tract(s) removed" if dropped else "")
        return path[first:last + 1], kept, note

    def _lengthened_crack_path(self, path, which, target, segments):
        """(path, width tracts, note) with the A* route from the grabbed end to target added, or None."""
        anchor = path[0] if which == 'start' else path[-1]
        route = self.a_star_pathfinding(anchor, target, 'crack')
        if not route:
            print(" [EDIT EXTREMES] No route along the crack to that point: click closer to the crack.")
            return None
        if which == 'start':
            prefix = self._stitch(list(route)[::-1], [anchor])
            new_path = prefix[:-1] + path
            kept, _ = self._remap_width_segments(segments, 0, len(path) - 1, shift=len(prefix) - 1)
        else:
            suffix = self._stitch([anchor], list(route))
            new_path = path + suffix[1:]
            kept = [dict(s) for s in segments or []]
        if len(new_path) == len(path):
            print(" [EDIT EXTREMES] The route adds nothing: click farther along the crack.")
            return None
        return new_path, kept, f"{which} lengthened"

    # --- Drawing ------------------------------------------------------------------------

    def _draw_crack_extremes(self, win_out):
        """Large red dots on every crack end (no numbers); once one is picked only that dot stays, ringed, with
        the line to the cursor (the others come back when it is released)."""
        if not self.cfg.crack_extremes_state["active"]:
            return
        r = self.cfg.EXTREME_DOT_WIN_PX
        self._draw_crack_extreme_drag(win_out, r)   # under the dot, so the preview line starts at its edge
        grabbed = self._grabbed_extreme_point()
        points = [grabbed] if grabbed is not None else self._all_crack_extreme_points()
        for pt in points:
            wx, wy = self.transform_real_to_window_coords(int(pt[0]), int(pt[1]))
            cv2.circle(win_out, (wx, wy), r + 2, EXTREME_DOT_RING_BGR, -1, cv2.LINE_AA)
            cv2.circle(win_out, (wx, wy), r, EXTREME_DOT_BGR, -1, cv2.LINE_AA)
        self._draw_crack_extremes_hud(win_out)

    def _all_crack_extreme_points(self):
        """Start and end points of every visible active crack."""
        points = []
        for f in self.cfg.saved_cracks:
            path = f.get('path')
            if f.get('active', True) and path and len(path) >= 2 and self._crack_shown(f):
                points += [path[0], path[-1]]
        return points

    def _draw_crack_extreme_drag(self, win_out, r):
        anchor = self._grabbed_extreme_point()
        if anchor is None:
            return
        ax, ay = self.transform_real_to_window_coords(int(anchor[0]), int(anchor[1]))
        cursor = self.cfg.crack_extremes_state["cursor"]
        if cursor is not None:
            cv2.line(win_out, (ax, ay), (int(cursor[0]), int(cursor[1])), EXTREME_PREVIEW_BGR, 2, cv2.LINE_AA)
            cv2.circle(win_out, (int(cursor[0]), int(cursor[1])), r, EXTREME_DOT_BGR, 2, cv2.LINE_AA)
        cv2.circle(win_out, (ax, ay), r + 5, EXTREME_GRAB_BGR, 2, cv2.LINE_AA)

    def _draw_crack_extremes_hud(self, win_out):
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 6 * (45 if large else 30)
        text = ("EDIT EXTREMES: click where the crack should reach" if self.cfg.crack_extremes_state["grab"]
                else "EDIT EXTREMES: click a red end dot (button or [4] again = save)")
        self._put_hud_text(win_out, text, (15, y), CRACK_EXTREMES_HUD_BGR)
