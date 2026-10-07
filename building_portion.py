"""building_portion.py
v1.0.7 Building portion, mixed into CrackSegmentation (smart_segmentation.py).
For a photo that matches the other photos of its group only in part, the sidebar button (code 20 in
process_keypress()) or key [5] lets the operator drag a window over the matching part of the building. While a
portion is set, [W]/[L] imports align on that part only (SIFT features inside the window) and keep only
the projected shapes lying inside it. Drag inside the window to move it; a click outside it removes it.
When the tool is turned on, the part of the photo shared with the other photos of the group is found and
tinted orange (compatible_area.py), and the window is pre-set around it.
"""
import cv2
import numpy as np


BUILDING_PORTION_CODE = 20
BUILDING_PORTION_BGR = (255, 160, 60)   # light blue dashed window


class BuildingPortionMixin:
    """Building portion tool. Relies on CrackSegmentation's cfg, coordinate helpers and the dashed-rect drawing."""

    # --- state ------------------------------------------------------------------------------

    def _reset_building_portion(self):
        """New photo: the window belongs to the photo it was drawn on."""
        self.cfg.portion_state.update(active=False, rect=None, drag=None, anchor=None,
                                      rect_at_press=None, press_window_xy=None, compatible=None)

    def _building_portion_button(self):
        """Code 20 / key [5] [Building portion]: first press = drag a window over the matching part of the building;
        second press = done (the window stays and limits [W]/[L] until removed)."""
        st = self.cfg.portion_state
        if st["active"]:
            st["active"], st["drag"] = False, None
            if st["rect"] is None:
                print(" [BUILDING PORTION] Closed: no window set, imports use the whole photo.")
            else:
                print(" [BUILDING PORTION] Window set: [W]/[L] now import only inside it. "
                      "Press [5] again to move or remove it.")
        else:
            self._close_other_tools_for_portion()
            st["active"] = True
            if st.get("compatible") is None:
                self._find_compatible_area()  # once per photo: the orange shared area
            if st["rect"] is None and self._compatible_area_rect() is not None:
                st["rect"] = self._compatible_area_rect()
                print(" [BUILDING PORTION] Window pre-set around the orange area.")
            print(" [BUILDING PORTION] Drag a window over the part of the building shared with the other photos "
                  "(orange); drag inside it to move it, click outside it to remove it. Press [5] again when done.")
        self.refresh_zoom_viewport()

    def _close_other_tools_for_portion(self):
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._reset_cut_join_state()
        self._stop_crack_report(hide_panel=True)
        self._reset_alt_route_state()
        self.cfg.edit_mode = False
        self.cfg.temp_start = None
        self.cfg.temp_path.clear()

    # --- mouse ---------------------------------------------------------------------------------

    def _clip_portion_point(self, p):
        return (int(min(max(p[0], 0), self.cfg.W_img - 1)), int(min(max(p[1], 0), self.cfg.H_img - 1)))

    @staticmethod
    def _point_in_rect(p, rect):
        return rect is not None and rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]

    def _portion_moved_enough(self, x, y):
        press = self.cfg.portion_state["press_window_xy"]
        return press is None or max(abs(x - press[0]), abs(y - press[1])) >= self.cfg.ZOOM_WINDOW_MIN_DRAG_PX

    def _handle_building_portion_mouse(self, event, x, y):
        """Press + drag outside the window = new window; press + drag inside = move it; plain click outside = remove it."""
        st = self.cfg.portion_state
        p = self._clip_portion_point(self.transform_window_to_real_coords(x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = "move" if self._point_in_rect(p, st["rect"]) else "new"
            st["anchor"], st["rect_at_press"], st["press_window_xy"] = p, st["rect"], (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"] and self._portion_moved_enough(x, y):
            st["rect"] = self._dragged_portion_rect(st, p)
        elif event == cv2.EVENT_LBUTTONUP and st["drag"]:
            self._finish_portion_drag(st, p, self._portion_moved_enough(x, y))
            self.refresh_zoom_viewport()

    def _dragged_portion_rect(self, st, p):
        """The window while dragging: spanned from the press point, or shifted (kept inside the photo)."""
        ax, ay = st["anchor"]
        if st["drag"] == "new":
            return (min(ax, p[0]), min(ay, p[1]), max(ax, p[0]), max(ay, p[1]))
        x0, y0, x1, y1 = st["rect_at_press"]
        dx = int(np.clip(p[0] - ax, -x0, self.cfg.W_img - 1 - x1))
        dy = int(np.clip(p[1] - ay, -y0, self.cfg.H_img - 1 - y1))
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def _finish_portion_drag(self, st, p, moved):
        drag, st["drag"], st["press_window_xy"] = st["drag"], None, None
        if not moved:
            st["rect"] = st["rect_at_press"]
            if drag == "new" and st["rect"] is not None:
                st["rect"] = None
                print(" [BUILDING PORTION] Window removed: imports use the whole photo again.")
            return
        if drag == "new":
            x0, y0, x1, y1 = st["rect"]
            min_side = max(8, int(0.02 * max(self.cfg.W_img, self.cfg.H_img)))
            if x1 - x0 < min_side or y1 - y0 < min_side:
                st["rect"] = st["rect_at_press"]
                print(" [BUILDING PORTION] Window too small: press and drag to draw it.")
                return
        x0, y0, x1, y1 = st["rect"]
        print(f" [BUILDING PORTION] Window {x1 - x0}x{y1 - y0} px at ({x0}, {y0}). Press [5] again when done.")

    # --- import ---------------------------------------------------------------------------------

    def _portion_rect_for(self, w2, h2):
        """The window scaled to an image of w2 x h2 pixels (the photo as read for the import), or None."""
        rect = self.cfg.portion_state["rect"]
        if rect is None or self.cfg.W_img <= 0 or self.cfg.H_img <= 0:
            return None
        sx, sy = w2 / float(self.cfg.W_img), h2 / float(self.cfg.H_img)
        return (rect[0] * sx, rect[1] * sy, (rect[2] + 1) * sx, (rect[3] + 1) * sy)

    def _portion_target_mask(self, img_shape):
        """SIFT mask of the target photo limited to the window (None = whole photo)."""
        h2, w2 = img_shape[:2]
        rect = self._portion_rect_for(w2, h2)
        if rect is None:
            return None
        mask = np.zeros((h2, w2), dtype=np.uint8)
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        mask[max(0, y0):min(h2, y1), max(0, x0):min(w2, x1)] = 255
        return mask

    def _keep_shapes_in_portion(self, shapes, w2, h2):
        """Projected shapes with at least IMPORT_PORTION_MIN_INSIDE of their points inside the window.
        Returns (kept, number dropped); everything is kept when no window is set."""
        rect = self._portion_rect_for(w2, h2)
        if rect is None:
            return shapes, 0
        x0, y0, x1, y1 = rect
        kept = []
        for shape in shapes:
            pts = np.asarray(shape.get("points") or [], dtype=np.float64).reshape(-1, 2)
            if len(pts) == 0:
                continue
            inside = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            if inside.mean() >= self.cfg.IMPORT_PORTION_MIN_INSIDE:
                kept.append(shape)
        return kept, len(shapes) - len(kept)

    # --- drawing --------------------------------------------------------------------------------

    def _draw_building_portion(self, win_out):
        """The window as a dashed rectangle, shown whenever one is set; a prompt while the tool is on."""
        st = self.cfg.portion_state
        self._draw_compatible_area(win_out)
        if st["rect"] is not None:
            x0, y0, x1, y1 = st["rect"]
            p1 = self.transform_real_to_window_coords(x0, y0)
            p2 = self.transform_real_to_window_coords(x1, y1)
            self._draw_dashed_rect(win_out, p1, p2, BUILDING_PORTION_BGR)
        if st["active"]:
            y = 135 if self.cfg.hud_large_size else 120
            text = "BUILDING PORTION: drag a window" if st["rect"] is None \
                else "BUILDING PORTION: drag inside = move, click outside = remove"
            self._put_hud_text(win_out, text, (15, y), BUILDING_PORTION_BGR)
