"""ignore_regions.py
v1.0.7 ignore regions mixed into CrackSegmentation (smart_segmentation.py).
[I] (or the "Ignore region" sidebar button) opens a small options dialog, then selects the tool:
  - Rectangle / Square: two clicks (opposite corners); each side ticked in the dialog is pushed onto that photo edge.
  - Polygon: click the corners, close with [Y] (corners near a photo edge snap onto it).
Click inside a saved region (with nothing in progress) to delete it; drag a corner (or a rectangle side) to reshape it. Occluders: safety nets, scaffolding, cables, vegetation.
Saved in the JSON with label "ignore" and exported as <photo>-ignore_mask.png (255 = exclude from loss and metrics).
"""
import time

import cv2
import numpy as np


IGNORE_OUTLINE_BGR = (255, 0, 255)   # magenta outline and corner markers
IGNORE_FILL_BGR = (70, 70, 70)       # dark grey glass
IGNORE_FILL_ALPHA = 0.45
IGNORE_HATCH_SPACING_PX = 16         # window pixels between hatch lines
IGNORE_HANDLE_WIN_PX = 12            # grab distance (window pixels) for a corner or a rectangle side
IGNORE_EDGE_SNAP_WIN_PX = 15         # a polygon corner this close to a photo edge (window pixels) snaps onto it

IGNORE_SHAPES = ("rectangle", "square", "polygon")
IGNORE_EDGES = ("left", "right", "top", "bottom")
DEFAULT_IGNORE_OPTIONS = {"shape": "rectangle", "edges": {e: False for e in IGNORE_EDGES}}


class IgnoreRegionsMixin:
    """[I] ignore-region tool. Relies on CrackSegmentation's cfg, undo history and coordinate helpers."""

    # --- tool selection ---------------------------------------------------------------

    def _discard_in_progress_ignore(self):
        """Drops the corners of a polygon not yet closed with [Y], and any half-done reshape."""
        if self.cfg.current_tool == 'ignore':
            self.cfg.temp_nodes, self.cfg.temp_path = [], []
        self.cfg.ignore_drag = None

    def _ignore_options(self):
        """Current dialog choices (shape + edges to extend to), remembered for the session."""
        opts = getattr(self.cfg, "ignore_options", None)
        if opts is None:
            opts = {"shape": DEFAULT_IGNORE_OPTIONS["shape"], "edges": dict(DEFAULT_IGNORE_OPTIONS["edges"])}
            self.cfg.ignore_options = opts
        return opts

    def _prompt_ignore_options(self, current):
        """Asks for shape and edges. Returns a new options dict, or None to cancel. Default (no GUI): keeps the
        current options without asking; the GUI wrapper overrides this with a small dialog."""
        return current

    @staticmethod
    def _normalize_ignore_options(opts):
        shape = opts.get("shape") if opts.get("shape") in IGNORE_SHAPES else "rectangle"
        edges = {e: bool((opts.get("edges") or {}).get(e)) for e in IGNORE_EDGES}
        return {"shape": shape, "edges": edges}

    def _select_ignore_tool(self):
        """[I]: asks for the ignore options, then switches to the tool (same cleanup as [C]/[D]). Cancel = no change."""
        chosen = self._prompt_ignore_options(self._ignore_options())
        if chosen is None:
            print(" [IGNORE] Cancelled: tool unchanged.")
            return
        self.cfg.ignore_options = self._normalize_ignore_options(chosen)
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._reset_cut_join_state()
        self.cfg.current_tool, self.cfg.temp_start = 'ignore', None
        self.cfg.temp_nodes, self.cfg.temp_path = [], []
        self.cfg.user_clicked_nodes = []
        print(self._ignore_mode_message())

    def _ignore_mode_message(self):
        opts = self._ignore_options()
        if opts["shape"] == "polygon":
            how = "click the corners of the occluded area, [Y] to close it"
        else:
            edges = [e for e in IGNORE_EDGES if opts["edges"][e]]
            reach = f", extended to the {'/'.join(edges)} edge(s)" if edges else ""
            how = f"click two opposite corners of the {opts['shape']}{reach}"
        return f" [MODE] IGNORE tool: {how}. Click inside a saved region to delete it. [C] back to cracks."

    # --- rectangle / square -----------------------------------------------------------

    def _ignore_rect_corners(self, p1, p2):
        """Four corners of the p1-p2 rectangle (a square uses the longer side, growing toward p2),
        with every edge ticked in the options pushed onto that photo edge."""
        opts = self._ignore_options()
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        if opts["shape"] == "square":
            side = max(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + (side if x1 >= x0 else -side)
            y1 = y0 + (side if y1 >= y0 else -side)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        max_x, max_y = self.cfg.W_img - 1, self.cfg.H_img - 1
        edges = opts["edges"]
        left = 0 if edges["left"] else max(0, left)
        right = max_x if edges["right"] else min(max_x, right)
        top = 0 if edges["top"] else max(0, top)
        bottom = max_y if edges["bottom"] else min(max_y, bottom)
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def _ignore_rect_click(self, real_x, real_y):
        """First click: anchors a corner. Second click: saves the rectangle (a zero-area one is ignored)."""
        if not self.cfg.temp_nodes:
            self.cfg.redo_history.clear()
            self.cfg.temp_nodes = [(int(real_x), int(real_y))]
            self.cfg.temp_path = self._ignore_rect_corners(self.cfg.temp_nodes[0], (real_x, real_y))
            self.refresh_zoom_viewport()
            return
        corners = self._ignore_rect_corners(self.cfg.temp_nodes[0], (real_x, real_y))
        self.cfg.temp_nodes, self.cfg.temp_path = [], []
        if corners[0][0] == corners[1][0] or corners[0][1] == corners[2][1]:
            print(" [IGNORE] Empty rectangle: nothing saved.")
            self.refresh_zoom_viewport()
            return
        self._save_ignore_region(corners)

    # --- drawing a polygon ------------------------------------------------------------

    def _delete_ignore_at(self, real_x, real_y):
        """Deletes the ignore polygon containing the click (the most recent one where regions overlap). Returns True if one was."""
        for idx in range(len(self.cfg.saved_ignores) - 1, -1, -1):
            path = self.cfg.saved_ignores[idx].get('path') or []
            if len(path) < 3:
                continue
            contour = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(contour, (float(real_x), float(real_y)), False) >= 0:
                removed = self.cfg.saved_ignores.pop(idx)
                self.cfg.action_history.append(('ignore_delete', removed))
                print(" [IGNORE] Region deleted ([U] to restore it).")
                self.refresh_zoom_viewport()
                return True
        return False

    def _snap_to_photo_edges(self, real_x, real_y):
        """Moves a corner within IGNORE_EDGE_SNAP_WIN_PX (screen) of a photo edge exactly onto it,
        so an occluder reaching the edge leaves no unmasked strip. The tolerance follows the zoom."""
        z_xmin, z_ymin, z_xmax, z_ymax = self.cfg.zoom_box
        tol_x = IGNORE_EDGE_SNAP_WIN_PX * max(1, z_xmax - z_xmin) / 1200.0
        tol_y = IGNORE_EDGE_SNAP_WIN_PX * max(1, z_ymax - z_ymin) / 900.0
        max_x, max_y = self.cfg.W_img - 1, self.cfg.H_img - 1
        x = 0 if real_x <= tol_x else (max_x if real_x >= max_x - tol_x else int(real_x))
        y = 0 if real_y <= tol_y else (max_y if real_y >= max_y - tol_y else int(real_y))
        return x, y

    def _add_ignore_node(self, real_x, real_y):
        """Adds one straight-edged corner (no A* snapping), snapped onto a photo edge when close to it."""
        if not self.cfg.temp_nodes:
            self.cfg.redo_history.clear()
        self.cfg.temp_nodes.append(self._snap_to_photo_edges(real_x, real_y))
        self.cfg.temp_path = list(self.cfg.temp_nodes)
        self.refresh_zoom_viewport()

    def _ignore_tool_click(self, x, y):
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        # Grabbing/deleting only while nothing is in progress, so corners can still be placed over an existing region.
        if not self.cfg.temp_nodes and self._start_ignore_drag(x, y):
            return
        if not self.cfg.temp_nodes and self._delete_ignore_at(real_x, real_y):
            return
        if self._ignore_options()["shape"] == "polygon":
            self._add_ignore_node(real_x, real_y)
        else:
            self._ignore_rect_click(real_x, real_y)

    def _ignore_tool_preview(self, x, y):
        """Live preview: the rectangle as it would be saved, or the rubber-band edge of a polygon."""
        if self.cfg.temp_nodes:
            real_x, real_y = self.transform_window_to_real_coords(x, y)
            if self._ignore_options()["shape"] != "polygon":
                self.cfg.temp_path = self._ignore_rect_corners(self.cfg.temp_nodes[0], (real_x, real_y))
                self.refresh_zoom_viewport()
                return
            self.cfg.temp_path = list(self.cfg.temp_nodes) + [self._snap_to_photo_edges(real_x, real_y)]
            self.refresh_zoom_viewport()

    def _handle_ignore_tool_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._ignore_tool_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._ignore_drag_state() is not None:
                self._drag_ignore_handle(x, y)
            else:
                self._ignore_tool_preview(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._end_ignore_drag()

    # --- reshaping a saved region -------------------------------------------------------

    @staticmethod
    def _is_axis_rectangle(path):
        """True for a 4-corner axis-aligned region (rectangle/square tool): edited by corners and sides."""
        if len(path) != 4:
            return False
        xs = sorted({int(p[0]) for p in path})
        ys = sorted({int(p[1]) for p in path})
        return len(xs) == 2 and len(ys) == 2 and all((int(p[0]), int(p[1])) in
                                                     {(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])}
                                                     for p in path)

    def _ignore_drag_state(self):
        return getattr(self.cfg, "ignore_drag", None)

    def _find_ignore_handle(self, x, y):
        """(region, kind, key) of the corner or rectangle side under window point (x, y), most recent region first.
        kind 'vertex' -> key = corner index (polygon) or ('l'|'r', 't'|'b') (rectangle); kind 'side' -> 'l'|'r'|'t'|'b'."""
        tol = IGNORE_HANDLE_WIN_PX
        for region in reversed(self.cfg.saved_ignores):
            path = region.get('path') or []
            if len(path) < 3:
                continue
            win = self._ignore_polygon_window_pts(path)
            is_rect = self._is_axis_rectangle(path)
            for i, (wx, wy) in enumerate(win):
                if abs(wx - x) <= tol and abs(wy - y) <= tol:
                    if not is_rect:
                        return region, 'vertex', i
                    l, t, r, b = self._rect_bounds(path)
                    px, py = path[i]
                    return region, 'vertex', ('l' if px == l else 'r', 't' if py == t else 'b')
            if is_rect:
                side = self._rect_side_under(win, x, y, tol)
                if side is not None:
                    return region, 'side', side
        return None

    @staticmethod
    def _rect_bounds(path):
        xs = [int(p[0]) for p in path]
        ys = [int(p[1]) for p in path]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _rect_side_under(win, x, y, tol):
        wl, wt = int(min(p[0] for p in win)), int(min(p[1] for p in win))
        wr, wb = int(max(p[0] for p in win)), int(max(p[1] for p in win))
        if wt - tol <= y <= wb + tol:
            if abs(x - wl) <= tol:
                return 'l'
            if abs(x - wr) <= tol:
                return 'r'
        if wl - tol <= x <= wr + tol:
            if abs(y - wt) <= tol:
                return 't'
            if abs(y - wb) <= tol:
                return 'b'
        return None

    def _start_ignore_drag(self, x, y):
        """Press on a corner/side: starts reshaping that region. Returns True if one was grabbed."""
        hit = self._find_ignore_handle(x, y)
        if hit is None:
            return False
        region, kind, key = hit
        self.cfg.ignore_drag = {"region": region, "kind": kind, "key": key, "old_path": list(region['path'])}
        return True

    def _drag_ignore_handle(self, x, y):
        drag = self._ignore_drag_state()
        region = drag["region"]
        px, py = self._snap_to_photo_edges(*self.transform_window_to_real_coords(x, y))
        if isinstance(drag["key"], int):
            path = list(region['path'])
            path[drag["key"]] = (px, py)
            region['path'] = path
        else:
            l, t, r, b = self._rect_bounds(drag["old_path"])
            sides = drag["key"] if drag["kind"] == 'vertex' else (drag["key"],)
            for side in sides:
                if side == 'l':
                    l = px
                elif side == 'r':
                    r = px
                elif side == 't':
                    t = py
                elif side == 'b':
                    b = py
            l, r = sorted((l, r))
            t, b = sorted((t, b))
            region['path'] = [(l, t), (r, t), (r, b), (l, b)]
        self.refresh_zoom_viewport()

    def _end_ignore_drag(self):
        """Release: keeps the new shape (undoable with [U]); a degenerate result reverts to the old one."""
        drag = self._ignore_drag_state()
        if drag is None:
            return
        self.cfg.ignore_drag = None
        region, old_path = drag["region"], drag["old_path"]
        if self._ignore_area(region['path']) < 1:
            region['path'] = old_path
            print(" [IGNORE] Shape too small: change undone.")
        elif list(region['path']) != list(old_path):
            self.cfg.redo_history.clear()
            self.cfg.action_history.append(('ignore_edit', (region, old_path)))
            print(" [IGNORE] Region reshaped ([U] to undo).")
        self.refresh_zoom_viewport()

    @staticmethod
    def _ignore_area(path):
        if len(path) < 3:
            return 0.0
        return abs(cv2.contourArea(np.array(path, dtype=np.float32).reshape(-1, 1, 2)))

    def _close_ignore_polygon(self):
        """[Y] with the ignore tool: saves the polygon (needs >=3 corners)."""
        if not (self.cfg.current_tool == 'ignore' and self._ignore_options()["shape"] == "polygon"
                and len(self.cfg.temp_nodes) >= 3):
            return False
        corners = list(self.cfg.temp_nodes)
        self.cfg.temp_nodes, self.cfg.temp_path = [], []
        self._save_ignore_region(corners)
        return True

    def _save_ignore_region(self, corners):
        self.cfg.saved_ignores.append({'path': list(corners), 'active': True, 'session_id': str(time.time())})
        self.cfg.action_history.append('ignore')
        print(f" [IGNORE] Region saved ({len(self.cfg.saved_ignores)} in this photo).")
        self.refresh_zoom_viewport()

    def _close_polygon_for_current_tool(self):
        """[Y]: closes the polygon of whichever polygon tool is active."""
        if self.cfg.current_tool == 'ignore':
            self._close_ignore_polygon()
        else:
            self._close_detachment_polygon()

    # --- undo / redo ------------------------------------------------------------------

    def _undo_in_progress_ignore_node(self):
        """[U] while clicking corners: removes just the last one. Returns True if this applied."""
        if not (self.cfg.current_tool == 'ignore' and self.cfg.temp_nodes):
            return False
        self.cfg.temp_nodes.pop()
        self.cfg.temp_path = list(self.cfg.temp_nodes)
        return True

    def _undo_ignore_action(self, last_action):
        """Undoes a saved/deleted ignore polygon. Returns True if last_action was one."""
        if last_action == 'ignore':
            if self.cfg.saved_ignores:
                self.cfg.redo_history.append(('ignore', self.cfg.saved_ignores.pop()))
            return True
        if isinstance(last_action, tuple) and last_action and last_action[0] == 'ignore_delete':
            self.cfg.saved_ignores.append(last_action[1])
            self.cfg.redo_history.append(('ignore_redelete', last_action[1]))
            return True
        if isinstance(last_action, tuple) and last_action and last_action[0] == 'ignore_edit':
            region, old_path = last_action[1]
            self.cfg.redo_history.append(('ignore_edit', (region, list(region['path']))))
            region['path'] = list(old_path)
            return True
        return False

    def _redo_ignore_action(self, action_type, restored):
        """Redoes an undone ignore save or delete. Returns True if action_type was one."""
        if action_type == 'ignore':
            self.cfg.saved_ignores.append(restored)
            self.cfg.action_history.append('ignore')
            print("[REDO] Ignore region restored.")
            return True
        if action_type == 'ignore_redelete':
            self.cfg.saved_ignores[:] = [r for r in self.cfg.saved_ignores if r is not restored]
            self.cfg.action_history.append(('ignore_delete', restored))
            print("[REDO] Ignore region deleted again.")
            return True
        if action_type == 'ignore_edit':
            region, new_path = restored
            self.cfg.action_history.append(('ignore_edit', (region, list(region['path']))))
            region['path'] = list(new_path)
            print("[REDO] Ignore region reshaped again.")
            return True
        return False

    # --- JSON and mask export ---------------------------------------------------------

    def _is_ignore_shape(self, shape):
        """True for an ignore polygon in a LabelMe JSON (label "ignore" or "ignore_<photo>", or the ignore flag)."""
        label = str(shape.get("label", ""))
        base = self.cfg.IGNORE_LABEL
        return bool((shape.get("flags") or {}).get("ignore")) or label == base or label.startswith(base + "_")

    def _build_ignore_shapes(self):
        """LabelMe polygon shapes for every active ignore region."""
        return [{"label": self.cfg.IGNORE_LABEL,
                 "points": [[float(pt[0]), float(pt[1])] for pt in r['path']],
                 "group_id": None, "shape_type": "polygon", "flags": {"ignore": True}}
                for r in self.cfg.saved_ignores if r.get('active', True) and len(r.get('path', [])) >= 3]

    @staticmethod
    def _build_loaded_ignore(points):
        return {'path': points, 'active': True, 'session_id': 'loaded'}

    def _build_ignore_area_mask(self):
        """Filled binary mask of every active ignore polygon."""
        mask = np.zeros((self.cfg.H_img, self.cfg.W_img), dtype=np.uint8)
        for r in self.cfg.saved_ignores:
            if r.get('active', True) and len(r.get('path', [])) >= 3:
                pts = np.array([[int(p[0]), int(p[1])] for p in r['path']], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def _export_ignore_mask(self):
        """Writes <photo>-ignore_mask.png; removes a stale one when no region is left."""
        if not self.cfg.IGNORE_EXPORT_MASK:
            return
        path = self._segmented_output_path("-ignore_mask", ".png")
        n = sum(1 for r in self.cfg.saved_ignores if r.get('active', True))
        self._write_mask_if_nonempty(self._build_ignore_area_mask(), path, f"Ignore mask ({n} region(s))")

    # --- on-screen rendering ----------------------------------------------------------

    def _ignore_polygon_window_pts(self, path):
        return np.array([self.transform_real_to_window_coords(int(p[0]), int(p[1])) for p in path], dtype=np.int32)

    @staticmethod
    def _paint_hatched_glass(win_out, polys):
        """Grey translucent fill with diagonal hatching, clipped to the polygons (window coordinates)."""
        h, w = win_out.shape[:2]
        area = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(area, polys, 255)
        if not np.any(area):
            return
        tinted = win_out.copy()
        tinted[area == 255] = IGNORE_FILL_BGR
        blended = cv2.addWeighted(tinted, IGNORE_FILL_ALPHA, win_out, 1 - IGNORE_FILL_ALPHA, 0)
        # Opaque hatching on top of the glass, so the region reads clearly even over busy textures.
        for c in range(-h, w, IGNORE_HATCH_SPACING_PX):
            cv2.line(blended, (c, 0), (c + h, h), IGNORE_OUTLINE_BGR, 1, cv2.LINE_AA)
        win_out[area == 255] = blended[area == 255]

    def _draw_ignore_regions(self, win_out, marker_dim):
        """Saved regions (always visible, like the detachment fill) plus the polygon being clicked out."""
        polys = [self._ignore_polygon_window_pts(r['path']) for r in self.cfg.saved_ignores
                 if r.get('active', True) and len(r.get('path', [])) >= 3]
        if polys:
            self._paint_hatched_glass(win_out, polys)
            cv2.polylines(win_out, polys, True, IGNORE_OUTLINE_BGR, 2)
            if self.cfg.current_tool == 'ignore' and self.cfg.show_markers:
                self._draw_ignore_handles(win_out, polys)
        self._draw_in_progress_ignore(win_out, marker_dim)

    @staticmethod
    def _draw_ignore_handles(win_out, polys):
        """Small filled squares on every corner: drag them to reshape the region."""
        for poly in polys:
            for wx, wy in poly:
                cv2.rectangle(win_out, (int(wx) - 5, int(wy) - 5), (int(wx) + 5, int(wy) + 5), IGNORE_OUTLINE_BGR, -1)
                cv2.rectangle(win_out, (int(wx) - 5, int(wy) - 5), (int(wx) + 5, int(wy) + 5), (255, 255, 255), 1)

    @staticmethod
    def _draw_first_corner_marker(win_out, corner, marker_dim):
        """Hollow square on the first corner of the polygon being drawn."""
        x, y = int(corner[0]), int(corner[1])
        half = max(4, marker_dim // 2)
        cv2.rectangle(win_out, (x - half, y - half), (x + half, y + half), IGNORE_OUTLINE_BGR, 2)

    def _draw_in_progress_ignore(self, win_out, marker_dim):
        if not (self.cfg.current_tool == 'ignore' and self.cfg.temp_nodes):
            return
        trace = self._ignore_polygon_window_pts(self.cfg.temp_path or self.cfg.temp_nodes)
        is_rect = self._ignore_options()["shape"] != "polygon"
        cv2.polylines(win_out, [trace], is_rect, IGNORE_OUTLINE_BGR, 2)
        for corner in self._ignore_polygon_window_pts(self.cfg.temp_nodes):
            cv2.circle(win_out, (int(corner[0]), int(corner[1])), 4, (0, 255, 255), -1)
        self._draw_first_corner_marker(win_out, self._ignore_polygon_window_pts(self.cfg.temp_nodes[:1])[0], marker_dim)
