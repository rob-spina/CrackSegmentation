"""route_options.py
v1.0.7 Alternative route, mixed into CrackSegmentation (smart_segmentation.py).
The sidebar button (code 18 in process_keypress()) shows, one route per press, other edge routes between
the start and the end of the crack traced last, each in its own color. When no other route is left the
button's arrow turns left and further presses step back through the routes already shown.
[$] makes the route on screen the crack's new (blue) path and closes the tool.
"""
import cv2
import numpy as np


# One color per alternative route, in the order they are shown (BGR): red, green, orange, magenta, yellow, cyan.
ALT_ROUTE_COLORS_BGR = [(0, 0, 255), (0, 200, 0), (0, 140, 255), (255, 0, 255), (0, 230, 255), (255, 255, 0)]
ALT_ROUTE_COLOR_NAMES = ["red", "green", "orange", "magenta", "yellow", "cyan"]
ALT_ROUTE_HUD_BGR = (235, 235, 235)
ALT_ROUTE_CODE = 18


def alt_route_color(index):
    """(BGR, name) of route index (1 = first alternative); index 0 is the crack's own blue route."""
    k = (index - 1) % len(ALT_ROUTE_COLORS_BGR)
    return ALT_ROUTE_COLORS_BGR[k], ALT_ROUTE_COLOR_NAMES[k]


class AlternativeRoutesMixin:
    """Alternative route tool. Relies on CrackSegmentation's cfg, A* (_route_avoiding) and crack-edit helpers."""

    # --- state ------------------------------------------------------------------------------

    def _reset_alt_route_state(self):
        """Closes the tool and forgets the routes found (the cracks are left as they are)."""
        self.cfg.alt_route_state.update(active=False, crack=None, routes=[], index=0, direction=1, exhausted=False)

    def _reset_alt_routes_for_new_image(self):
        self._reset_alt_route_state()
        self.cfg.last_traced_crack = None

    def _alt_route_target(self):
        """(index, crack) the button re-routes: the crack traced last, else the last crack in the list."""
        cracks = self.cfg.saved_cracks
        usable = [i for i, c in enumerate(cracks) if c.get('active', True) and len(c.get('path') or []) >= 2]
        for i in usable:
            if cracks[i] is self.cfg.last_traced_crack:
                return i, cracks[i]
        if usable:
            return usable[-1], cracks[usable[-1]]
        return None, None

    def _alt_route_crack_index(self):
        """saved_cracks index of the crack being re-routed, or None if it was changed or removed since (e.g. Undo)."""
        crack = self.cfg.alt_route_state["crack"]
        for i, c in enumerate(self.cfg.saved_cracks):
            if c is crack:
                return i
        return None

    # --- route search -------------------------------------------------------------------------

    def _next_alt_route(self):
        """A new edge route between the crack's two ends avoiding every route found so far, or None."""
        routes = self.cfg.alt_route_state["routes"]
        if len(routes) > self.cfg.ALT_ROUTE_MAX_ROUTES:
            return None
        blocked = [r[1:-1] for r in routes] + list(self.cfg.cut_exclusions)
        route = self._route_avoiding(routes[0][0], routes[0][-1], blocked)
        if not route or len(route) < 2:
            return None
        route = [(int(p[0]), int(p[1])) for p in route]
        return route if self._is_plausible_alt_route(route) else None

    def _route_crop(self, routes, margin=4):
        """Bounding box (x0, y0, x1, y1) around the given routes, clipped to the photo."""
        pts = np.concatenate([np.asarray(r, dtype=np.int64).reshape(-1, 2) for r in routes])
        x0, y0 = np.maximum(pts.min(axis=0) - margin, 0)
        x1, y1 = pts.max(axis=0) + margin + 1
        if self.cfg.skeleton_mask is not None:
            h, w = self.cfg.skeleton_mask.shape[:2]
            x1, y1 = min(x1, w), min(y1, h)
        return int(x0), int(y0), int(x1), int(y1)

    def _is_plausible_alt_route(self, route):
        """Offered only if not much longer than the current path, mostly on the detected edges
        and mostly away from the routes already shown."""
        cfg = self.cfg
        routes = cfg.alt_route_state["routes"]
        if self._path_length(route) > cfg.ALT_ROUTE_MAX_DETOUR * self._path_length(routes[0]) + 10:
            return False
        x0, y0, x1, y1 = self._route_crop(routes + [route])
        pts = np.asarray(route, dtype=np.int64)
        xs = np.clip(pts[:, 0] - x0, 0, x1 - x0 - 1)
        ys = np.clip(pts[:, 1] - y0, 0, y1 - y0 - 1)
        if cfg.skeleton_mask is not None:
            edge = cv2.dilate(cfg.skeleton_mask[y0:y1, x0:x1], np.ones((3, 3), np.uint8))
            if np.mean(edge[ys, xs] > 0) < cfg.ALT_ROUTE_MIN_ON_EDGE:
                return False
        shown = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for r in routes:
            arr = (np.asarray(r, dtype=np.int32).reshape(-1, 2) - [x0, y0]).reshape(-1, 1, 2)
            cv2.polylines(shown, [arr.astype(np.int32)], False, 255, thickness=5)
        return np.mean(shown[ys, xs] == 0) >= cfg.ALT_ROUTE_MIN_NEW

    def _prefetch_alt_route(self):
        """Looks one route ahead, so the arrow already points left while the LAST route is on screen."""
        st = self.cfg.alt_route_state
        if st["exhausted"] or st["index"] < len(st["routes"]) - 1:
            return
        route = self._next_alt_route()
        if route is None:
            st["exhausted"] = True
        else:
            st["routes"].append(route)

    def _update_alt_route_direction(self):
        """Back on the blue route the arrow points right again; on the last route found it points left."""
        st = self.cfg.alt_route_state
        if st["index"] <= 0:
            st["direction"] = 1
        elif st["exhausted"] and st["index"] >= len(st["routes"]) - 1:
            st["direction"] = -1

    # --- button and keys ------------------------------------------------------------------------

    def _alt_route_button(self):
        """Code 18 [Alternative route]: first press shows another route; each press moves to the next one
        (arrow right) or, once none is left, back through the ones already shown (arrow left)."""
        st = self.cfg.alt_route_state
        if st["active"] and self._alt_route_crack_index() is None:
            self._reset_alt_route_state()  # the crack changed since (Undo, CUT, ...): start over
        if st["active"]:
            st["index"] += st["direction"]
            if st["direction"] > 0:
                self._prefetch_alt_route()
            self._update_alt_route_direction()
            self._announce_alt_route()
        else:
            self._start_alt_routes()
        self.refresh_zoom_viewport()

    def _close_other_tools_for_alt_route(self):
        self._reset_translate_state()
        self._reset_width_edit_state()
        self._reset_cut_join_state()
        self._stop_crack_report(hide_panel=True)
        self.cfg.portion_state["active"] = False
        self.cfg.edit_mode = False
        self.cfg.temp_start = None
        self.cfg.temp_path.clear()

    def _start_alt_routes(self):
        idx, crack = self._alt_route_target()
        if crack is None:
            print(" [ALT ROUTE] No crack to re-route: trace a crack first.")
            return
        self._close_other_tools_for_alt_route()
        st = self.cfg.alt_route_state
        st.update(active=True, crack=crack, index=0, direction=1, exhausted=False,
                  routes=[[(int(p[0]), int(p[1])) for p in crack['path']]])
        route = self._next_alt_route()
        if route is None:
            self._reset_alt_route_state()
            print(f" [ALT ROUTE] Crack #{idx + 1}: no alternative route between its start and its end.")
            return
        st["routes"].append(route)
        st["index"] = 1
        self._prefetch_alt_route()
        self._update_alt_route_direction()
        self._announce_alt_route()

    def _announce_alt_route(self):
        st = self.cfg.alt_route_state
        idx = self._alt_route_crack_index()
        crack_no = idx + 1 if idx is not None else "?"
        if st["index"] == 0:
            print(f" [ALT ROUTE] Crack #{crack_no}: original route (blue). Press the button again for the alternatives.")
            return
        _, name = alt_route_color(st["index"])
        found = len(st["routes"]) - 1
        more = "press the button for the next one" if st["direction"] > 0 else "no other route: press the button to go back"
        print(f" [ALT ROUTE] Crack #{crack_no}: alternative route {st['index']} of {found}{'' if st['exhausted'] else '+'} "
              f"({name}). [$] uses it, {more}.")

    def _apply_alt_route(self):
        """[$]: the route on screen becomes the crack's path (blue again) and the tool closes; Undo [U] reverts it."""
        st = self.cfg.alt_route_state
        if not st["active"]:
            print(" [ALT ROUTE] [$] applies an alternative route: press the Alternative route button first.")
            return
        idx = self._alt_route_crack_index()
        if idx is None or st["index"] == 0:
            self._reset_alt_route_state()
            print(" [ALT ROUTE] Closed: the crack keeps its route.")
            self.refresh_zoom_viewport()
            return
        crack = self.cfg.saved_cracks[idx]
        n_tracts = len(crack.get('width_segments') or [])
        before = self._snapshot_crack_edit_state()
        new_crack = self._crack_with_path(crack, st["routes"][st["index"]], None)
        self.cfg.saved_cracks[idx] = new_crack
        if crack is self.cfg.last_traced_crack:
            self.cfg.last_traced_crack = new_crack
        self._commit_crack_edit(before)
        self._reset_alt_route_state()
        print(f" [ALT ROUTE] Crack #{idx + 1} now follows the chosen route ({len(new_crack['path'])} points)"
              + (f"; its {n_tracts} width tract(s) no longer matched and were removed" if n_tracts else "")
              + ". [U] undoes, [S] saves JSON and masks.")
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _handle_alt_route_mouse(self, event, x, y):
        """A click on the photo while routes are shown closes the tool, keeping the crack's route."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self._reset_alt_route_state()
        print(" [ALT ROUTE] Closed: the crack keeps its route.")
        self.refresh_zoom_viewport()

    # --- drawing --------------------------------------------------------------------------------

    def _draw_alt_route(self, win_out):
        """The route on screen in its own color, plus a one-line prompt."""
        st = self.cfg.alt_route_state
        if not st["active"] or self._alt_route_crack_index() is None:
            return
        y = 135 if self.cfg.hud_large_size else 120
        if st["index"] == 0:
            self._put_hud_text(win_out, "ORIGINAL ROUTE (blue): button = alternatives", (15, y), ALT_ROUTE_HUD_BGR)
            return
        color, _ = alt_route_color(st["index"])
        route = st["routes"][st["index"]]
        pts = np.array([self.transform_real_to_window_coords(p[0], p[1]) for p in route], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(win_out, [pts], False, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.polylines(win_out, [pts], False, color, 3, cv2.LINE_AA)
        arrow = "<" if st["direction"] < 0 else ">"
        self._put_hud_text(win_out, f"ROUTE {st['index']}/{len(st['routes']) - 1}: $ = use it, button {arrow} = "
                           + ("back" if st["direction"] < 0 else "next"), (15, y), color)
