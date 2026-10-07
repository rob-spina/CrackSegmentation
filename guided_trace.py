"""guided_trace.py
v1.0.7 Guided tracing through Shift+clicked points, mixed into CrackSegmentation (smart_segmentation.py).
A plain two-click trace keeps using A* on the edge skeleton. When the operator Shift+clicks intermediate
points, every leg is instead routed on the photo itself: thin dark lines (cracks) are cheap, broad light/shadow
boundaries are not, the route is kept near the line joining the clicked points, and a leg may not run back
along the previous one -- so the crack passes THROUGH each point instead of reaching it and turning back.
"""
import cv2
import numpy as np
from skimage.graph import route_through_array


class GuidedTraceMixin:
    """Guided legs between clicked points. Relies on CrackSegmentation's cfg and _stitch."""

    def _leg_roi(self, p, q):
        """(x0, y0, x1, y1) around two points, padded like the A* search window and clipped to the photo."""
        pad = self.cfg.TRACE_GUIDE_PAD_PX
        h, w = self.cfg.img_original.shape[:2]
        x0, y0 = max(0, min(p[0], q[0]) - pad), max(0, min(p[1], q[1]) - pad)
        x1, y1 = min(w, max(p[0], q[0]) + pad + 1), min(h, max(p[1], q[1]) + pad + 1)
        return int(x0), int(y0), int(x1), int(y1)

    def _thin_dark_line_strength(self, roi):
        """Per-pixel strength of thin dark LINES at crack scale: the larger Hessian eigenvalue, scale-normalised,
        best over TRACE_GUIDE_RIDGE_SIGMAS (same response as skimage's Sato filter, ~20x faster with OpenCV).
        A faint crack still scores clearly above the plaster texture; a broad light/shadow boundary does not."""
        x0, y0, x1, y1 = roi
        gray = cv2.cvtColor(self.cfg.img_original[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out = np.zeros_like(gray)
        for sigma in self.cfg.TRACE_GUIDE_RIDGE_SIGMAS:
            g = cv2.GaussianBlur(gray, (0, 0), sigma)
            dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
            lam = (dxx + dyy) * 0.5 + np.sqrt(((dxx - dyy) * 0.5) ** 2 + dxy ** 2)
            np.maximum(out, np.maximum(lam, 0.0) * (sigma * sigma), out=out)
        return out

    @staticmethod
    def _distance_to_segment(shape, p, q, x0, y0):
        """Per-pixel distance (px) to the segment p-q, for a crop of the given shape at (x0, y0); plus its length."""
        # Broadcast row/column vectors in float32: long legs give big crops.
        X = (np.arange(shape[1], dtype=np.float32) + (x0 - p[0]))[None, :]
        Y = (np.arange(shape[0], dtype=np.float32) + (y0 - p[1]))[:, None]
        vx, vy = float(q[0] - p[0]), float(q[1] - p[1])
        L2 = max(1.0, vx * vx + vy * vy)
        t = np.clip((X * vx + Y * vy) / L2, 0.0, 1.0)
        return np.hypot(X - t * vx, Y - t * vy), L2 ** 0.5

    def _snap_to_dark_line(self, p):
        """The clicked point moved onto the darkest line nearby (clicks are rarely exact): within TRACE_GUIDE_SNAP_PX,
        or 3 screen pixels when zoomed out, since one screen pixel then covers several photo pixels."""
        view_w = max(1, self.cfg.zoom_box[2] - self.cfg.zoom_box[0])
        r = max(int(self.cfg.TRACE_GUIDE_SNAP_PX), int(round(3 * view_w / 1200.0)))
        h, w = self.cfg.img_original.shape[:2]
        x, y = int(min(max(p[0], 0), w - 1)), int(min(max(p[1], 0), h - 1))
        roi = (max(0, x - r), max(0, y - r), min(w, x + r + 1), min(h, y + r + 1))
        # Filtered with some context around the window (no border effects), then scaled 0..1 inside it.
        m = 8
        ctx = (max(0, roi[0] - m), max(0, roi[1] - m), min(w, roi[2] + m), min(h, roi[3] + m))
        ridge = self._thin_dark_line_strength(ctx)
        ridge = ridge[roi[1] - ctx[1]:roi[3] - ctx[1], roi[0] - ctx[0]:roi[2] - ctx[0]]
        strength = ridge / max(1e-9, float(ridge.max()))
        cy, cx = y - roi[1], x - roi[0]
        if strength[cy, cx] >= 0.8:
            return x, y  # already on the line: keep the click exactly
        yy, xx = np.mgrid[0:strength.shape[0], 0:strength.shape[1]]
        score = strength - 0.1 * np.hypot(yy - cy, xx - cx) / max(1, r)  # nearer wins a near tie
        iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
        return roi[0] + int(ix), roi[1] + int(iy)

    def _guided_leg(self, p, q, prev_leg=None):
        """Cheapest 8-connected route p -> q (inclusive) on the guided cost; [] if it cannot be computed."""
        cfg = self.cfg
        x0, y0, x1, y1 = roi = self._leg_roi(p, q)
        dist, length = self._distance_to_segment((y1 - y0, x1 - x0), p, q, x0, y0)
        corridor = max(float(cfg.TRACE_GUIDE_CORRIDOR_MIN_PX), cfg.TRACE_GUIDE_CORRIDOR_FRAC * length)
        ridge = self._thin_dark_line_strength(roi)
        # Scaled on the lines found near the guide only: a strong crack farther away cannot make a faint one look weak.
        band = ridge[dist < corridor]
        scale = max(1e-9, float(np.percentile(band, 97))) if band.size else max(1e-9, float(ridge.max()))
        strength = np.clip(ridge / scale, 0.0, 1.0)
        cost = (1.0 + cfg.TRACE_GUIDE_RIDGE_WEIGHT * (1.0 - strength) ** 2) * (1.0 + (dist / corridor) ** 4)
        if prev_leg:
            # No running back along the previous leg: only the clicked point itself stays free.
            back = np.zeros(ridge.shape, dtype=np.uint8)
            pts = (np.asarray(prev_leg, dtype=np.int32) - [x0, y0]).reshape(-1, 1, 2)
            cv2.polylines(back, [pts], False, 255, 9)
            cv2.circle(back, (int(p[0]) - x0, int(p[1]) - y0), 8, 0, -1)
            cost[back > 0] *= cfg.TRACE_GUIDE_BACKTRACK_FACTOR
        try:
            path, _ = route_through_array(cost, (int(p[1]) - y0, int(p[0]) - x0), (int(q[1]) - y0, int(q[0]) - x0),
                                          fully_connected=True, geometric=True)
        except (ValueError, IndexError) as e:
            print(f"[GUIDED TRACE] Leg not computed: {e}")
            return []
        return [(int(c) + x0, int(r) + y0) for r, c in path]

    def _guided_route(self, points):
        """One continuous route through all the clicked points, in order; [] if any leg fails."""
        snapped = [self._snap_to_dark_line(p) for p in points]
        route, prev_leg = [], None
        for p, q in zip(snapped, snapped[1:]):
            leg = self._guided_leg(p, q, prev_leg)
            if not leg:
                return []
            route = self._stitch(route, leg)
            prev_leg = leg
        return route
