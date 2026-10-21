"""test_v130_edge_trace.py

1.3.0 fix (USER REPORT): a faint crack running a few pixels from a strong edge or shadow line (e.g. right under a
cornice) was abandoned for that edge by the guided trace, even through Shift+clicked points: the leg cost was scaled
on the strongest line in the corridor, and the clicked points were snapped onto the strongest line nearby.
Now every line counts as much as its neighbours (local normalisation), so the operator's points choose the line,
and a click is snapped onto the NEAREST line.

Run with:  python3 -m unittest test_v130_edge_trace -v
"""
import unittest

import cv2
import numpy as np

from test_v105_link import _two_crack_app


def _cornice_photo(gap, contrast=25, seed=0):
    """A dark cornice above y=60 with a thin shadow line under its lip, and a faint crack `gap` px below the edge
    that runs along it, then turns down-right. Returns (BGR photo, crack centre-line points)."""
    rng = np.random.default_rng(seed)
    img = np.full((260, 300), 185, np.float32)
    img[:60] = 95
    img[56:60] = 60
    pts = [(x, 60 + gap) for x in range(40, 150)] + [(150 + t, 60 + gap + int(1.2 * t)) for t in range(120)]
    for x, y in pts:
        cv2.circle(img, (x, y), 1, 185 - contrast, -1)
    img = np.clip(cv2.GaussianBlur(img, (3, 3), 0) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), pts


def _app(img):
    app = _two_crack_app()
    h, w = img.shape[:2]
    app.cfg.H_img, app.cfg.W_img, app.cfg.zoom_box, app.cfg.img_original = h, w, [0, 0, w, h], img
    app._compute_binary_and_skeleton_masks()
    return app


def _share_off_crack(route, pts, tol=2.5):
    p, r = np.asarray(pts, float), np.asarray(route, float)
    d = np.min(np.hypot(r[:, None, 0] - p[None, :, 0], r[:, None, 1] - p[None, :, 1]), axis=1)
    return float((d > tol).mean())


class TestCrackNextToAnEdge(unittest.TestCase):

    def test_route_through_clicked_points_stays_on_the_faint_crack(self):
        for gap in (4, 6, 8, 12):
            for seed in (0, 1):
                img, pts = _cornice_photo(gap, seed=seed)
                app = _app(img)
                clicks = [pts[i] for i in (0, 40, 80, 120, len(pts) - 1)]
                route = app._guided_route(clicks)
                self.assertLess(_share_off_crack(route, pts), 0.05, f"gap {gap}px, seed {seed}: route left the crack")

    def test_click_next_to_the_crack_snaps_onto_it_not_onto_the_edge(self):
        for gap in (4, 6, 8):
            img, pts = _cornice_photo(gap)
            app = _app(img)
            for x, y in (pts[0], pts[40], pts[80]):
                sx, sy = app._snap_to_dark_line((x, y - 1))   # clicked 1 px towards the edge
                self.assertLessEqual(abs(sy - y), 1, f"gap {gap}px: snapped to y={sy}, crack at y={y}")

    def test_snap_with_the_photo_zoomed_out_still_picks_the_nearest_line(self):
        img, pts = _cornice_photo(4)
        app = _app(img)
        app.cfg.zoom_box = [0, 0, 4000, 3000]   # a 4K photo seen whole: bigger snap radius
        x, y = pts[40]
        self.assertEqual(app._snap_to_dark_line((x, y - 1))[1], y)

    def test_texture_alone_stays_expensive(self):
        img, _ = _cornice_photo(30)
        app = _app(img)
        ridge = app._thin_dark_line_strength((0, 120, 300, 260))   # plaster only, below the crack's start
        band = np.ones(ridge.shape, bool)
        strength = app._guide_strength(ridge, band)
        self.assertLess(float(np.median(strength)), 0.3, "most of the wall must not look like a line")


if __name__ == "__main__":
    unittest.main()
