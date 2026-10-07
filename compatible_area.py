"""compatible_area.py
v1.0.7 Compatible area for the Building portion tool, mixed into CrackSegmentation (smart_segmentation.py).
When [5] / Building portion is turned on, every other photo of the same building group (already segmented,
with its JSON) is aligned with the current photo at reduced size, through the same perspective pre-warp as
[W]/[L]. The part of the current photo each of them shows is tinted orange, and the window is pre-set
around it, so the operator does not have to remember which part of the building the other photos share.
"""
import os

import cv2
import numpy as np


COMPATIBLE_AREA_BGR = (0, 140, 255)   # orange
COMPATIBLE_AREA_ALPHA = 0.28


class CompatibleAreaMixin:
    """Finds and draws the area of the current photo shared with the rest of its group."""

    def _group_source_images(self):
        """Photos of the current photo's building group that already have a JSON (the current one excluded)."""
        cur = self.cfg.CURRENT_IMAGE_PATH
        if not cur or "__" not in os.path.basename(cur):
            return []
        code = os.path.splitext(os.path.basename(cur).split("__")[-1])[0]
        code = code[:-4] if code.endswith("-seg") else code
        folder = self._json_folder()
        if not os.path.isdir(folder):
            return []
        cur_base = os.path.splitext(os.path.basename(cur))[0].lower()
        sources = []
        for j_file in sorted(os.listdir(folder)):
            if not j_file.lower().endswith(".json") or os.path.splitext(j_file)[0].lower() == cur_base:
                continue
            found = self._score_one_candidate_session(folder, j_file, code)
            if found is not None and os.path.abspath(found[2]) != os.path.abspath(cur):
                sources.append(found[2])
        return sources

    @staticmethod
    def _small_gray(img, max_dim):
        """Grayscale copy at most max_dim pixels on its longer side, and the scale used."""
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, max_dim / float(max(gray.shape[:2])))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray, scale

    def _footprint_in_current(self, src_path, cur_small, cur_scale):
        """Polygon (current-photo pixels) of what src_path shows of the current photo, or None if they do not align."""
        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if src is None:
            return None
        src_small, src_scale = self._small_gray(src, self.cfg.COMPATIBLE_AREA_MAX_DIM)
        found = self._find_good_sift_matches(src_small, cur_small)
        if found is None or len(found[2]) < 8:
            return None
        homography = self._compute_warp_homography(*found)
        if homography is None:
            return None
        refined = self._refine_alignment_by_prewarp(src_small, cur_small, None, homography[0])
        if refined is None and homography[3]:
            return None  # not the same building after all, or too different to say
        H_small = refined[0] if refined is not None else homography[0]
        H = np.diag([1.0 / cur_scale, 1.0 / cur_scale, 1.0]) @ H_small @ np.diag([src_scale, src_scale, 1.0])
        if not self._is_usable_prewarp_start(H, src.shape):
            return None
        h, w = src.shape[:2]
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
        frame = np.float32([[0, 0], [self.cfg.W_img, 0], [self.cfg.W_img, self.cfg.H_img], [0, self.cfg.H_img]])
        area, poly = cv2.intersectConvexConvex(corners.reshape(-1, 2), frame)
        return poly.reshape(-1, 2) if poly is not None and area > 1.0 else None

    def _find_compatible_area(self):
        """Orange polygons of the current photo shared with each other photo of its group (computed once per photo)."""
        st = self.cfg.portion_state
        st["compatible"] = []
        sources = self._group_source_images()
        if not sources or self.cfg.img_original is None:
            print(" [BUILDING PORTION] No other segmented photo in this building group: draw the window by hand.")
            return
        print(f" [BUILDING PORTION] Looking for the part shared with {len(sources)} other photo(s) of the group...")
        cur_small, cur_scale = self._small_gray(self.cfg.img_original, self.cfg.COMPATIBLE_AREA_MAX_DIM)
        for src_path in sources:
            try:
                poly = self._footprint_in_current(src_path, cur_small, cur_scale)
            except cv2.error:
                poly = None
            if poly is not None:
                st["compatible"].append(poly)
        n = len(st["compatible"])
        print(f" [BUILDING PORTION] Shared part found with {n} of {len(sources)} photo(s), tinted orange."
              if n else " [BUILDING PORTION] No shared part found automatically: draw the window by hand.")

    def _compatible_area_rect(self):
        """Bounding box (x0, y0, x1, y1) of the orange area, or None."""
        polys = self.cfg.portion_state.get("compatible") or []
        if not polys:
            return None
        pts = np.concatenate(polys)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        return (max(0, int(x0)), max(0, int(y0)), min(self.cfg.W_img - 1, int(x1)), min(self.cfg.H_img - 1, int(y1)))

    def _draw_compatible_area(self, win_out):
        """Orange glass over the shared area while the Building portion tool is on."""
        st = self.cfg.portion_state
        if not st["active"] or not st.get("compatible"):
            return
        overlay = win_out.copy()
        outlines = []
        for poly in st["compatible"]:
            pts = np.array([self.transform_real_to_window_coords(p[0], p[1]) for p in poly], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], COMPATIBLE_AREA_BGR)
            outlines.append(pts)
        cv2.addWeighted(overlay, COMPATIBLE_AREA_ALPHA, win_out, 1 - COMPATIBLE_AREA_ALPHA, 0, win_out)
        cv2.polylines(win_out, outlines, True, COMPATIBLE_AREA_BGR, 2, cv2.LINE_AA)
