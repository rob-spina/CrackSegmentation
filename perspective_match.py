"""perspective_match.py
v1.0.7 Perspective pre-warp for [W]/[L] imports, mixed into CrackSegmentation (smart_segmentation.py).
Two photos of the same building taken from different viewpoints (a close-up and a wide oblique view) share
few SIFT matches that agree, so the first alignment is rough and was rejected. Here the target photo is
re-seen, in memory only, from the source photo's viewpoint through that rough alignment; the two now look
alike, so matching them again gives many more consistent matches and a precise correction. The photo on
disk and on screen is never changed: only the alignment used to project the cracks improves.
"""
import cv2
import numpy as np


class PerspectiveMatchMixin:
    """Pre-warp refinement of the import homography. Relies on CrackSegmentation's cfg and SIFT/homography helpers."""

    def _detect_warp_sift(self, img, mask=None):
        """SIFT keypoints and descriptors with the same settings as the import matching."""
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
        return sift.detectAndCompute(img, mask)

    def _is_usable_prewarp_start(self, H, src_shape):
        """A rough alignment can seed the pre-warp only if it keeps orientation, has a sane scale
        and keeps every corner of the source photo in front of the camera (no fold over the horizon)."""
        cfg = self.cfg
        if H is None or not np.all(np.isfinite(H)):
            return False
        det = float(np.linalg.det(H[:2, :2]))
        if not (cfg.WARP_PREWARP_MIN_AREA_SCALE <= det <= cfg.WARP_PREWARP_MAX_AREA_SCALE):
            return False
        h, w = src_shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
        return bool(np.all(corners @ H[2] > 0))

    def _refine_alignment_by_prewarp(self, img1, img2, src_mask, H, src_features=None):
        """Re-sees img2 from img1's viewpoint through H, matches again and corrects H.
        Returns (H_refined, num_inliers, n_good_matches), or None when the refinement cannot be trusted."""
        cfg = self.cfg
        if not cfg.WARP_PREWARP_REFINE or not self._is_usable_prewarp_start(H, img1.shape):
            return None
        h1, w1 = img1.shape[:2]
        H_inv = np.linalg.inv(H)
        warped = cv2.warpPerspective(img2, H_inv, (w1, h1), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full(img2.shape[:2], 255, np.uint8), H_inv, (w1, h1), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))  # away from the black border of the warp
        if src_mask is not None:
            valid = cv2.bitwise_and(valid, src_mask)
        if cv2.countNonZero(valid) < 0.01 * w1 * h1:
            return None
        if src_features is not None:
            src_features = self._keypoints_inside(src_features, valid)
        result = self._find_good_sift_matches(img1, warped, valid, valid, src_features=src_features)
        if result is None or len(result[2]) < 8:
            return None
        kp1, kp2, good = result
        homography = self._compute_warp_homography(kp1, kp2, good)
        if homography is None or homography[3]:
            return None
        H_fix, num_inliers = homography[0], homography[1]
        if not self._is_small_correction(H_fix, w1, h1):
            return None
        H_refined = H @ H_fix
        H_refined = H_refined / H_refined[2, 2]
        if not self._is_usable_prewarp_start(H_refined, img1.shape):
            return None
        print(f"[WARP PREWARP] Alignment refined from the source viewpoint: {num_inliers}/{len(good)} matches.")
        return H_refined, num_inliers, len(good)

    @staticmethod
    def _keypoints_inside(features, mask):
        """The (keypoints, descriptors) whose position lies inside mask."""
        kps, des = features
        if des is None:
            return features
        h, w = mask.shape[:2]
        keep = [i for i, k in enumerate(kps)
                if 0 <= int(k.pt[1]) < h and 0 <= int(k.pt[0]) < w and mask[int(k.pt[1]), int(k.pt[0])]]
        return [kps[i] for i in keep], des[keep] if keep else None

    def _is_small_correction(self, H_fix, w, h):
        """The correction found after the pre-warp must be small: a big one means the rough alignment was wrong."""
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(corners, H_fix)
        shift = float(np.max(np.linalg.norm(moved - corners, axis=2)))
        return shift <= self.cfg.WARP_PREWARP_MAX_CORRECTION * float(np.hypot(w, h))

    @staticmethod
    def _local_linear_scale(H, pt):
        """How much H enlarges lengths around pt (sqrt of the local area scale), for the span plausibility check."""
        x, y = float(pt[0]), float(pt[1])
        pts = np.float32([[x, y], [x + 1, y], [x, y + 1]]).reshape(-1, 1, 2)
        a, b, c = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.float64)
        u, v = b - a, c - a
        return abs(u[0] * v[1] - u[1] * v[0]) ** 0.5

    @staticmethod
    def _fraction_inside(points, H, w2, h2):
        """Share of a shape's points that H puts inside the w2 x h2 target photo (before any clamping)."""
        if not points:
            return 0.0
        warped = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = (warped[:, 0] >= 0) & (warped[:, 0] < w2) & (warped[:, 1] >= 0) & (warped[:, 1] < h2)
        return float(np.mean(inside))

    @staticmethod
    def _longest_inside_run(points, H, w2, h2):
        """(first, last) indices of the longest run of consecutive points that H puts inside the target photo,
        so a crack leaving the view is cut where it leaves, not squashed onto the border. None if no point is inside."""
        warped = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = (warped[:, 0] >= 0) & (warped[:, 0] < w2) & (warped[:, 1] >= 0) & (warped[:, 1] < h2)
        best, start = None, None
        for i, ok in enumerate(list(inside) + [False]):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if best is None or i - start > best[1] - best[0] + 1:
                    best = (start, i - 1)
                start = None
        return best
