#!/usr/bin/env python3
"""compare_and_filter_cracks.py (object-oriented port): standalone CLI comparing each crack's shape (sinuosity) across two or more LabelMe JSON exports of the same building group, removing any that disagree too much.
Usage: python3 compare_and_filter_cracks.py FILE1.json FILE2.json [...] [--dry-run]
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # overlay regeneration is skipped gracefully if cv2 is missing


class LoadedJson:
    """Unchanged from the original script -- was already its own class."""

    def __init__(self, path):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.shapes = self.data.get("shapes", [])
        self.scale = float(self.data.get("pixel_to_cm_scale", 0.05))
        self.image_height = self.data.get("imageHeight")
        self.image_width = self.data.get("imageWidth")
        self.image_path = self.data.get("imagePath", "")
        # indices, within self.shapes, of the crack-type shapes, in order
        self.crack_positions = [i for i, s in enumerate(self.shapes) if CrackComparator.is_crack_shape(s)]

    def crack_points(self, crack_index):
        shape_idx = self.crack_positions[crack_index]
        return [tuple(pt) for pt in self.shapes[shape_idx]["points"]]

    @property
    def n_cracks(self):
        return len(self.crack_positions)


class CrackComparator:
    """Groups every comparison/mask-rebuild helper for cross-photo crack filtering (used by both the standalone script and fast_segmentation.py's [G] key).
    """

    def __init__(self, cfg=None):
        """cfg: optional shared Config instance (see config.py); when None, falls back to hardcoded defaults so this module still works fully standalone.
        """
        self._cfg = cfg
        self.mask_dilation_px = getattr(cfg, "CRACK_MASK_DILATION_PX", 0) if cfg is not None else 0
        self.fill_close_px = getattr(cfg, "WIDTH_EDIT_FILL_CLOSE_PX", 5) if cfg is not None else 5
        self.auto_edge_max_offset_px = getattr(cfg, "CRACK_AUTO_EDGE_MAX_OFFSET_PX", 2) if cfg is not None else 2
        self.auto_edge_enabled = getattr(cfg, "CRACK_MASK_AUTO_EDGE_ENABLED", True) if cfg is not None else True

        # Resolved immediately, before any method needing the result can run.
        self.resolve_crack_edge_width_from_gsd()

    # GSD calibration

    def resolve_crack_edge_width_from_gsd(self):
        """Derives mask_dilation_px/auto_edge_max_offset_px from IMAGE_GSD_MM_PER_PX so the crack band stays a consistent physical width. No-op if no Config was given or GSD is unset.
        """
        if self._cfg is None:
            return None
        gsd = getattr(self._cfg, "IMAGE_GSD_MM_PER_PX", 0.0) or 0.0
        if gsd <= 0:
            return None

        target_mm = getattr(self._cfg, "CRACK_TARGET_PHYSICAL_WIDTH_MM", 3.0)
        ceiling_px = getattr(self._cfg, "CRACK_AUTO_EDGE_GSD_SAFETY_CEILING_PX", 20)

        target_width_px = target_mm / gsd
        offset_px = max(0, round((target_width_px - 1) / 2.0))
        offset_px = min(offset_px, ceiling_px)

        self.mask_dilation_px = offset_px
        self.auto_edge_max_offset_px = offset_px

        print(f"[GSD CALIBRATION] IMAGE_GSD_MM_PER_PX={gsd:.4f} mm/px, "
              f"CRACK_TARGET_PHYSICAL_WIDTH_MM={target_mm:.2f}mm -> "
              f"automatic crack band set to {offset_px}px per side "
              f"(~{2 * offset_px + 1}px total).")
        return offset_px

    # Shape helpers

    @staticmethod
    def is_crack_shape(shape):
        label = str(shape.get("label", "")).lower()
        shape_type = shape.get("shape_type", "")
        return label.startswith("crack") or label.startswith("crepa") or shape_type in ("linestrip", "linestring")

    @staticmethod
    def path_length(points):
        total = 0.0
        for i in range(1, len(points)):
            total += math.dist(points[i - 1], points[i])
        return total

    @staticmethod
    def sinuosity(points):
        if len(points) < 2:
            return float("nan")
        chord = math.dist(points[0], points[-1])
        if chord <= 0:
            return float("nan")
        return CrackComparator.path_length(points) / chord

    # Comparison

    @staticmethod
    def relative_diff(a, b):
        if a != a or b != b:  # NaN check
            return float("nan")
        denom = max(abs(a), abs(b))
        if denom == 0:
            return 0.0
        return abs(a - b) / denom

    @staticmethod
    def _compute_crack_metrics_across_files(loaded_jsons, idx):
        """Length (cm) and sinuosity of crack #idx, one entry per
        file."""
        metrics = []
        for lj in loaded_jsons:
            pts = lj.crack_points(idx)
            L_px = CrackComparator.path_length(pts)
            L_cm = L_px * lj.scale
            s = CrackComparator.sinuosity(pts)
            metrics.append({"file": os.path.basename(lj.path), "length_cm": L_cm, "sinuosity": s})
        return metrics

    @staticmethod
    def _compute_max_pairwise_diffs(metrics):
        """Largest relative sinuosity/length difference across every
        pair of files for one crack."""
        max_sin_diff = 0.0
        max_len_diff = 0.0
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                max_sin_diff = max(max_sin_diff, CrackComparator.relative_diff(metrics[i]["sinuosity"], metrics[j]["sinuosity"]))
                max_len_diff = max(max_len_diff, CrackComparator.relative_diff(metrics[i]["length_cm"], metrics[j]["length_cm"]))
        return max_sin_diff, max_len_diff

    @staticmethod
    def _build_comparison_row(idx, metrics, max_sin_diff, max_len_diff, sinuosity_threshold, length_threshold):
        """Builds one crack's report row. Returns (row,
        is_incompatible)."""
        bad_sin = max_sin_diff > sinuosity_threshold
        bad_len = (length_threshold is not None) and (max_len_diff > length_threshold)
        verdict = "DISCARDED" if (bad_sin or bad_len) else "ok"
        row = {
            "crack_index": idx,
            "metrics": metrics,
            "max_sinuosity_diff_pct": max_sin_diff * 100,
            "max_length_diff_pct": max_len_diff * 100,
            "verdict": verdict,
            "reason": ("sinuosity" if bad_sin else "") + ("+length" if bad_sin and bad_len else ("length" if bad_len else "")),
        }
        return row, (bad_sin or bad_len)

    @staticmethod
    def _compute_crack_count_note(loaded_jsons, n_common):
        """Note explaining a mismatched crack count between files, if
        any."""
        counts = [lj.n_cracks for lj in loaded_jsons]
        if len(set(counts)) > 1:
            return (
                f"Crack count differs between files ({counts}) -- only the first "
                f"{n_common} (in order) present in every file were compared; the extra cracks in the "
                f"longer files were left unchanged (no counterpart to compare them against)."
            )
        return None

    @staticmethod
    def compare(loaded_jsons, sinuosity_threshold, length_threshold):
        """Returns (rows, incompatible_crack_indices, extra_note): one report dict per crack index, and the set of crack positions to remove from every file.
        """
        n_common = min(lj.n_cracks for lj in loaded_jsons)
        rows = []
        incompatible = set()

        for idx in range(n_common):
            metrics = CrackComparator._compute_crack_metrics_across_files(loaded_jsons, idx)
            max_sin_diff, max_len_diff = CrackComparator._compute_max_pairwise_diffs(metrics)
            row, is_incompatible = CrackComparator._build_comparison_row(idx, metrics, max_sin_diff, max_len_diff, sinuosity_threshold, length_threshold)
            if is_incompatible:
                incompatible.add(idx)
            rows.append(row)

        extra_note = CrackComparator._compute_crack_count_note(loaded_jsons, n_common)
        return rows, incompatible, extra_note

    @staticmethod
    def print_report(rows, sinuosity_threshold, length_threshold, extra_note):
        print()
        print("=== Crack compatibility comparison across photos ===")
        if extra_note:
            print(f"[NOTE] {extra_note}")
        print(f"Thresholds: sinuosity > {sinuosity_threshold*100:.0f}%", end="")
        if length_threshold is not None:
            print(f", length > {length_threshold*100:.0f}%")
        else:
            print(" (length check disabled)")
        print()
        for row in rows:
            print(f"crack #{row['crack_index']}: max sinuosity diff={row['max_sinuosity_diff_pct']:.1f}%  "
                  f"max length diff={row['max_length_diff_pct']:.1f}%  -> {row['verdict']}"
                  + (f" ({row['reason']})" if row["verdict"] == "DISCARDED" else ""))
            for m in row["metrics"]:
                print(f"     {m['file']:<40} length={m['length_cm']:.1f}cm  sinuosity={m['sinuosity']:.3f}")
        print()

    @staticmethod
    def write_csv_report(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["crack_index", "file", "length_cm", "sinuosity", "max_sinuosity_diff_pct", "max_length_diff_pct", "verdict", "reason"])
            for row in rows:
                for m in row["metrics"]:
                    writer.writerow([row["crack_index"], m["file"], f"{m['length_cm']:.2f}", f"{m['sinuosity']:.4f}",
                                      f"{row['max_sinuosity_diff_pct']:.1f}", f"{row['max_length_diff_pct']:.1f}",
                                      row["verdict"], row["reason"]])
        print(f"[REPORT] Full table saved to: {path}")

    # Removal / regeneration

    @staticmethod
    def stamp_points(mask, list_of_point_lists, width, height):
        xs_parts, ys_parts = [], []
        for pts in list_of_point_lists:
            if not pts:
                continue
            arr = np.asarray(pts, dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            xs_parts.append(arr[:, 0])
            ys_parts.append(arr[:, 1])
        if not xs_parts:
            return
        all_x = np.concatenate(xs_parts)
        all_y = np.concatenate(ys_parts)
        valid = (all_x >= 0) & (all_x < width) & (all_y >= 0) & (all_y < height)
        mask[all_y[valid], all_x[valid]] = 255

    @staticmethod
    def rle_rows_to_mask(rle_rows, height, width):
        """Rebuilds a mask from stored [y, x0, x1] row-runs, defensively skipping any malformed row."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for row in (rle_rows or []):
            try:
                y, x0, x1 = int(row[0]), int(row[1]), int(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            if y < 0 or y >= height:
                continue
            x0, x1 = max(0, x0), min(width - 1, x1)
            if x0 <= x1:
                mask[y, x0:x1 + 1] = 255
        return mask

    @staticmethod
    def _compute_local_perpendicular(points, i, i0, i1):
        """Local tangent/perpendicular direction at one path point. None if degenerate."""
        p_prev = points[i - 1] if i - 1 >= i0 else points[i]
        p_next = points[i + 1] if i + 1 <= i1 else points[i]
        dx, dy = (p_next[0] - p_prev[0]), (p_next[1] - p_prev[1])
        tangent_norm = (dx * dx + dy * dy) ** 0.5
        if tangent_norm < 1e-6:
            return None
        return -dy / tangent_norm, dx / tangent_norm

    def _measure_current_extent(self, mask, px, py, side_sign, perp, scan_cap, width, height):
        """Farthest already-filled pixel on the ray (not the first
        contiguous run) -- robust to a stray one-pixel gap."""
        perp_x, perp_y = perp
        cur_extent = 0
        for step in range(1, scan_cap + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            if mask[sy, sx] != 0:
                cur_extent = step
        return cur_extent

    def _grow_side_at_point(self, mask, px, py, side_sign, perp, new_extent, width, height):
        perp_x, perp_y = perp
        for step in range(1, new_extent + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            mask[sy, sx] = 255

    def _shrink_side_at_point(self, mask, px, py, side_sign, perp, cur_extent, new_extent, width, height):
        perp_x, perp_y = perp
        for step in range(new_extent + 1, cur_extent + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if 0 <= sx < width and 0 <= sy < height:
                mask[sy, sx] = 0

    def _apply_directional_margin_at_point(self, mask, points, i, i0, i1, side_sign, delta, scan_cap, width, height):
        """Grows or shrinks one side of the crack at a single path
        point."""
        perp = self._compute_local_perpendicular(points, i, i0, i1)
        if perp is None:
            return
        px, py = points[i]
        cur_extent = self._measure_current_extent(mask, px, py, side_sign, perp, scan_cap, width, height)
        new_extent = max(0, cur_extent + delta)
        if new_extent > cur_extent:
            self._grow_side_at_point(mask, px, py, side_sign, perp, new_extent, width, height)
        elif new_extent < cur_extent:
            self._shrink_side_at_point(mask, px, py, side_sign, perp, cur_extent, new_extent, width, height)

    def apply_directional_margin(self, mask, points, i0, i1, side_sign, delta, height, width):
        """Grows or shrinks only ONE side of a tract mask, following the crack's local perpendicular direction per point. Modifies `mask` in place."""
        if delta == 0 or not points:
            return mask
        n = len(points)
        if i0 < 0 or i1 >= n or i0 > i1:
            return mask
        scan_cap = 400  # generous safety bound, see the app's own scan_cap

        for i in range(i0, i1 + 1):
            self._apply_directional_margin_at_point(mask, points, i, i0, i1, side_sign, delta, scan_cap, width, height)
        return mask

    def _extract_segment_points_and_validity(self, points, i0, i1, height, width):
        """The segment's path points and which of them fall within
        image bounds."""
        seg_pts = np.asarray(points[i0:i1 + 1], dtype=np.int64) if points else np.empty((0, 2), dtype=np.int64)
        valid = np.zeros(0, dtype=bool)
        if seg_pts.ndim == 2 and seg_pts.shape[1] >= 2 and seg_pts.shape[0] > 0:
            valid = (seg_pts[:, 0] >= 0) & (seg_pts[:, 0] < width) & (seg_pts[:, 1] >= 0) & (seg_pts[:, 1] < height)
        return seg_pts, valid

    def _get_segment_margins(self, seg):
        """The segment's independent left/right directional margins,
        defaulting to 0 on any malformed value."""
        try:
            margin_left = int(seg.get("margin_left_px", 0) or 0)
        except (TypeError, ValueError):
            margin_left = 0
        try:
            margin_right = int(seg.get("margin_right_px", 0) or 0)
        except (TypeError, ValueError):
            margin_right = 0
        return margin_left, margin_right

    def _render_fill_segment_mask(self, seg, seg_pts, valid, height, width):
        """Renders an auto-detected-fill segment: the stored RLE shape,
        refined by a signed dilate/erode margin."""
        seg_mask = self.rle_rows_to_mask(seg["fill"], height, width)
        try:
            margin = int(seg.get("width_px", 0))
        except (TypeError, ValueError):
            margin = 0
        if margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            seg_mask = cv2.dilate(seg_mask, k)
        elif margin < 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(margin) + 1, 2 * abs(margin) + 1))
            seg_mask = cv2.erode(seg_mask, k)
            if not np.any(seg_mask) and np.any(valid):
                seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
        return seg_mask

    def _render_legacy_segment_mask(self, seg, seg_pts, valid, height, width):
        """Renders a legacy plain-scalar segment: dilates straight from
        the centerline points."""
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        if np.any(valid):
            seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
            try:
                w = max(0, int(seg.get("width_px", 0)))
            except (TypeError, ValueError):
                w = 0
            if w > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
                seg_mask = cv2.dilate(seg_mask, k)
        return seg_mask

    def _apply_directional_margins_to_segment(self, seg_mask, points, i0, i1, margin_left, margin_right, seg_pts, valid, height, width):
        """Applies the independent per-side margins on top of the base
        mask, never letting a heavy shrink erase the tract outright."""
        if margin_left:
            seg_mask = self.apply_directional_margin(seg_mask, points, i0, i1, side_sign=+1,
                                                       delta=margin_left, height=height, width=width)
        if margin_right:
            seg_mask = self.apply_directional_margin(seg_mask, points, i0, i1, side_sign=-1,
                                                       delta=margin_right, height=height, width=width)
        if (margin_left or margin_right) and not np.any(seg_mask) and np.any(valid):
            seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
        return seg_mask

    def _close_gaps_in_segment_mask(self, seg_mask):
        """Morphological closing to merge the "comb of rods" artifact
        from per-point ray gaps into one solid band."""
        if np.any(seg_mask):
            close_px = max(1, int(self.fill_close_px))
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, close_k)
        return seg_mask

    def render_width_segment_mask(self, points, i0, i1, seg, height, width):
        """Builds the mask for one width_segments tract: an auto-filled shape or legacy centerline dilation, refined by independent per-side margins.
        """
        seg_pts, valid = self._extract_segment_points_and_validity(points, i0, i1, height, width)
        margin_left, margin_right = self._get_segment_margins(seg)

        if seg.get("fill"):
            seg_mask = self._render_fill_segment_mask(seg, seg_pts, valid, height, width)
        else:
            seg_mask = self._render_legacy_segment_mask(seg, seg_pts, valid, height, width)
            if not (margin_left or margin_right):
                # No directional refinement ever recorded for this legacy
                # tract -- return EXACTLY as before this revision.
                return seg_mask

        seg_mask = self._apply_directional_margins_to_segment(seg_mask, points, i0, i1, margin_left, margin_right, seg_pts, valid, height, width)

        # Same "bastoncini" homogenization as fast_segmentation.py's own
        # _render_width_segment_mask -- see WIDTH_EDIT_FILL_CLOSE_PX.
        return self._close_gaps_in_segment_mask(seg_mask)

    @staticmethod
    def _scan_one_side_extent(px, py, sign, perp, max_offset, binary_mask, width, height):
        """Counts contiguous foreground pixels along one perpendicular
        direction, capped at max_offset steps."""
        perp_x, perp_y = perp
        extent = 0
        for step in range(1, max_offset + 1):
            sx = int(round(px + sign * perp_x * step))
            sy = int(round(py + sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            if binary_mask[sy, sx] == 0:
                break
            extent = step
        return extent

    @staticmethod
    def _detect_edge_offsets_at_point(points, i, n, binary_mask, height, width, max_offset, left_offsets, right_offsets):
        """Fills in left_offsets[i]/right_offsets[i] for one path
        point."""
        perp = CrackComparator._compute_local_perpendicular(points, i, 0, n - 1)
        if perp is None:
            return
        px, py = points[i]
        left_offsets[i] = CrackComparator._scan_one_side_extent(px, py, 1, perp, max_offset, binary_mask, width, height)
        right_offsets[i] = CrackComparator._scan_one_side_extent(px, py, -1, perp, max_offset, binary_mask, width, height)

    @staticmethod
    def auto_detect_crack_edge_offsets(points, binary_mask, height, width, max_offset):
        """Detects both edges of a crack against binary_mask (computed by the caller from the original photo, or None for all-zero offsets)."""
        n = len(points) if points else 0
        left_offsets = [0] * n
        right_offsets = [0] * n
        if n == 0 or binary_mask is None:
            return left_offsets, right_offsets

        max_offset = max(0, int(max_offset))
        for i in range(n):
            CrackComparator._detect_edge_offsets_at_point(points, i, n, binary_mask, height, width, max_offset, left_offsets, right_offsets)
        return left_offsets, right_offsets

    @staticmethod
    def _edge_point_at_offset(points, i, n, sign, offset):
        """One edge point: the path point itself if offset is 0,
        otherwise shifted along the local perpendicular by `offset`."""
        px, py = points[i]
        if offset == 0:
            return (px, py)
        perp = CrackComparator._compute_local_perpendicular(points, i, 0, n - 1)
        if perp is None:
            return (px, py)
        perp_x, perp_y = perp
        return (px + sign * perp_x * offset, py + sign * perp_y * offset)

    @staticmethod
    def _fill_one_ribbon_segment(canvas, points, n, left_offsets, right_offsets, i):
        """Fills the small quad connecting path points i and i+1 along
        their two detected edges."""
        l0 = CrackComparator._edge_point_at_offset(points, i, n, 1, left_offsets[i])
        l1 = CrackComparator._edge_point_at_offset(points, i + 1, n, 1, left_offsets[i + 1])
        r0 = CrackComparator._edge_point_at_offset(points, i, n, -1, right_offsets[i])
        r1 = CrackComparator._edge_point_at_offset(points, i + 1, n, -1, right_offsets[i + 1])
        quad = np.array([l0, l1, r1, r0], dtype=np.int32)
        cv2.fillPoly(canvas, [quad], 255)

    @staticmethod
    def _build_single_point_ribbon(points, height, width):
        """Special case: a single-point "crack" just marks that one
        pixel."""
        canvas = np.zeros((height, width), dtype=np.uint8)
        px, py = points[0]
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < width and 0 <= iy < height:
            canvas[iy, ix] = 255
        return canvas

    @staticmethod
    def build_ribbon_mask_from_edge_offsets(points, left_offsets, right_offsets, height, width):
        """Fills a mask circumscribing the crack between its two detected edges, one small quad per point pair so sharp curves can't self-intersect."""
        n = len(points) if points else 0
        if n == 0:
            return np.zeros((height, width), dtype=np.uint8)
        if n == 1:
            return CrackComparator._build_single_point_ribbon(points, height, width)

        canvas = np.zeros((height, width), dtype=np.uint8)
        for i in range(n - 1):
            CrackComparator._fill_one_ribbon_segment(canvas, points, n, left_offsets, right_offsets, i)
        return canvas

    def compute_auto_edge_crack_mask(self, points, height, width, floor_px, binary_mask, max_offset=None):
        """Detects both edges and renders the resulting ribbon. binary_mask is required here; pass None to get an all-zero ribbon back.
        """
        if max_offset is None:
            max_offset = self.auto_edge_max_offset_px
        if binary_mask is None:
            return np.zeros((height, width), dtype=np.uint8)
        left_off, right_off = self.auto_detect_crack_edge_offsets(points, binary_mask, height, width, max_offset)
        floor_px = max(0, int(floor_px))
        if floor_px:
            left_off = [max(v, floor_px) for v in left_off]
            right_off = [max(v, floor_px) for v in right_off]
        ribbon = self.build_ribbon_mask_from_edge_offsets(points, left_off, right_off, height, width)
        if np.any(ribbon):
            close_px = max(1, int(self.fill_close_px))
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
            ribbon = cv2.morphologyEx(ribbon, cv2.MORPH_CLOSE, close_k)
        return ribbon

    def _compute_widths_and_special_segments(self, shape, n, default_dilation_px):
        """Builds the per-point width array and the list of "special"
        segments (auto-filled/margin-refined) needing their own renderer."""
        widths = [max(0, int(default_dilation_px))] * n
        special_segments = []
        for seg in (shape.get("width_segments") or []):
            try:
                i0, i1 = int(seg.get("i0", 0)), int(seg.get("i1", 0))
            except (TypeError, ValueError):
                continue
            if i0 > i1:
                i0, i1 = i1, i0
            i0 = max(0, min(i0, n - 1))
            i1 = max(0, min(i1, n - 1))
            try:
                has_margin = bool(int(seg.get("margin_left_px", 0) or 0) or int(seg.get("margin_right_px", 0) or 0))
            except (TypeError, ValueError):
                has_margin = False
            if seg.get("fill") or has_margin:
                special_segments.append((i0, i1, seg))
                for i in range(i0, i1 + 1):
                    widths[i] = None
            else:
                try:
                    w = max(0, int(seg.get("width_px", 0)))
                except (TypeError, ValueError):
                    continue
                for i in range(i0, i1 + 1):
                    widths[i] = w
        return widths, special_segments

    def _compute_auto_edge_ribbon(self, points, height, width, default_dilation_px, binary_mask):
        """Tries the automatic two-edge crack outline; None if disabled,
        no binary_mask given, or detection fails."""
        if not (self.auto_edge_enabled and binary_mask is not None):
            return None
        try:
            return self.compute_auto_edge_crack_mask(points, height, width, default_dilation_px, binary_mask)
        except Exception as e:
            print(f"  [MASK WARNING] Automatic edge detection failed ({e}); using uniform dilation only.")
            return None

    def _dilate_scalar_width_runs(self, out, points, widths, n, height, width):
        """Legacy flat-dilation fallback: groups the path into
        consecutive same-width runs and dilates/ORs each one in."""
        i = 0
        while i < n:
            w = widths[i]
            if w is None:
                i += 1
                continue
            j = i
            while j + 1 < n and widths[j + 1] == w:
                j += 1
            run_pts = np.asarray(points[i:j + 1], dtype=np.int64)
            if run_pts.ndim == 2 and run_pts.shape[1] >= 2:
                valid = (run_pts[:, 0] >= 0) & (run_pts[:, 0] < width) & (run_pts[:, 1] >= 0) & (run_pts[:, 1] < height)
                if np.any(valid):
                    run_mask = np.zeros((height, width), dtype=np.uint8)
                    run_mask[run_pts[valid, 1], run_pts[valid, 0]] = 255
                    if w > 0:
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
                        run_mask = cv2.dilate(run_mask, k)
                    out = cv2.bitwise_or(out, run_mask)
            i = j + 1
        return out

    def _apply_special_segments(self, out, special_segments, points, height, width):
        """ORs in every special (auto-filled/margin-refined) segment's
        own rendered mask."""
        for seg_i0, seg_i1, seg in special_segments:
            out = cv2.bitwise_or(out, self.render_width_segment_mask(points, seg_i0, seg_i1, seg, height, width))
        return out

    def _build_one_crack_mask_contribution(self, out, shape, height, width, default_dilation_px, binary_mask):
        """Adds one crack shape's contribution (base dilation +
        special segments) to the running export mask."""
        points = shape.get("points") or []
        n = len(points)
        if n == 0:
            return out
        widths, special_segments = self._compute_widths_and_special_segments(shape, n, default_dilation_px)

        auto_ribbon = self._compute_auto_edge_ribbon(points, height, width, default_dilation_px, binary_mask)
        if auto_ribbon is not None and np.any(auto_ribbon):
            out = cv2.bitwise_or(out, auto_ribbon)
        else:
            out = self._dilate_scalar_width_runs(out, points, widths, n, height, width)

        return self._apply_special_segments(out, special_segments, points, height, width)

    def build_variable_width_crack_mask(self, crack_shapes, width, height, default_dilation_px, binary_mask=None):
        """Builds the export crack mask: each shape's path is dilated by default_dilation_px, auto-edge-detected against binary_mask, or overridden per-tract via width_segments.
        """
        out = np.zeros((height, width), dtype=np.uint8)
        for shape in crack_shapes:
            out = self._build_one_crack_mask_contribution(out, shape, height, width, default_dilation_px, binary_mask)
        return out

    @staticmethod
    def compute_binary_mask_from_image(img_original):
        """Reproduces the same adaptive-threshold crack-candidate mask the live app builds. Returns None on any error (caller falls back to flat dilation).
        """
        if cv2 is None or img_original is None:
            return None
        try:
            gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        except Exception:
            return None

    @staticmethod
    def find_source_image(lj, extra_dirs):
        """Best-effort search for the original photo referenced by lj.image_path. Returns a path or None."""
        if not lj.image_path:
            return None
        base_name = os.path.basename(lj.image_path)
        json_dir = os.path.dirname(os.path.abspath(lj.path))
        candidates_dirs = [json_dir]
        parent = os.path.dirname(json_dir)
        candidates_dirs.append(os.path.join(parent, "already processed images"))
        candidates_dirs.append(os.path.join(parent, "segmentated images"))
        candidates_dirs.append(os.path.join(json_dir, "..", "already processed images"))
        candidates_dirs.append(os.path.join(json_dir, "..", "segmentated images"))
        candidates_dirs.extend(extra_dirs or [])
        for d in candidates_dirs:
            candidate = os.path.join(d, base_name)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _backup_json_if_requested(self, lj, make_backup):
        """Saves a .bak copy of the JSON before it's overwritten, if
        one doesn't already exist."""
        if not make_backup:
            return
        backup_path = lj.path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(lj.path, backup_path)
            print(f"  [BACKUP] {backup_path}")

    def _write_filtered_json(self, lj, new_shapes, n_removed_here):
        """Overwrites the JSON with the incompatible shapes removed."""
        lj.data["shapes"] = new_shapes
        with open(lj.path, "w", encoding="utf-8") as f:
            json.dump(lj.data, f, ensure_ascii=False, indent=2)
        print(f"  [JSON] {lj.path}: removed {n_removed_here} shape(s), saved.")

    def _load_source_photo(self, lj, images_dir):
        """Finds and reads the photo this JSON belongs to. Returns
        (img_original, src_image) -- either may be None."""
        if cv2 is None:
            return None, None
        src_image = self.find_source_image(lj, [images_dir] if images_dir else None)
        img_original = None
        if src_image is not None:
            img_original = cv2.imread(src_image)
            if img_original is None:
                print(f"  [OVERLAY WARNING] Source photo found but unreadable: {src_image}")
        return img_original, src_image

    def _regenerate_crack_mask(self, lj, remaining_crack_shapes, remaining_crack_points, img_original, width, height, crack_mask_path, mask_dilation_px):
        """Rebuilds (or removes) the crack training mask after
        filtering."""
        if cv2 is None:
            print("  [MASK WARNING] cv2 not available: binary mask not regenerated.")
            return
        photo_binary_mask = self.compute_binary_mask_from_image(img_original) if img_original is not None else None
        if img_original is not None and photo_binary_mask is None:
            print(f"  [MASK WARNING] Source photo found but computing binary_mask failed: "
                  f"using uniform dilation only for automatic edge detection.")
        elif img_original is None:
            print(f"  [MASK] Source photo '{lj.image_path}' not found "
                  f"(use --images-dir to indicate where to look for it): automatic edge detection "
                  f"skipped, using uniform dilation only (JSON and mask are still updated).")
        crack_mask = self.build_variable_width_crack_mask(remaining_crack_shapes, width, height, mask_dilation_px,
                                                           binary_mask=photo_binary_mask)
        if np.any(crack_mask):
            cv2.imwrite(crack_mask_path, crack_mask)
            print(f"  [MASK] {crack_mask_path} regenerated ({len(remaining_crack_points)} crack(s) remaining).")
        else:
            if os.path.exists(crack_mask_path):
                os.remove(crack_mask_path)
                print(f"  [MASK] {crack_mask_path} removed (no active crack remaining).")
            else:
                print(f"  [MASK] No active crack remaining, no mask to save.")

    def _regenerate_overlay_image(self, lj, new_shapes, remaining_crack_points, img_original, src_image, width, height, mask_dir):
        """Rebuilds the colored -seg overlay photo after filtering."""
        if cv2 is None:
            return
        if img_original is not None:
            remaining_detachment_points = [s["points"] for s in new_shapes if not self.is_crack_shape(s)]
            blue_mask = np.zeros((height, width), dtype=np.uint8)
            green_mask = np.zeros((height, width), dtype=np.uint8)
            self.stamp_points(blue_mask, remaining_crack_points, width, height)
            self.stamp_points(green_mask, remaining_detachment_points, width, height)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mb, mv = cv2.dilate(blue_mask, k), cv2.dilate(green_mask, k)
            segmented_img = img_original.copy()
            segmented_img[mv == 255] = (0, 200, 0)
            segmented_img[mb == 255] = (255, 0, 0)
            img_name, img_ext = os.path.splitext(os.path.basename(lj.image_path))
            seg_path = os.path.join(mask_dir, f"{img_name}-seg{img_ext}")
            cv2.imwrite(seg_path, segmented_img)
            print(f"  [OVERLAY] {seg_path} regenerated.")
        elif src_image is not None:
            pass  # already warned above ("found but unreadable")
        else:
            print(f"  [OVERLAY WARNING] Source photo '{lj.image_path}' not found "
                  f"(use --images-dir to indicate where to look for it): -seg overlay not regenerated.")

    def _regenerate_masks_for_one_json(self, lj, new_shapes, width, height, mask_dilation_px, images_dir):
        """Rebuilds the crack mask and -seg overlay for one JSON's
        remaining shapes."""
        remaining_crack_shapes = [s for s in new_shapes if self.is_crack_shape(s)]
        remaining_crack_points = [s["points"] for s in remaining_crack_shapes]

        base_name_no_ext = os.path.splitext(os.path.basename(lj.path))[0]
        mask_dir = os.path.dirname(os.path.abspath(lj.path))
        crack_mask_path = os.path.join(mask_dir, f"{base_name_no_ext}-crack_mask.png")

        img_original, src_image = self._load_source_photo(lj, images_dir)
        self._regenerate_crack_mask(lj, remaining_crack_shapes, remaining_crack_points, img_original, width, height, crack_mask_path, mask_dilation_px)
        self._regenerate_overlay_image(lj, new_shapes, remaining_crack_points, img_original, src_image, width, height, mask_dir)

    def _process_one_loaded_json(self, lj, incompatible, dry_run, make_backup, images_dir, mask_dilation_px):
        """Removes the incompatible shapes from one loaded JSON, and
        regenerates its mask/overlay to match (unless dry_run)."""
        shape_positions_to_remove = {lj.crack_positions[i] for i in incompatible if i < lj.n_cracks}
        new_shapes = [s for i, s in enumerate(lj.shapes) if i not in shape_positions_to_remove]
        n_removed_here = len(lj.shapes) - len(new_shapes)

        if dry_run:
            print(f"  [DRY-RUN] {os.path.basename(lj.path)}: would remove {n_removed_here} shape(s), "
                  f"JSON/mask/overlay NOT modified.")
            return

        self._backup_json_if_requested(lj, make_backup)
        self._write_filtered_json(lj, new_shapes, n_removed_here)

        width, height = lj.image_width, lj.image_height
        if width and height:
            self._regenerate_masks_for_one_json(lj, new_shapes, width, height, mask_dilation_px, images_dir)

    def remove_incompatible_and_save(self, loaded_jsons, incompatible, dry_run, make_backup, images_dir,
                                      mask_dilation_px=None):
        """mask_dilation_px=None resolves to self.mask_dilation_px at call
        time (see the module docstring for why)."""
        if mask_dilation_px is None:
            mask_dilation_px = self.mask_dilation_px
        if not incompatible:
            print("[OK] No crack discarded: no file modified.")
            return

        removed_idx_sorted = sorted(incompatible)
        print(f"[ACTION] Discarded cracks (common index): {removed_idx_sorted}")

        for lj in loaded_jsons:
            self._process_one_loaded_json(lj, incompatible, dry_run, make_backup, images_dir, mask_dilation_px)

# CLI entry point

def _build_arg_parser(comparator):
    """Builds the CLI argument parser (comparator supplies the mask_dilation_px default)."""
    parser = argparse.ArgumentParser(description="Compares cracks across LabelMe JSON files of the same building group and discards those that are not compatible across the photos.")
    parser.add_argument("json_files", nargs="+", help="At least two JSON files exported from the same building group.")
    parser.add_argument("--sinuosity-threshold", type=float, default=0.30,
                         help="Maximum relative difference allowed in sinuosity (shape) between photos, e.g. 0.30 = 30%% (default: 0.30).")
    parser.add_argument("--length-threshold", type=float, default=None,
                         help="Maximum relative difference allowed in length (cm) between photos (default: disabled).")
    parser.add_argument("--images-dir", default=None, help="Additional folder to search for the original photos when regenerating the -seg overlay.")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be discarded, without modifying any file.")
    parser.add_argument("--no-backup", action="store_true", help="Do not save a .bak copy of the original JSON before modifying it.")
    parser.add_argument("--report", default=None, help="Path of a CSV file where the full comparison table will be saved.")
    parser.add_argument("--mask-dilation-px", type=int, default=comparator.mask_dilation_px,
                         help=f"Dilation (in pixels) applied to the regenerated -crack_mask.png mask "
                              f"(default: {comparator.mask_dilation_px}, read from CRACK_MASK_DILATION_PX in config.py if present). "
                              f"Does not affect the -seg overlay.")
    return parser


def _run_comparison_and_filter(comparator, args):
    """The actual work: compare, report, optionally save a CSV, then
    filter the incompatible cracks out."""
    loaded_jsons = [LoadedJson(p) for p in args.json_files]

    rows, incompatible, extra_note = comparator.compare(loaded_jsons, args.sinuosity_threshold, args.length_threshold)
    comparator.print_report(rows, args.sinuosity_threshold, args.length_threshold, extra_note)

    if args.report:
        comparator.write_csv_report(args.report, rows)

    comparator.remove_incompatible_and_save(loaded_jsons, incompatible, args.dry_run, not args.no_backup, args.images_dir,
                                             mask_dilation_px=args.mask_dilation_px)


def main(argv=None):
    try:
        from config import Config
        cfg = Config()
    except ImportError:
        cfg = None
    comparator = CrackComparator(cfg)

    parser = _build_arg_parser(comparator)
    args = parser.parse_args(argv)

    if len(args.json_files) < 2:
        print("[ERROR] At least two JSON files are needed to compare.", file=sys.stderr)
        sys.exit(1)

    _run_comparison_and_filter(comparator, args)

if __name__ == "__main__":
    main()
