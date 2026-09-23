"""fast_segmentation.py
Interactive OpenCV tool for manual segmentation of cracks and detachments on building facade photos, with A*-guided tracing. Run: python3 fast_segmentation.py
"""
import base64
import csv
import heapq
import json
import os
import shutil
import sys
import time
import uuid

import cv2
import numpy as np
from skimage.morphology import skeletonize

from config import Config, resolve_script_dir


class CrackSegmentation:
    """Wraps the entire interactive segmentation pipeline. All state lives on self.cfg (see config.py), so it can be constructed fresh, shared, or tested independently.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else Config()

        # GUI shell integration points (see crack_segmentation_gui.py) --
        # no-ops in headless use, drained/queried by an optional subclass.
        self._pending_keys = []
        self._window_name = None

    def _prompt_mode_choice(self):
        """Asks the operator to pick "1" (new images) or "2" (reload). Default: terminal input(). A GUI wrapper overrides this with a graphical dialog.
        """
        try:
            return input("Type your choice (1 or 2) and press ENTER: ").strip()
        except Exception:
            return "1"

    def _prompt_pixel_scale(self, current_value):
        """Asks for a new PIXEL_TO_CM_SCALE value (KEY K). Returns a float, or None if invalid/empty. A GUI wrapper overrides this with a graphical dialog.
        """
        print("\n[PROMPT] Switch to the terminal to enter the scale...")
        try:
            return float(input(f"Scale (current: {current_value}): "))
        except ValueError:
            return None

    def _prompt_calibration_distance(self):
        """Asks for the real-world distance (cm) between the two calibration clicks. Returns a float, or None on invalid input. A GUI wrapper overrides this with a graphical dialog.
        """
        try:
            return float(input("Enter the real-world distance in centimeters (cm): "))
        except ValueError:
            return None

    def _pump_extra_events(self):
        """Hook called once per render loop iteration; does nothing by default. A GUI wrapper overrides this to pump its own event loop (e.g. Tk's update()).
        """
        return None

    def _wait_key(self, delay_ms):
        """Reads one key with a timeout -- default: exactly `cv2.waitKey(delay_ms)`. A GUI wrapper overrides this to return -1 immediately instead, since real keystrokes arrive through _pending_keys there.
        """
        return cv2.waitKey(delay_ms)

    def _build_save_warning_canvas(self):
        """Draws the warning text onto a fresh popup canvas. Returns
        (popup_name, popup_canvas)."""
        popup_name = "Save Warning"
        popup_canvas = np.zeros((400, 300, 3), dtype=np.uint8)
        lines = [
            "WARNING:",
            "Saving this image will",
            "automatically proceed",
            "to the next file in",
            "the queue.",
            "",
            "To modify this image",
            "again later, restart",
            "the program in",
            "Mode 2.",
            "",
            "Press any key to close",
            "and complete saving...",
        ]
        y_text = 40
        for line in lines:
            color = (0, 0, 255) if "WARNING" in line else (255, 255, 255)
            scale = 0.55 if "WARNING" in line else 0.45
            thick = 2 if "WARNING" in line else 1
            cv2.putText(popup_canvas, line, (15, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y_text += 25
        return popup_name, popup_canvas

    def _resolve_popup_center_position(self):
        """Centers the popup relative to the main window, offset up-
        left so it doesn't sit directly on top of it."""
        scr_w, scr_h = self._resolve_screen_size_for_popup()
        center_x = max(0, ((scr_w - 300) // 2) - 200)
        center_y = max(0, ((scr_h - 400) // 2) - 100)
        return center_x, center_y

    def _acknowledge_save_warning(self):
        """Shows the mandatory pre-save warning and blocks until acknowledged. A GUI wrapper replaces this with a native dialog instead of an invisible/unsafe cv2 popup.
        """
        popup_name, popup_canvas = self._build_save_warning_canvas()
        center_x, center_y = self._resolve_popup_center_position()

        cv2.namedWindow(popup_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(popup_name, popup_canvas)
        cv2.moveWindow(popup_name, center_x, center_y)
        cv2.setWindowProperty(popup_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)  # Waits for any key press
        cv2.destroyWindow(popup_name)

    def _build_crack_filter_confirm_canvas(self, bldg_tag, json_paths, incompatible):
        """Draws the confirmation text onto a fresh popup canvas.
        Returns (popup_name, popup_canvas)."""
        popup_name = "Confirm Crack Filter"
        popup_canvas = np.zeros((320, 480, 3), dtype=np.uint8)
        lines = [
            f"Group {bldg_tag[2:]}: {len(json_paths)} photos compared.",
            f"{len(incompatible)} incompatible crack(s) found",
            f"(indices: {sorted(incompatible)}).",
            "",
            "They will be removed from the JSON, mask,",
            "and overlay of EVERY photo in the group.",
            "Full details in the console.",
            "",
            "Press [Y] to confirm,",
            "any other key to cancel.",
        ]
        y_text = 40
        for line in lines:
            color = (0, 165, 255) if line.startswith("They will") else (255, 255, 255)
            cv2.putText(popup_canvas, line, (15, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_text += 28
        return popup_name, popup_canvas

    def _resolve_screen_size_for_popup(self):
        """Real window size, falling back to a generic screen size if it can't be read; shared by every centering popup."""
        try:
            _, _, _, _, scr_w, scr_h = cv2.getWindowImageRect("Crack Detector Workspace")
            if scr_w <= 100 or scr_h <= 100:
                return 1920, 1080
            return scr_w, scr_h
        except Exception:
            return 1920, 1080

    def _confirm_crack_filter_removal(self, bldg_tag, json_paths, incompatible):
        """Shows the "remove N incompatible crack(s)?" confirmation and returns True only if the operator confirmed. A GUI wrapper replaces this with a native dialog.
        """
        popup_name, popup_canvas = self._build_crack_filter_confirm_canvas(bldg_tag, json_paths, incompatible)
        scr_w, scr_h = self._resolve_screen_size_for_popup()

        cv2.namedWindow(popup_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(popup_name, popup_canvas)
        cv2.moveWindow(popup_name, max(0, (scr_w - 480) // 2), max(0, (scr_h - 320) // 2))
        cv2.setWindowProperty(popup_name, cv2.WND_PROP_TOPMOST, 1)
        confirm_key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(popup_name)
        return confirm_key in (ord('y'), ord('Y'))

    def _report_fatal_error(self, message):
        """Called right before a fatal startup error exits the process. A GUI wrapper overrides this to also show a graphical error dialog.
        """
        return None

    def _print_pipeline_banner(self):
        print("\n=======================================================")
        print("      CRACK & DETACHMENT SEGMENTATION PIPELINE")
        print("=======================================================")
        print("Choose the data source:")
        print("1) Load NEW images to process (from the 'Images' folder)")
        print("2) Reload ALREADY SEGMENTED images for edits (from 'already processed images')")

    def _load_mode2_queue(self, valid_extensions):
        """MODE 2: rebuilds the queue from already-segmented JSON+image
        pairs in the archive folder. Falls back to Mode 1 if none found."""
        self.cfg.IMAGE_FOLDER = os.path.join(self.cfg.SCRIPT_DIR, "already processed images")
        self.cfg.OUTPUT_FOLDER = self.cfg.IMAGE_FOLDER  # Overwrites the JSON directly in the same archive
        print(f"\n[INFO] Scanning segmented files in: {self.cfg.IMAGE_FOLDER}")

        if not os.path.exists(self.cfg.IMAGE_FOLDER):
            os.makedirs(self.cfg.IMAGE_FOLDER)

        all_files = os.listdir(self.cfg.IMAGE_FOLDER)
        # Case-insensitive lookup by real on-disk name -- an uppercase
        # extension (.PNG, .JPG, common straight off a phone/camera) is
        # still found this way on a case-sensitive filesystem (Linux),
        # not just on macOS/Windows where the filesystem itself already
        # ignores case (which is why this only ever surfaced on Linux).
        files_by_lower_name = {f.lower(): f for f in all_files}

        json_files = [f for f in all_files if f.lower().endswith('.json')]
        for j_file in json_files:
            base_name, _ = os.path.splitext(j_file)
            if base_name.endswith('-seg'):
                base_name = base_name[:-4]

            for ext in valid_extensions:
                real_name = files_by_lower_name.get(f"{base_name}{ext}".lower())
                if real_name:
                    self.cfg.image_queue.append(os.path.join(self.cfg.IMAGE_FOLDER, real_name))
                    break

        if not self.cfg.image_queue:
            print("[WARNING] No valid JSON file found in 'already processed images'. Forcing mode 1.")
            self.cfg.modalita_scelta = "1"

    def _load_mode1_queue(self, valid_extensions):
        """MODE 1: rebuilds the queue from every image in the 'Images'
        input folder."""
        self.cfg.IMAGE_FOLDER = os.path.join(self.cfg.SCRIPT_DIR, "Images")
        # OUTPUT_FOLDER stays pointed at 'segmentated images' for new exports
        self.cfg.OUTPUT_FOLDER = os.path.join(self.cfg.SCRIPT_DIR, self.cfg.folder_seg_img)
        print(f"\n[INFO] Scanning the new images folder: {self.cfg.IMAGE_FOLDER}")

        if not os.path.exists(self.cfg.IMAGE_FOLDER):
            os.makedirs(self.cfg.IMAGE_FOLDER)

        self.cfg.image_queue = [
            os.path.join(self.cfg.IMAGE_FOLDER, f) for f in os.listdir(self.cfg.IMAGE_FOLDER)
            if f.lower().endswith(valid_extensions) and not f.lower().endswith('-seg.jpg')
        ]

    def _finalize_queue_or_exit(self):
        """Sorts the queue chronologically/alphabetically; exits the
        whole program if it ends up empty (nothing to work on)."""
        self.cfg.image_queue.sort()
        if not self.cfg.image_queue:
            error_message = f"No valid file found in: {self.cfg.IMAGE_FOLDER}"
            print(f"[CRITICAL ERROR] {error_message}")
            self._report_fatal_error(error_message)
            sys.exit(1)
        print(f"[INFO] Configuration complete. Found {len(self.cfg.image_queue)} file(s) ready for analysis.")

    def _rebuild_queue_for_mode(self, new_mode):
        """Rebuilds image_queue for the given mode ("1"/"2") without
        re-prompting -- used by the runtime Switch Mode action (code 6).
        Unlike the startup _finalize_queue_or_exit, an empty result
        reverts to the previous mode/queue and returns False instead of
        exiting the whole app."""
        previous_mode = self.cfg.modalita_scelta
        previous_queue = list(self.cfg.image_queue)
        previous_image_folder = self.cfg.IMAGE_FOLDER
        previous_output_folder = self.cfg.OUTPUT_FOLDER

        self.cfg.modalita_scelta = new_mode
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.cfg.image_queue = []
        if new_mode == "2":
            self._load_mode2_queue(valid_extensions)
        if self.cfg.modalita_scelta != "2":
            self._load_mode1_queue(valid_extensions)
        self.cfg.image_queue.sort()

        if not self.cfg.image_queue:
            print(f"[SWITCH MODE] No valid file found for mode {new_mode} -- staying in mode {previous_mode}.")
            self.cfg.modalita_scelta = previous_mode
            self.cfg.image_queue = previous_queue
            self.cfg.IMAGE_FOLDER = previous_image_folder
            self.cfg.OUTPUT_FOLDER = previous_output_folder
            return False

        print(f"[SWITCH MODE] Now in mode {self.cfg.modalita_scelta} -- found {len(self.cfg.image_queue)} file(s).")
        return True

    def build_chronological_queue(self):
        """Scans folders and builds the image processing queue based on user choice."""
        self._print_pipeline_banner()
        scelta = self._prompt_mode_choice()

        self.cfg.modalita_scelta = scelta
        VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.cfg.image_queue = []

        # MODE 2: Reload from 'already processed images'
        if scelta == "2":
            self._load_mode2_queue(VALID_EXTENSIONS)

        # MODE 1: Load new images from 'Images'
        if self.cfg.modalita_scelta != "2":
            self._load_mode1_queue(VALID_EXTENSIONS)

        # Chronological/alphabetical file sort plus a safety exit check
        self._finalize_queue_or_exit()

    def _resolve_render_window_size(self):
        """Real window size (matching render_scene()'s own resolution
        logic), falling back to 1200x900 if it can't be read."""
        try:
            _, _, w_win, h_win = cv2.getWindowImageRect("Crack Detector Workspace")
            if w_win <= 100 or h_win <= 100:
                return 1200, 900
            return w_win, h_win
        except Exception:
            return 1200, 900

    def _compute_letterbox_offset(self, w_win, h_win):
        """The black-letterbox centering offset and displayed size for
        the 1200x900 ROI within the real window."""
        w_roi, h_roi = 1200, 900
        scala_proporzionale = min(w_win / float(w_roi), h_win / float(h_roi))
        w_nuovo = int(w_roi * scala_proporzionale)
        h_nuovo = int(h_roi * scala_proporzionale)
        x_offset = (w_win - w_nuovo) // 2
        y_offset = (h_win - h_nuovo) // 2
        return x_offset, y_offset, w_nuovo, h_nuovo

    def transform_window_to_real_coords(self, x, y):
        """Transforms click coordinates from dynamic window space into real image pixels
        compensating for automatic black-letterbox padding centering offsets.
        """
        w_win, h_win = self._resolve_render_window_size()

        try:
            z_xmin, z_ymin, z_xmax, z_ymax = self.cfg.zoom_box

            # Reproduces the exact scaling math used in the graphical rendering
            x_offset, y_offset, w_nuovo, h_nuovo = self._compute_letterbox_offset(w_win, h_win)

            # Subtracts the black offsets to compute the exact position inside the real image
            click_x_relativo = (x - x_offset) / float(w_nuovo) if w_nuovo > 0 else 0.0
            click_y_relativo = (y - y_offset) / float(h_nuovo) if h_nuovo > 0 else 0.0

            # Clamps values between 0 and 1 so clicks outside the black borders don't crash the masks
            click_x_relativo = max(0.0, min(1.0, click_x_relativo))
            click_y_relativo = max(0.0, min(1.0, click_y_relativo))

            real_x = z_xmin + click_x_relativo * (z_xmax - z_xmin)
            real_y = z_ymin + click_y_relativo * (z_ymax - z_ymin)
            return int(real_x), int(real_y)
        except Exception:
            return x, y

    def transform_real_to_window_coords(self, x_real, y_real):
        """Maps absolute image pixel positions to current 1200x900 viewport screen points."""
        try:
            w_b = self.cfg.zoom_box[2] - self.cfg.zoom_box[0]
            h_b = self.cfg.zoom_box[3] - self.cfg.zoom_box[1]
            if w_b == 0 or h_b == 0: 
                return 0, 0
            return int(((x_real - self.cfg.zoom_box[0]) / w_b) * 1200), int(((y_real - self.cfg.zoom_box[1]) / h_b) * 900)
        except Exception: 
            return 0, 0

    def _stamp_points_on_mask(self, mask, features):
        """Vectorized helper: collects all points from the active traces and performs
        a single numpy assignment instead of a per-point Python loop."""
        xs_parts, ys_parts = [], []
        for feat in features:
            if not feat.get('active', True):
                continue
            path = feat.get('path')
            if not path:
                continue
            pts = np.asarray(path, dtype=np.int64)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            xs_parts.append(pts[:, 0])
            ys_parts.append(pts[:, 1])

        if not xs_parts:
            return

        all_x = np.concatenate(xs_parts)
        all_y = np.concatenate(ys_parts)
        valid = (all_x >= 0) & (all_x < self.cfg.W_img) & (all_y >= 0) & (all_y < self.cfg.H_img)
        mask[all_y[valid], all_x[valid]] = 255


    def _mask_to_rle_rows(self, mask, y_offset=0, x_offset=0):
        """Encodes a binary mask's nonzero pixels as compact row-runs [[y, x_start, x_end], ...], for storing an auto-filled crack shape in width_segments/JSON."""
        rows = []
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return rows
        order = np.lexsort((xs, ys))
        ys, xs = ys[order], xs[order]
        run_start = 0
        n = len(ys)
        for k in range(1, n + 1):
            if k == n or ys[k] != ys[run_start] or xs[k] != xs[k - 1] + 1:
                rows.append([int(ys[run_start] + y_offset), int(xs[run_start] + x_offset), int(xs[k - 1] + x_offset)])
                run_start = k
        return rows


    def _rle_rows_to_mask(self, rle_rows, height, width):
        """Inverse of _mask_to_rle_rows: rebuilds a mask from stored row-runs, skipping any malformed row rather than raising."""
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


    def _mark_centerline_pixel(self, canvas, px, py):
        """Marks one path point's own pixel on the canvas, if in
        bounds."""
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < self.cfg.W_img and 0 <= iy < self.cfg.H_img:
            canvas[iy, ix] = 255

    def _scan_one_side_for_fill(self, canvas, px, py, sign, perp, margin):
        """Scans outward along one perpendicular direction, marking foreground pixels until background or the margin cap. Returns True if anything marked."""
        perp_x, perp_y = perp
        hit = False
        for step in range(1, margin + 1):
            sx = int(round(px + sign * perp_x * step))
            sy = int(round(py + sign * perp_y * step))
            if sx < 0 or sx >= self.cfg.W_img or sy < 0 or sy >= self.cfg.H_img:
                break
            if self.cfg.binary_mask[sy, sx] == 0:
                break
            canvas[sy, sx] = 255
            hit = True
        return hit

    def _autofill_one_point(self, canvas, path, i, i0, i1, margin):
        """Marks one path point's centerline pixel and both perpendicular scans. Returns True if a fill pixel was found (centerline alone doesn't count)."""
        px, py = path[i]
        self._mark_centerline_pixel(canvas, px, py)
        perp = self._compute_local_perpendicular(path, i, i0, i1)
        if perp is None:
            return False
        any_hit = False
        for sign in (1, -1):
            if self._scan_one_side_for_fill(canvas, px, py, sign, perp, margin):
                any_hit = True
        return any_hit

    def _autofill_tract_mask(self, path, i0, i1):
        """Auto-fills a width-edit tract from the crack's real extent in binary_mask: scans perpendicular to the path at each point. Returns an RLE row-list, or [] if nothing was found (fall back to legacy scalar width).
        """
        try:
            n = len(path)
            if n == 0 or i0 < 0 or i1 >= n or i0 > i1:
                return []

            margin = int(self.cfg.WIDTH_EDIT_AUTOFILL_SEARCH_MARGIN_PX)
            canvas = np.zeros((self.cfg.H_img, self.cfg.W_img), dtype=np.uint8)
            any_hit = False

            for i in range(i0, i1 + 1):
                if self._autofill_one_point(canvas, path, i, i0, i1, margin):
                    any_hit = True

            if not any_hit:
                return []
            return self._mask_to_rle_rows(canvas)
        except Exception as e:
            print(f"[WIDTH] Automatic fill detection failed ({e}); tract left empty -- use [+] to widen it manually.", file=sys.stderr)
            return []

    def _compute_local_perpendicular(self, path, i, i0, i1):
        """Local tangent/perpendicular direction at one path point --
        same estimate _autofill_tract_mask uses. None if degenerate."""
        p_prev = path[i - 1] if i - 1 >= i0 else path[i]
        p_next = path[i + 1] if i + 1 <= i1 else path[i]
        dx, dy = (p_next[0] - p_prev[0]), (p_next[1] - p_prev[1])
        tangent_norm = (dx * dx + dy * dy) ** 0.5
        if tangent_norm < 1e-6:
            return None
        return -dy / tangent_norm, dx / tangent_norm

    def _measure_current_extent(self, mask, px, py, side_sign, perp, scan_cap, width, height):
        """How far this side is ALREADY filled -- the FARTHEST filled
        pixel on the ray, so a stray one-pixel gap doesn't collapse it."""
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

    def _measure_contour_limit(self, px, py, side_sign, perp, scan_cap, width, height):
        """How far the crack REALLY extends here per binary_mask --
        growth must never step past this."""
        perp_x, perp_y = perp
        contour_limit = 0
        for step in range(1, scan_cap + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            if self.cfg.binary_mask[sy, sx] == 0:
                break
            contour_limit = step
        return contour_limit

    def _grow_side_at_point(self, mask, px, py, side_sign, perp, cur_extent, delta, scan_cap, width, height):
        """Extends one side outward, capped at the crack's real
        contour -- never shrinks via a growth key."""
        perp_x, perp_y = perp
        contour_limit = self._measure_contour_limit(px, py, side_sign, perp, scan_cap, width, height)
        target = min(cur_extent + delta, max(cur_extent, contour_limit))
        for step in range(1, target + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            mask[sy, sx] = 255

    def _shrink_side_at_point(self, mask, px, py, side_sign, perp, cur_extent, delta, width, height):
        """Retracts one side inward from its current outer edge."""
        perp_x, perp_y = perp
        new_extent = max(0, cur_extent + delta)
        for step in range(new_extent + 1, cur_extent + 1):
            sx = int(round(px + side_sign * perp_x * step))
            sy = int(round(py + side_sign * perp_y * step))
            if 0 <= sx < width and 0 <= sy < height:
                mask[sy, sx] = 0

    def _apply_directional_margin_at_point(self, mask, path, i, i0, i1, side_sign, delta, scan_cap, width, height):
        """Grows or shrinks one side of the crack at a single path
        point."""
        perp = self._compute_local_perpendicular(path, i, i0, i1)
        if perp is None:
            return
        px, py = path[i]
        cur_extent = self._measure_current_extent(mask, px, py, side_sign, perp, scan_cap, width, height)
        if delta > 0:
            self._grow_side_at_point(mask, px, py, side_sign, perp, cur_extent, delta, scan_cap, width, height)
        else:
            self._shrink_side_at_point(mask, px, py, side_sign, perp, cur_extent, delta, width, height)

    def _apply_directional_margin(self, mask, path, i0, i1, side_sign, delta, height, width):
        """Grows or shrinks ONE side of a width-edit tract, following the crack's local perpendicular direction per point -- capped at the real edge in binary_mask, filled solid with no gaps. Modifies and returns `mask` in place.
        """
        if delta == 0 or not path:
            return mask
        n = len(path)
        if i0 < 0 or i1 >= n or i0 > i1:
            return mask
        # Generous safety cap against a pathological/corrupted delta from a hand-edited JSON.
        scan_cap = int(self.cfg.WIDTH_EDIT_AUTOFILL_SEARCH_MARGIN_PX) + int(self.cfg.WIDTH_EDIT_MAX_PX)

        for i in range(i0, i1 + 1):
            self._apply_directional_margin_at_point(mask, path, i, i0, i1, side_sign, delta, scan_cap, width, height)
        return mask

    def _extract_segment_points_and_validity(self, path, i0, i1, height, width):
        """The segment's path points and which of them fall within
        image bounds."""
        seg_pts = np.asarray(path[i0:i1 + 1], dtype=np.int64) if path else np.empty((0, 2), dtype=np.int64)
        valid = np.zeros(0, dtype=bool)
        if seg_pts.ndim == 2 and seg_pts.shape[1] >= 2 and seg_pts.shape[0] > 0:
            valid = (seg_pts[:, 0] >= 0) & (seg_pts[:, 0] < width) & (seg_pts[:, 1] >= 0) & (seg_pts[:, 1] < height)
        return seg_pts, valid

    def _get_segment_margins(self, seg):
        """The segment's independent left/right directional margins,
        defaulting to 0 on any malformed value."""
        try:
            margin_left = int(seg.get('margin_left_px', 0) or 0)
        except (TypeError, ValueError):
            margin_left = 0
        try:
            margin_right = int(seg.get('margin_right_px', 0) or 0)
        except (TypeError, ValueError):
            margin_right = 0
        return margin_left, margin_right

    def _render_fill_segment_mask(self, seg, seg_pts, valid, height, width):
        """Renders an auto-detected-fill segment: the stored RLE shape,
        refined by a signed dilate/erode margin."""
        seg_mask = self._rle_rows_to_mask(seg['fill'], height, width)
        try:
            margin = int(seg.get('width_px', 0))
        except (TypeError, ValueError):
            margin = 0
        if margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            seg_mask = cv2.dilate(seg_mask, k)
        elif margin < 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(margin) + 1, 2 * abs(margin) + 1))
            seg_mask = cv2.erode(seg_mask, k)
            if not np.any(seg_mask) and np.any(valid):
                # Safety: never let a heavy [-] refinement erase the tract
                # entirely -- fall back to its bare centerline.
                seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
        return seg_mask

    def _render_legacy_segment_mask(self, seg, seg_pts, valid, height, width):
        """Renders a legacy plain-scalar segment: dilates straight from
        the centerline points."""
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        if np.any(valid):
            seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
            try:
                w = max(0, int(seg.get('width_px', 0)))
            except (TypeError, ValueError):
                w = 0
            if w > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
                seg_mask = cv2.dilate(seg_mask, k)
        return seg_mask

    def _apply_directional_margins_to_segment(self, seg_mask, path, i0, i1, margin_left, margin_right, seg_pts, valid, height, width):
        """Applies the independent per-side margins on top of the base
        mask, never letting a heavy shrink erase the tract outright."""
        if margin_left:
            seg_mask = self._apply_directional_margin(seg_mask, path, i0, i1, side_sign=+1,
                                                  delta=margin_left, height=height, width=width)
        if margin_right:
            seg_mask = self._apply_directional_margin(seg_mask, path, i0, i1, side_sign=-1,
                                                  delta=margin_right, height=height, width=width)
        if (margin_left or margin_right) and not np.any(seg_mask) and np.any(valid):
            # Same safety floor as the [-] case above, for the directional
            # keys: a heavy [{/}] shrink can never erase the tract outright.
            seg_mask[seg_pts[valid, 1], seg_pts[valid, 0]] = 255
        return seg_mask

    def _close_gaps_in_segment_mask(self, seg_mask):
        """Morphological closing to merge disconnected per-point ray fragments into one solid band."""
        if np.any(seg_mask):
            close_px = max(1, int(self.cfg.WIDTH_EDIT_FILL_CLOSE_PX))
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, close_k)
        return seg_mask

    def _render_width_segment_mask(self, path, i0, i1, seg, height, width):
        """Builds the mask for ONE width_segments tract, shared by the live preview and the export builder: an auto-filled RLE shape or a legacy centerline dilation, refined by width_px and independent per-side margins.
        """
        seg_pts, valid = self._extract_segment_points_and_validity(path, i0, i1, height, width)
        margin_left, margin_right = self._get_segment_margins(seg)

        if seg.get('fill'):
            seg_mask = self._render_fill_segment_mask(seg, seg_pts, valid, height, width)
        else:
            seg_mask = self._render_legacy_segment_mask(seg, seg_pts, valid, height, width)
            if not (margin_left or margin_right):
                return seg_mask

        seg_mask = self._apply_directional_margins_to_segment(seg_mask, path, i0, i1, margin_left, margin_right, seg_pts, valid, height, width)

        # Closes gaps between adjacent per-point rays before returning.
        return self._close_gaps_in_segment_mask(seg_mask)

    def _compute_gsd_offset_px(self, gsd, target_mm, ceiling_px):
        """The actual per-side offset (px) computation from GSD/target
        width, clamped to the safety ceiling."""
        target_width_px = target_mm / gsd
        # Subtracts the 1px centerline, splits the remainder between the two sides.
        offset_px = max(0, round((target_width_px - 1) / 2.0))
        return min(offset_px, ceiling_px)

    def _apply_gsd_offset(self, offset_px, gsd, target_mm):
        """Applies the resolved offset to both config constants (kept equal on purpose) and reports it."""
        self.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX = offset_px
        self.cfg.CRACK_MASK_DILATION_PX = offset_px
        print(f"[GSD CALIBRATION] IMAGE_GSD_MM_PER_PX={gsd:.4f} mm/px, "
              f"CRACK_TARGET_PHYSICAL_WIDTH_MM={target_mm:.2f}mm -> "
              f"automatic crack band set to {offset_px}px per side "
              f"(~{2 * offset_px + 1}px total).")

    def resolve_crack_edge_width_from_gsd(self):
        """Derives CRACK_AUTO_EDGE_MAX_OFFSET_PX/CRACK_MASK_DILATION_PX from IMAGE_GSD_MM_PER_PX so the exported crack band stays a consistent physical width; a no-op if GSD isn't set. Returns the resolved offset in px, or None.
        """
        gsd = self.cfg.IMAGE_GSD_MM_PER_PX or 0.0
        if gsd <= 0:
            return None

        target_mm = self.cfg.CRACK_TARGET_PHYSICAL_WIDTH_MM
        ceiling_px = self.cfg.CRACK_AUTO_EDGE_GSD_SAFETY_CEILING_PX
        offset_px = self._compute_gsd_offset_px(gsd, target_mm, ceiling_px)
        self._apply_gsd_offset(offset_px, gsd, target_mm)
        return offset_px


    def _scan_one_side_extent(self, px, py, sign, perp, max_offset, mask_ref, width, height):
        """Counts contiguous foreground pixels along one perpendicular
        direction, capped at max_offset steps."""
        perp_x, perp_y = perp
        extent = 0
        for step in range(1, max_offset + 1):
            sx = int(round(px + sign * perp_x * step))
            sy = int(round(py + sign * perp_y * step))
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                break
            if mask_ref[sy, sx] == 0:
                break
            extent = step
        return extent

    def _detect_edge_offsets_at_point(self, path, i, n, mask_ref, height, width, max_offset, left_offsets, right_offsets):
        """Fills in left_offsets[i]/right_offsets[i] for one path
        point -- left is unchanged, no detectable coverage."""
        perp = self._compute_local_perpendicular(path, i, 0, n - 1)
        if perp is None:
            return
        px, py = path[i]
        left_offsets[i] = self._scan_one_side_extent(px, py, 1, perp, max_offset, mask_ref, width, height)
        right_offsets[i] = self._scan_one_side_extent(px, py, -1, perp, max_offset, mask_ref, width, height)

    def _auto_detect_crack_edge_offsets(self, path, mask_ref, height, width, max_offset):
        """For every path point, scans outward on each side against mask_ref until background or max_offset, detecting the crack's own two real edges. Returns (left_offsets, right_offsets), one int per path point.
        """
        n = len(path) if path else 0
        left_offsets = [0] * n
        right_offsets = [0] * n
        if n == 0 or mask_ref is None:
            return left_offsets, right_offsets

        max_offset = max(0, int(max_offset))
        for i in range(n):
            self._detect_edge_offsets_at_point(path, i, n, mask_ref, height, width, max_offset, left_offsets, right_offsets)
        return left_offsets, right_offsets

    def _edge_point_at_offset(self, path, i, n, sign, offset):
        """One edge point: the path point itself if offset is 0,
        otherwise shifted along the local perpendicular by `offset`."""
        px, py = path[i]
        if offset == 0:
            return (px, py)
        perp = self._compute_local_perpendicular(path, i, 0, n - 1)
        if perp is None:
            return (px, py)
        perp_x, perp_y = perp
        return (px + sign * perp_x * offset, py + sign * perp_y * offset)

    def _fill_one_ribbon_segment(self, canvas, path, n, left_offsets, right_offsets, i):
        """Fills the small quad connecting path points i and i+1 along
        their two detected edges."""
        l0 = self._edge_point_at_offset(path, i, n, 1, left_offsets[i])
        l1 = self._edge_point_at_offset(path, i + 1, n, 1, left_offsets[i + 1])
        r0 = self._edge_point_at_offset(path, i, n, -1, right_offsets[i])
        r1 = self._edge_point_at_offset(path, i + 1, n, -1, right_offsets[i + 1])
        quad = np.array([l0, l1, r1, r0], dtype=np.int32)
        cv2.fillPoly(canvas, [quad], 255)

    def _build_single_point_ribbon(self, path, height, width):
        """Special case: a single-point "crack" just marks that one
        pixel."""
        canvas = np.zeros((height, width), dtype=np.uint8)
        px, py = path[0]
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < width and 0 <= iy < height:
            canvas[iy, ix] = 255
        return canvas

    def _build_ribbon_mask_from_edge_offsets(self, path, left_offsets, right_offsets, height, width):
        """Fills a mask circumscribing the crack between its two detected edges, one small quad per consecutive point pair so sharp curves can never self-intersect.
        """
        n = len(path) if path else 0
        if n == 0:
            return np.zeros((height, width), dtype=np.uint8)
        if n == 1:
            return self._build_single_point_ribbon(path, height, width)

        canvas = np.zeros((height, width), dtype=np.uint8)
        for i in range(n - 1):
            self._fill_one_ribbon_segment(canvas, path, n, left_offsets, right_offsets, i)
        return canvas

    def _apply_offset_floor(self, left_off, right_off, floor_px):
        """Applies a uniform minimum on both detected offsets, never a ceiling, so a stretch with no visible coverage still gets some dilation."""
        floor_px = max(0, int(floor_px))
        if floor_px:
            left_off = [max(v, floor_px) for v in left_off]
            right_off = [max(v, floor_px) for v in right_off]
        return left_off, right_off

    def _compute_auto_edge_crack_mask(self, path, height, width, floor_px=0, mask_ref=None, max_offset=None):
        """Detects both edges of one crack against mask_ref and renders the resulting ribbon, floored at floor_px and morphologically closed to avoid gaps.
        """
        if mask_ref is None:
            mask_ref = self.cfg.binary_mask
        if max_offset is None:
            max_offset = self.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX
        left_off, right_off = self._auto_detect_crack_edge_offsets(path, mask_ref, height, width, max_offset)
        left_off, right_off = self._apply_offset_floor(left_off, right_off, floor_px)
        ribbon = self._build_ribbon_mask_from_edge_offsets(path, left_off, right_off, height, width)
        return self._close_gaps_in_segment_mask(ribbon)

    def _compute_widths_and_special_segments(self, f, n, default_dilation_px):
        """Builds the per-point width array and the list of "special"
        segments (auto-filled/margin-refined) needing their own renderer."""
        widths = [max(0, int(default_dilation_px))] * n
        special_segments = []
        for seg in (f.get('width_segments') or []):
            try:
                i0, i1 = int(seg.get('i0', 0)), int(seg.get('i1', 0))
            except (TypeError, ValueError):
                continue
            if i0 > i1:
                i0, i1 = i1, i0
            i0 = max(0, min(i0, n - 1))
            i1 = max(0, min(i1, n - 1))
            try:
                has_margin = bool(int(seg.get('margin_left_px', 0) or 0) or int(seg.get('margin_right_px', 0) or 0))
            except (TypeError, ValueError):
                has_margin = False
            if seg.get('fill') or has_margin:
                special_segments.append((i0, i1, seg))
                for i in range(i0, i1 + 1):
                    widths[i] = None
            else:
                try:
                    w = max(0, int(seg.get('width_px', 0)))
                except (TypeError, ValueError):
                    continue
                for i in range(i0, i1 + 1):
                    widths[i] = w
        return widths, special_segments

    def _compute_auto_edge_ribbon(self, path, height, width, default_dilation_px):
        """Tries the automatic two-edge crack outline; None if disabled
        or detection fails."""
        if not self.cfg.CRACK_MASK_AUTO_EDGE_ENABLED:
            return None
        try:
            return self._compute_auto_edge_crack_mask(path, height, width, floor_px=default_dilation_px)
        except Exception as e:
            print(f"[CRACK MASK] Automatic edge detection failed ({e}); using uniform dilation only.", file=sys.stderr)
            return None

    def _dilate_scalar_width_runs(self, out, path, widths, n, height, width):
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
            run_pts = np.asarray(path[i:j + 1], dtype=np.int64)
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

    def _apply_special_segments(self, out, special_segments, path, height, width):
        """ORs in every special (auto-filled/margin-refined) segment's
        own rendered mask."""
        for seg_i0, seg_i1, seg in special_segments:
            out = cv2.bitwise_or(out, self._render_width_segment_mask(path, seg_i0, seg_i1, seg, height, width))
        return out

    def _build_one_crack_mask_contribution(self, out, f, height, width, default_dilation_px):
        """Adds one crack's contribution (base dilation + special
        segments) to the running export mask."""
        path = f.get('path')
        if not path:
            return out
        n = len(path)
        widths, special_segments = self._compute_widths_and_special_segments(f, n, default_dilation_px)

        # Covers the whole path with the automatic two-edge outline when enabled.
        auto_ribbon = self._compute_auto_edge_ribbon(path, height, width, default_dilation_px)
        if auto_ribbon is not None:
            out = cv2.bitwise_or(out, auto_ribbon)
        else:
            out = self._dilate_scalar_width_runs(out, path, widths, n, height, width)

        return self._apply_special_segments(out, special_segments, path, height, width)

    def _build_variable_width_crack_mask(self, cracks, height, width, default_dilation_px):
        """Builds the export crack mask: each crack's path is dilated by default_dilation_px, or auto-edge-detected, or overridden per-tract via saved_cracks[i]['width_segments'] (see [A] key / _render_width_segment_mask).
        """
        out = np.zeros((height, width), dtype=np.uint8)
        for f in cracks:
            if not f.get('active', True):
                continue
            out = self._build_one_crack_mask_contribution(out, f, height, width, default_dilation_px)
        return out

    def recalculate_masks(self):
        """Safely refreshes overlay arrays via a vectorized numpy assignment (_stamp_points_on_mask), pixel-by-pixel identical to the old per-point loop.
        """
        try:
            self.cfg.blue_visual_mask[:, :] = 0
            self.cfg.green_visual_mask[:, :] = 0
            self._stamp_points_on_mask(self.cfg.blue_visual_mask, self.cfg.saved_cracks)
            self._stamp_points_on_mask(self.cfg.green_visual_mask, self.cfg.saved_detachments)
        except Exception as e:
            print(f"[RUNTIME EXCEPTION] Overlay mask refresh anomaly: {e}", file=sys.stderr)

    def refresh_zoom_viewport(self):
        """Safely computes zoom bounds preventing any possible division-by-zero."""
        try:
            if self.cfg.zoom_factor <= 1.0:
                self.cfg.zoom_box = [0, 0, self.cfg.W_img, self.cfg.H_img]
            else:
                w, h = int(self.cfg.W_img / self.cfg.zoom_factor), int(self.cfg.H_img / self.cfg.zoom_factor)
                x_min = max(0, min(self.cfg.zoom_center[0] - w // 2, self.cfg.W_img - w))
                y_min = max(0, min(self.cfg.zoom_center[1] - h // 2, self.cfg.H_img - h))
                self.cfg.zoom_box = [x_min, y_min, x_min + w, y_min + h]
        except Exception as e:
            self.cfg.zoom_box = [0, 0, self.cfg.W_img, self.cfg.H_img]
            print(f"[VIEWPORT EXCEPTION] Fallback zoom anomaly: {e}", file=sys.stderr)

    def calcola_area_poligono_cm2(self, percorso_pixel, scala):
        """Computes detachment polygon closed surface area in square 
        centimeters using Shoelace theorem."""
        if len(percorso_pixel) < 3:
            return 0.0
        try:
            # Correctly extracts the X and Y coordinates from each point (tuple)
            x = [float(p[0]) for p in percorso_pixel]
            y = [float(p[1]) for p in percorso_pixel]
            area_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return area_px * (scala ** 2)
        except Exception as e:
            print(f"[AREA ERROR] {e}")
            return 0.0

    def resume_last_building_index(self):
        """Retrieves the last building index used, from the processed-files folder."""
        processed_folder = os.path.join(self.cfg.SCRIPT_DIR, "already processed images")
        max_idx = None
        if os.path.exists(processed_folder):
            for f in os.listdir(processed_folder):
                if "__BLDG" in f:
                    try:
                        # Extracts the digits right after the __BLDG tag
                        part = f.split("__BLDG")[-1]
                        idx_str = ''.join(filter(str.isdigit, part[:3]))
                        if idx_str:
                            idx = int(idx_str)
                            if max_idx is None or idx > max_idx:
                                max_idx = idx
                    except ValueError:
                        continue
        return max_idx

    def _compute_pathfinding_roi(self, p1, p2, tool_mode):
        """Crops a padded ROI around the two endpoints so A* only
        searches a small local window, not the whole image."""
        start_x, start_y = int(p1[0]), int(p1[1])
        end_x, end_y = int(p2[0]), int(p2[1])
        working_mask = self.cfg.skeleton_mask if tool_mode == 'crack' else self.cfg.binary_mask

        padding = 80
        roi_x_min = max(0, min(start_x, end_x) - padding)
        roi_x_max = min(self.cfg.W_img, max(start_x, end_x) + padding)
        roi_y_min = max(0, min(start_y, end_y) - padding)
        roi_y_max = min(self.cfg.H_img, max(start_y, end_y) + padding)

        local_start = (start_x - roi_x_min, start_y - roi_y_min)
        local_end = (end_x - roi_x_min, end_y - roi_y_min)
        local_mask = working_mask[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        return (roi_x_min, roi_y_min, roi_x_max, roi_y_max), local_start, local_end, local_mask

    def _compute_local_edge_mask(self, tool_mode, roi_bounds):
        """Detachment tool only: morphological-gradient edge mask, computed on a 1px-padded ROI crop instead of the whole image for speed."""
        if tool_mode != 'detachment':
            return None
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
        # A 3x3 kernel's gradient only depends on each pixel's 8 neighbors, so a
        # 1px-padded crop gives a pixel-identical result to the whole image.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_pad = 1  # = radius of the 3x3 kernel
        epx0, epy0 = max(0, roi_x_min - edge_pad), max(0, roi_y_min - edge_pad)
        epx1, epy1 = min(self.cfg.W_img, roi_x_max + edge_pad), min(self.cfg.H_img, roi_y_max + edge_pad)
        padded_local_mask = self.cfg.binary_mask[epy0:epy1, epx0:epx1]
        padded_edge_mask = cv2.morphologyEx(padded_local_mask, cv2.MORPH_GRADIENT, kernel)
        return padded_edge_mask[
            roi_y_min - epy0: roi_y_min - epy0 + (roi_y_max - roi_y_min),
            roi_x_min - epx0: roi_x_min - epx0 + (roi_x_max - roi_x_min)
        ]

    def _snap_point_to_nearest_white(self, point, local_mask, local_w, local_h):
        """Snaps a clicked point onto the nearest traceable (white)
        pixel within a 31x31 window, if it didn't land on one exactly."""
        lx, ly = point
        if not (0 <= lx < local_w and 0 <= ly < local_h and local_mask[ly, lx] != 255):
            return point
        r_min, r_max = max(0, ly-15), min(local_h, ly+16)
        c_min, c_max = max(0, lx-15), min(local_w, lx+16)
        white_pts = np.argwhere(local_mask[r_min:r_max, c_min:c_max] == 255)
        if len(white_pts) == 0:
            return point
        dist = np.sum((white_pts - [ly-r_min, lx-c_min])**2, axis=1)
        best = white_pts[np.argmin(dist)]
        return (c_min + best[1], r_min + best[0])

    def _validate_pathfinding_endpoints(self, local_start, local_end, local_mask, local_w, local_h):
        """True if both (possibly snapped) endpoints land on a
        traceable (white) pixel within the local ROI."""
        for (lx, ly) in (local_start, local_end):
            if lx < 0 or lx >= local_w or ly < 0 or ly >= local_h:
                return False
        return local_mask[local_start[1], local_start[0]] == 255 and local_mask[local_end[1], local_end[0]] == 255

    def a_star_pathfinding(self, p1, p2, tool_mode='crack'):
        """Ultra-high-speed A* pathfinding bounded within a dynamic ROI box, with an HUD timeout alert. The search loop stays inline for performance (a per-pixel hot loop)."""
        try:
            start_time = time.time()
            TIMEOUT_LIMIT = 3.0
            self.cfg.pathfinding_timeout_triggered = False

            roi_bounds, local_start, local_end, local_mask = self._compute_pathfinding_roi(p1, p2, tool_mode)
            roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
            local_h, local_w = local_mask.shape
            if local_h == 0 or local_w == 0:
                return []

            local_edge_mask = self._compute_local_edge_mask(tool_mode, roi_bounds)

            local_start = self._snap_point_to_nearest_white(local_start, local_mask, local_w, local_h)
            local_end = self._snap_point_to_nearest_white(local_end, local_mask, local_w, local_h)
            if not self._validate_pathfinding_endpoints(local_start, local_end, local_mask, local_w, local_h):
                return []

            h_start = ((local_start[0] - local_end[0])**2 + (local_start[1] - local_end[1])**2) ** 0.5
            queue = [(h_start, 0, local_start)]
            seen = {local_start: 0}
            came_from = {}

            movimenti = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
            ]

            step_counter = 0
            while queue:
                _, g_score, curr = heapq.heappop(queue)
                cx, cy = curr[0], curr[1]

                if curr == local_end:
                    global_path = []
                    while curr in came_from:
                        global_path.append((curr[0] + roi_x_min, curr[1] + roi_y_min))
                        curr = came_from[curr]
                    global_path.append((local_start[0] + roi_x_min, local_start[1] + roi_y_min))
                    return global_path[::-1]

                if g_score > seen.get(curr, float('inf')):
                    continue

                step_counter += 1
                if step_counter % 500 == 0:
                    if time.time() - start_time > TIMEOUT_LIMIT:
                        print(f"[TIMEOUT] Calculations exceeded {TIMEOUT_LIMIT}s limit.")
                        self.cfg.pathfinding_timeout_triggered = True
                        return []

                for r_m, c_m, dist_g in movimenti:
                    nx, ny = cx + c_m, cy + r_m
                    nxt = (nx, ny)

                    if 0 <= nx < local_w and 0 <= ny < local_h:
                        if tool_mode == 'crack':
                            base_weight = 1 if local_mask[ny, nx] == 255 else 150
                        else:
                            if local_edge_mask is not None and local_edge_mask[ny, nx] == 255:
                                base_weight = 1
                            elif local_mask[ny, nx] == 255:
                                base_weight = 8
                            else:
                                base_weight = 100

                        nxt_g = g_score + (base_weight * dist_g)

                        if nxt_g < seen.get(nxt, float('inf')):
                            seen[nxt] = nxt_g
                            came_from[nxt] = curr
                            h_score = ((nx - local_end[0])**2 + (ny - local_end[1])**2) ** 0.5
                            heapq.heappush(queue, (nxt_g + h_score, nxt_g, nxt))

        except Exception as e:
            print(f"[PATHFINDING RUNTIME ERROR] Bounded A* crash: {e}", file=sys.stderr)
            return []


    def _path_length(self, pts):
        """Sum of euclidean distances between consecutive points, used by [V] to sanity-check results against each crack's original length."""
        total = 0.0
        for i in range(1, len(pts)):
            total += ((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
        return total



    def _prepare_export_folders(self, cartella_base):
        """Ensures 'already processed images' and OUTPUT_FOLDER exist
        before anything gets written into them."""
        processed_folder = os.path.join(cartella_base, "already processed images")
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        if not os.path.exists(self.cfg.OUTPUT_FOLDER):
            os.makedirs(self.cfg.OUTPUT_FOLDER)
        return processed_folder

    def _get_current_image_base64(self):
        """Returns the current image's base64 data, reusing the bytes
        cached at load time instead of re-reading the file."""
        if self.cfg.CURRENT_IMAGE_RAW_BYTES is not None:
            return base64.b64encode(self.cfg.CURRENT_IMAGE_RAW_BYTES).decode('utf-8')
        with open(self.cfg.CURRENT_IMAGE_PATH, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _build_labelme_data_dict(self, img_b64):
        """Builds the base LabelMe JSON structure; shapes are filled in
        separately."""
        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(self.cfg.CURRENT_IMAGE_PATH),
            "imageData": img_b64,
            "imageHeight": self.cfg.H_img,
            "imageWidth": self.cfg.W_img,
            "pixel_to_cm_scale": self.cfg.PIXEL_TO_CM_SCALE
        }

    def _build_crack_shapes(self):
        """Builds the LabelMe shape dicts for every active crack."""
        shapes = []
        for f in self.cfg.saved_cracks:
            if f.get('active', True):
                shape_dict = {
                    "label": "crack",
                    "points": [[float(pt[0]), float(pt[1])] for pt in f['path']],
                    "group_id": None, "shape_type": "linestrip", "flags": {}
                }
                if f.get('width_segments'):
                    shape_dict["width_segments"] = f['width_segments']
                shapes.append(shape_dict)
        return shapes

    def _build_detachment_shapes(self):
        """Builds the LabelMe shape dicts for every active
        detachment."""
        shapes = []
        for d in self.cfg.saved_detachments:
            if d.get('active', True):
                shapes.append({
                    "label": "detachment",
                    "points": [[float(pt[0]), float(pt[1])] for pt in d['path']],
                    "group_id": None, "shape_type": "polygon", "flags": {}
                })
        return shapes

    def _assign_shape_group_id_and_label(self, shape):
        """Assigns a unique group_id if missing, and tags the label
        with the source image name if not already tagged."""
        group_id = shape.get('group_id')
        if group_id is None:
            group_id = int(uuid.uuid4().int % 1000000)
        label = shape.get('label', 'crack')
        if "_" not in label:
            nome_foto_corrente = os.path.basename(self.cfg.CURRENT_IMAGE_PATH)
            label = f"{label}_{nome_foto_corrente}"
        return group_id, label

    def _clean_labelme_shape(self, shape):
        """Rebuilds one shape with a protected group_id/label and
        guaranteed float points."""
        group_id, label = self._assign_shape_group_id_and_label(shape)
        shape_aggiornata = {
            "label": label,
            "points": [[float(pt[0]), float(pt[1])] for pt in shape.get('points', [])],
            "group_id": group_id,
            "shape_type": shape.get('shape_type', 'linestring'),
            "flags": shape.get('flags', {})
        }
        if shape.get("width_segments"):
            shape_aggiornata["width_segments"] = shape["width_segments"]
        return shape_aggiornata

    def _finalize_labelme_shapes(self):
        """Builds every shape (cracks + detachments), then cleans each
        one with a protected group_id/label."""
        shapes = self._build_crack_shapes() + self._build_detachment_shapes()
        return [self._clean_labelme_shape(s) for s in shapes]

    def _write_labelme_json(self, labelme_data):
        """Writes the LabelMe JSON to JSON_OUTPUT_PATH."""
        with open(self.cfg.JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] JSON configuration generated at: {self.cfg.JSON_OUTPUT_PATH}")

    def _export_segmented_preview_image(self):
        """Writes the colored crack/detachment overlay PNG, if enabled
        ([M])."""
        if not self.cfg.SAVE_SEG_IMAGE:
            return
        segmented_img = self.cfg.img_original.copy()
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mv, mb = cv2.dilate(self.cfg.green_visual_mask, k), cv2.dilate(self.cfg.blue_visual_mask, k)
        segmented_img[mv == 255] = (0, 200, 0)
        segmented_img[mb == 255] = (255, 0, 0)
        cv2.imwrite(self.cfg.EXPORT_IMAGE_PATH, segmented_img)
        print(f"[SUCCESS] Mask overlay map generated at: {self.cfg.EXPORT_IMAGE_PATH}")

    def _build_detachment_area_mask(self):
        """Fills a full-area binary mask for every active detachment
        polygon, not just its outline."""
        detachment_mask_out = np.zeros((self.cfg.H_img, self.cfg.W_img), dtype=np.uint8)
        for d in self.cfg.saved_detachments:
            if d.get('active', True) and len(d.get('path', [])) >= 3:
                poly_pts = np.array([[int(pt[0]), int(pt[1])] for pt in d['path']], dtype=np.int32)
                cv2.fillPoly(detachment_mask_out, [poly_pts], 255)
        return detachment_mask_out

    def _write_mask_if_nonempty(self, mask, path, success_label):
        """Writes a binary mask PNG only if it has at least one
        nonzero pixel. Returns whether it was written."""
        if np.any(mask):
            cv2.imwrite(path, mask)
            print(f"[SUCCESS] {success_label} saved: {path}")
            return True
        return False

    def _export_binary_masks(self):
        """Writes the crack/detachment binary training masks, skipping
        any that would be entirely empty."""
        try:
            crack_mask_out = self._build_variable_width_crack_mask(self.cfg.saved_cracks, self.cfg.H_img, self.cfg.W_img, self.cfg.CRACK_MASK_DILATION_PX)
            detachment_mask_out = self._build_detachment_area_mask()

            base_name_no_ext = os.path.splitext(os.path.basename(self.cfg.JSON_OUTPUT_PATH))[0]
            mask_dir = os.path.dirname(self.cfg.JSON_OUTPUT_PATH)
            crack_mask_path = os.path.join(mask_dir, f"{base_name_no_ext}-crack_mask.png")
            detachment_mask_path = os.path.join(mask_dir, f"{base_name_no_ext}-detachment_mask.png")

            if not self._write_mask_if_nonempty(crack_mask_out, crack_mask_path, "Binary crack mask"):
                print("[MASK EXPORT] No active crack: crack mask not generated (nothing to save).")
            if not self._write_mask_if_nonempty(detachment_mask_out, detachment_mask_path, "Binary detachment mask (filled area)"):
                print("[MASK EXPORT] No active detachment: detachment mask not generated (nothing to save).")
        except Exception as mask_error:
            print(f"[MASK EXPORT WARNING] Could not save the binary masks: {mask_error}", file=sys.stderr)

    def _compute_export_summary_stats(self):
        """Returns (num_cracks, num_detachments, total_area_cm2,
        minutes_spent) for the CSV report row."""
        num_cracks = sum(1 for f in self.cfg.saved_cracks if f.get('active', True))
        num_detachments = sum(1 for d in self.cfg.saved_detachments if d.get('active', True))
        total_detachments_area_cm2 = sum(self.calcola_area_poligono_cm2(d['path'], self.cfg.PIXEL_TO_CM_SCALE) for d in self.cfg.saved_detachments if d.get('active', True))
        tempo_totale_secondi = self.cfg.total_elapsed_paused_time + (time.time() - self.cfg.image_load_time)
        tempo_minuti = round(tempo_totale_secondi / 60.0, 2)
        return num_cracks, num_detachments, total_detachments_area_cm2, tempo_minuti

    def _append_csv_report_row(self, cartella_base, num_cracks, num_detachments, total_area, minutes):
        """Appends one row to segmentation_summary_report.csv, writing
        the header first if the file is new."""
        csv_output_path = os.path.join(cartella_base, "segmentation_summary_report.csv")
        f_exists = os.path.exists(csv_output_path)
        entry_row = {
            'filename': os.path.basename(self.cfg.CURRENT_IMAGE_PATH),
            'cracks_count': num_cracks,
            'cracks_length_cm': round(self.cfg.total_cracks_length_cm, 2),
            'detachments_count': num_detachments,
            'detachments_area_cm2': round(total_area, 2),
            'scale_used': round(self.cfg.PIXEL_TO_CM_SCALE, 5),
            'time_spent_minutes': minutes
        }
        with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=['filename', 'cracks_count', 'cracks_length_cm', 'detachments_count', 'detachments_area_cm2', 'scale_used', 'time_spent_minutes']
            )
            if not f_exists:
                writer.writeheader()
            writer.writerow(entry_row)
        print(f"[CSV APPEND] Data instantly saved to the report: {csv_output_path}")

    def _play_save_beep(self):
        """Plays a short platform-specific audio cue confirming the
        save."""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["say", "saved"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception:
            pass

    def _archive_saved_files(self, processed_folder):
        """Mode 1: copies the JSON and moves the source image into
        'already processed images'. Mode 2: leaves both in place."""
        if self.cfg.modalita_scelta == "1":
            dest_json_path = os.path.join(processed_folder, os.path.basename(self.cfg.JSON_OUTPUT_PATH))
            dest_img_path = os.path.join(processed_folder, os.path.basename(self.cfg.CURRENT_IMAGE_PATH))
            shutil.copy2(self.cfg.JSON_OUTPUT_PATH, dest_json_path)
            if os.path.exists(self.cfg.CURRENT_IMAGE_PATH):
                shutil.move(self.cfg.CURRENT_IMAGE_PATH, dest_img_path)
            if self.cfg.CURRENT_IMAGE_PATH in self.cfg.image_queue:
                self.cfg.image_queue.remove(self.cfg.CURRENT_IMAGE_PATH)
        else:
            print(f"[MODE 2] JSON file overwritten at: {self.cfg.JSON_OUTPUT_PATH}")

    def export_labelme_format(self):
        """Saves standard Labelme JSON, segmented overview image and
        appends instant entry to CSV report with an acoustic beep alert."""
        # LOCAL EMERGENCY IMPORT FOR SAFETY
        import csv
        import sys
        import json
        import base64
        import shutil
        import os
        import numpy as np
        import cv2

        self._acknowledge_save_warning()
        cartella_base = str(resolve_script_dir())

        try:
            processed_folder = self._prepare_export_folders(cartella_base)
            img_b64 = self._get_current_image_base64()
            labelme_data = self._build_labelme_data_dict(img_b64)
            labelme_data["shapes"] = self._finalize_labelme_shapes()

            self._write_labelme_json(labelme_data)
            self._export_segmented_preview_image()
            self._export_binary_masks()

            num_cracks, num_detachments, total_area, minutes = self._compute_export_summary_stats()
            self._append_csv_report_row(cartella_base, num_cracks, num_detachments, total_area, minutes)

            self._play_save_beep()
            self._archive_saved_files(processed_folder)

            self.cfg.save_timestamp = time.time()
        except Exception as e:
            print(f"[IO ERROR] Physical file transfer or serialization failed: {e}", file=sys.stderr)

    def _signal_import_warning(self, message):
        """Raises the on-screen banner warning the operator that a crack-projection attempt (W/L) did not import anything.
        """
        self.cfg.import_warning_triggered = True
        self.cfg.import_warning_message = message
        self.cfg.import_warning_timestamp = time.time()


    def _read_grayscale_and_color(self, base_img_path, new_img_path):
        """Reads the previous (grayscale) and current (color+grayscale)
        images for SIFT matching. Returns None if either is unreadable."""
        img1 = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
        img2_original = cv2.imread(new_img_path)
        if img1 is None or img2_original is None:
            return None
        img2 = cv2.cvtColor(img2_original, cv2.COLOR_BGR2GRAY)
        return img1, img2_original, img2

    def _find_good_sift_matches(self, img1, img2):
        """SIFT-detects and FLANN-matches features, keeping only
        matches passing the Lowe ratio test. None if too little detail."""
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return None

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_match in matches:
            if len(m_match) == 2:
                m, n = m_match[0], m_match[1]
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return kp1, kp2, good_matches

    def _compute_warp_homography(self, kp1, kp2, good_matches):
        """RANSAC-fits a homography from the good matches. None if the
        fit itself failed."""
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.5, maxIters=2000, confidence=0.99)
        if H_matrix is None:
            return None
        num_inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = num_inliers / max(1, len(good_matches))
        low_confidence_match = (num_inliers < self.cfg.HOMOGRAPHY_MIN_INLIERS) or (inlier_ratio < self.cfg.HOMOGRAPHY_MIN_INLIER_RATIO)
        if low_confidence_match:
            print(f"[WARP QUALITY] Low-confidence alignment: {num_inliers}/{len(good_matches)} valid matches ({inlier_ratio:.0%}).")
        return H_matrix, num_inliers, inlier_ratio, low_confidence_match

    def _load_and_retarget_json_payload(self, original_json_path, new_img_path, h2, w2):
        """Loads the source JSON and retargets its image metadata onto
        the new photo."""
        with open(original_json_path, 'r', encoding='utf-8') as f:
            json_payload = json.load(f)
        json_payload["imagePath"] = os.path.basename(new_img_path)
        json_payload["imageHeight"] = int(h2)
        json_payload["imageWidth"] = int(w2)
        with open(new_img_path, "rb") as b64_file:
            json_payload["imageData"] = base64.b64encode(b64_file.read()).decode('utf-8')
        return json_payload

    def _is_crack_shape_for_warp(self, shape):
        """Same label/shape_type check used for the plausibility guard
        below -- a crack has a comparable single "span" to check."""
        return (shape.get("label") in ("crack", "crepa")) or (shape.get("shape_type") in ("linestrip", "linestring"))

    def _warp_shape_points(self, old_points, H_matrix, w2, h2):
        """Perspective-warps a shape's points and clamps them to the
        new image's bounds."""
        new_points = []
        if old_points:
            pts_array = np.array([old_points], dtype=np.float32)
            warped_pts = cv2.perspectiveTransform(pts_array, H_matrix)
            squeezed_pts = np.squeeze(warped_pts, axis=0)
            if squeezed_pts.ndim == 1:
                squeezed_pts = np.array([squeezed_pts])
            for pt in squeezed_pts:
                clamped_x = max(0.0, min(float(pt[0]), float(w2 - 1)))
                clamped_y = max(0.0, min(float(pt[1]), float(h2 - 1)))
                new_points.append([clamped_x, clamped_y])
        return new_points

    def _is_crack_span_implausible(self, old_points, new_points, is_crack_shape, min_ratio, max_ratio):
        """Flags a crack whose warped length no longer plausibly
        matches its original -- a sign of a locally bad homography."""
        if not (is_crack_shape and len(old_points) >= 2 and len(new_points) >= 2):
            return False
        old_span = ((old_points[-1][0] - old_points[0][0]) ** 2 + (old_points[-1][1] - old_points[0][1]) ** 2) ** 0.5
        new_span = ((new_points[-1][0] - new_points[0][0]) ** 2 + (new_points[-1][1] - new_points[0][1]) ** 2) ** 0.5
        if old_span <= 1e-6:
            return False
        if new_span <= 1e-6:
            return True
        span_ratio = new_span / old_span
        return not (min_ratio <= span_ratio <= max_ratio)

    def _project_one_shape(self, shape, H_matrix, w2, h2, min_ratio, max_ratio):
        """Warps one shape, checks crack-span plausibility, and (if
        plausible) finalizes it. Returns (shape_or_None, was_rejected)."""
        old_points = shape.get("points", [])
        is_crack_shape = self._is_crack_shape_for_warp(shape)
        new_points = self._warp_shape_points(old_points, H_matrix, w2, h2)
        implausible = self._is_crack_span_implausible(old_points, new_points, is_crack_shape, min_ratio, max_ratio)
        if len(new_points) >= 2 and not implausible:
            shape["points"] = new_points
            if is_crack_shape and len(old_points) >= 2:
                shape["orig_points"] = [[float(pt[0]), float(pt[1])] for pt in old_points]
            shape.pop("width_segments", None)
            return shape, False
        return None, implausible

    def _project_all_shapes(self, shapes, H_matrix, w2, h2):
        """Projects every shape in the source JSON onto the new photo.
        Returns (projected_shapes, n_rejected_implausible)."""
        PLAUSIBLE_MIN_RATIO, PLAUSIBLE_MAX_RATIO = 0.30, 3.0
        projected_shapes = []
        n_rejected_implausible = 0
        for shape in shapes:
            result, implausible = self._project_one_shape(shape, H_matrix, w2, h2, PLAUSIBLE_MIN_RATIO, PLAUSIBLE_MAX_RATIO)
            if implausible:
                n_rejected_implausible += 1
            if result is not None:
                projected_shapes.append(result)
        return projected_shapes, n_rejected_implausible

    def _collect_current_session_shapes(self):
        """Builds shape dicts for cracks/detachments drawn in THIS
        session (not imported), to append alongside the projected ones."""
        current_workspace_shapes = []
        for f in self.cfg.saved_cracks:
            if f.get('active', True) and f.get('session_id') == 'current' and 'path' in f:
                cw_entry = {"label": "crack", "points": [[float(pt[0]), float(pt[1])] for pt in f['path']], "group_id": None, "shape_type": "linestrip", "flags": {}}
                if f.get('width_segments'):
                    cw_entry["width_segments"] = f['width_segments']
                current_workspace_shapes.append(cw_entry)
        for d in self.cfg.saved_detachments:
            if d.get('active', True) and d.get('session_id') == 'current' and 'path' in d:
                current_workspace_shapes.append({"label": "detachment", "points": [[float(pt[0]), float(pt[1])] for pt in d['path']], "group_id": None, "shape_type": "polygon", "flags": {}})
        return current_workspace_shapes

    def _report_warp_outcome(self, output_json_path, n_rejected_implausible, low_confidence_match, num_inliers, n_good_matches):
        """Implausible-crack rejections take priority over a
        low-confidence-match warning in the final status banner."""
        print(f"[SUCCESS] Sequential alignment completed for: {os.path.basename(output_json_path)}")
        self.cfg.warp_jitter_triggered = True
        if n_rejected_implausible > 0:
            plural = n_rejected_implausible > 1
            crack_word = "cracks" if plural else "crack"
            discarded_word = "discarded" if plural else "discarded"
            self._signal_import_warning(
                f"IMPORT: {n_rejected_implausible} {crack_word} {discarded_word} (implausible projection) -- "
                f"retrace them by hand, or try again with a different source photo."
            )
            print(f"[WARP PLAUSIBILITY] {n_rejected_implausible} crack(s) discarded: the projection altered "
                  f"their length beyond plausible limits (likely an imprecise local homography in that area).")
        elif low_confidence_match:
            self._signal_import_warning(f"IMPORT: low-confidence alignment ({num_inliers}/{n_good_matches} matches) - press [V] to retrace the cracks onto the real edge, or correct them individually with [T].")
        else:
            self.cfg.import_warning_triggered = False

    def warp_and_adapt_json_to_new_image(self, base_img_path, new_img_path, original_json_path, output_json_path):
        """Uses robust SIFT matching and precise perspective
        transformation to project shapes."""
        try:
            self.cfg.warp_jitter_triggered = False
            images = self._read_grayscale_and_color(base_img_path, new_img_path)
            if images is None:
                self._signal_import_warning("IMPORT FAILED: previous or current image is unreadable.")
                return False
            img1, img2_original, img2 = images
            h2, w2 = img2_original.shape[:2]

            sift_result = self._find_good_sift_matches(img1, img2)
            if sift_result is None:
                self._signal_import_warning("IMPORT FAILED: no recognizable detail in the images for alignment.")
                return False
            kp1, kp2, good_matches = sift_result
            if len(good_matches) < 8:
                self._signal_import_warning("IMPORT FAILED: view too different from the previous photo, not enough matches.")
                return False

            homography_result = self._compute_warp_homography(kp1, kp2, good_matches)
            if homography_result is None:
                self._signal_import_warning("IMPORT FAILED: view too different from the previous photo, alignment impossible.")
                return False
            H_matrix, num_inliers, inlier_ratio, low_confidence_match = homography_result

            json_payload = self._load_and_retarget_json_payload(original_json_path, new_img_path, h2, w2)
            projected_shapes, n_rejected_implausible = self._project_all_shapes(json_payload.get("shapes", []), H_matrix, w2, h2)
            json_payload["shapes"] = projected_shapes + self._collect_current_session_shapes()
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, ensure_ascii=False, indent=2)

            self._report_warp_outcome(output_json_path, n_rejected_implausible, low_confidence_match, num_inliers, len(good_matches))
            return True
        except Exception as error_msg:
            print(f"[WARP RUNTIME ERROR] Projection interrupted: {error_msg}")
            self._signal_import_warning(f"IMPORT FAILED: error during projection ({error_msg}).")
            return False

    def _resolve_missing_source_image(self, processed_folder):
        """The image no longer exists at its recorded path -- checks if it's already at the archive destination. Returns True if resolved."""
        check_dest = os.path.join(processed_folder, os.path.basename(self.cfg.CURRENT_IMAGE_PATH))
        if os.path.exists(check_dest):
            # Updates the pointer and falls through so the JSON still gets archived.
            self.cfg.CURRENT_IMAGE_PATH = check_dest
            return True
        print(f"[ARCHIVE WARNING] Cannot archive: the source file no longer exists: {self.cfg.CURRENT_IMAGE_PATH}")
        return False

    def _move_current_image_to_archive(self, processed_folder):
        """Safe move of the image file into the archive."""
        try:
            img_dest = os.path.join(processed_folder, os.path.basename(self.cfg.CURRENT_IMAGE_PATH))
            shutil.move(self.cfg.CURRENT_IMAGE_PATH, img_dest)
            print(f"[ARCHIVE SUCCESS] Image archived: {os.path.basename(img_dest)}")
            self.cfg.CURRENT_IMAGE_PATH = img_dest  # Update the global path with the new one
        except Exception as e:
            print(f"[ARCHIVE ERROR] Error while moving the image: {e}")

    def _move_current_json_to_archive(self, processed_folder):
        """Safe move of the associated JSON file into the archive."""
        if self.cfg.JSON_OUTPUT_PATH and os.path.exists(self.cfg.JSON_OUTPUT_PATH):
            try:
                json_dest = os.path.join(processed_folder, os.path.basename(self.cfg.JSON_OUTPUT_PATH))
                shutil.move(self.cfg.JSON_OUTPUT_PATH, json_dest)
                print(f"[ARCHIVE SUCCESS] Session JSON archived: {os.path.basename(json_dest)}")
                self.cfg.JSON_OUTPUT_PATH = json_dest
            except Exception as e:
                print(f"[ARCHIVE WARNING] Error while moving the JSON: {e}")

    def archive_current_session_to_processed(self):
        """ Safely moves the current image and its JSON file into the
        'already processed images' folder, avoiding file-not-found errors.
        """
        if self.cfg.CURRENT_IMAGE_PATH is None:
            return

        processed_folder = os.path.join(str(resolve_script_dir()), "already processed images")
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        # 1. Safety check: if the file is already inside the destination folder, do nothing
        if processed_folder in self.cfg.CURRENT_IMAGE_PATH:
            return

        # 2. Check whether the file still physically exists at the original path before moving it
        if not os.path.exists(self.cfg.CURRENT_IMAGE_PATH):
            if not self._resolve_missing_source_image(processed_folder):
                return
        else:
            # 3. Safe move of the image file
            self._move_current_image_to_archive(processed_folder)

        # 4. Safe move of the associated JSON file
        self._move_current_json_to_archive(processed_folder)

    def _parse_one_width_segment(self, seg):
        """Parses one width_segments entry defensively. None if the required i0/i1 fields are missing/invalid."""
        try:
            seg_entry = {
                'i0': int(seg['i0']), 'i1': int(seg['i1']), 'width_px': int(seg.get('width_px', 0))
            }
        except (KeyError, TypeError, ValueError):
            return None
        # margin_left_px/margin_right_px: absent on older files, defaulting to 0.
        try:
            seg_entry['margin_left_px'] = int(seg.get('margin_left_px', 0) or 0)
        except (TypeError, ValueError):
            seg_entry['margin_left_px'] = 0
        try:
            seg_entry['margin_right_px'] = int(seg.get('margin_right_px', 0) or 0)
        except (TypeError, ValueError):
            seg_entry['margin_right_px'] = 0
        fill_raw = seg.get('fill')
        if fill_raw:
            fill_clean = []
            for row in fill_raw:
                try:
                    fill_clean.append([int(row[0]), int(row[1]), int(row[2])])
                except (TypeError, ValueError, IndexError):
                    continue
            if fill_clean:
                seg_entry['fill'] = fill_clean
        return seg_entry

    def _load_width_segments(self, shape):
        """Restores every valid per-tract width override recorded by
        [A] (see width_edit_state/CRACK_MASK_DILATION_PX in config.py)."""
        width_segments = []
        for seg in (shape.get("width_segments") or []):
            seg_entry = self._parse_one_width_segment(seg)
            if seg_entry is not None:
                width_segments.append(seg_entry)
        return width_segments

    def _build_loaded_crack(self, shape, points):
        """Rebuilds one crack dict from its saved JSON shape."""
        # 'orig_points' (see warp_and_adapt_json_to_new_image) carries the
        # crack's original, unwarped shape, kept only so [V] can re-fit it.
        orig_pts_raw = shape.get("orig_points")
        orig_path = [(int(pt[0]), int(pt[1])) for pt in orig_pts_raw] if orig_pts_raw and len(orig_pts_raw) >= 2 else None
        width_segments = self._load_width_segments(shape)
        return {'start': points[0], 'end': points[-1], 'path': points, 'active': True,
                'session_id': 'imported', 'orig_path': orig_path, 'width_segments': width_segments}

    def _build_loaded_detachment(self, points):
        """Rebuilds one detachment dict from its saved JSON shape."""
        return {'start': points[0], 'nodes': points, 'path': points, 'active': True, 'session_id': 'imported'}

    def _load_one_shape(self, shape, loaded_cracks, loaded_detachments):
        """Parses one JSON shape and appends it to the right list,
        based on its label/shape_type."""
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "")
        points = [(int(pt[0]), int(pt[1])) for pt in shape.get("points", [])]
        if not points or len(points) < 2:
            return
        if label in ["crack", "crepa"] or shape_type in ["linestrip", "linestring"]:
            loaded_cracks.append(self._build_loaded_crack(shape, points))
        elif label in ["detachment", "distacco"] or shape_type == "polygon":
            loaded_detachments.append(self._build_loaded_detachment(points))

    def load_labelme_format(self):
        """Initializes dataset elements from JSON structures rapidly using saved nodes."""
        if not os.path.exists(self.cfg.JSON_OUTPUT_PATH):
            return
        try:
            with open(self.cfg.JSON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)

            self.cfg.PIXEL_TO_CM_SCALE = labelme_data.get("pixel_to_cm_scale", 0.05)
            loaded_cracks, loaded_detachments = [], []
            self.cfg.action_history.clear()
            self.cfg.redo_history.clear()

            for shape in labelme_data.get("shapes", []):
                self._load_one_shape(shape, loaded_cracks, loaded_detachments)

            self.cfg.saved_cracks = loaded_cracks
            self.cfg.saved_detachments = loaded_detachments
            self.recalculate_masks()
            print(f"[SUCCESS] Fast load restored: {len(self.cfg.saved_cracks)} cracks aligned.")
        except Exception as e:
            print(f"[IO ERROR] Rapid pre-load extraction aborted: {e}", file=sys.stderr)

    def _extract_building_code_from_filename(self, curr_filename):
        """Isolates the building tag (e.g. "BLDG001") from a filename's __TAG suffix. None if there's no tag or it can't be parsed."""
        if "__" not in curr_filename:
            print("[AUTO-ALIGN] The current image has no building code assigned. Waiting for the operator.")
            self._signal_import_warning("IMPORT NOT PERFORMED: no building code assigned (assign one with [B]).")
            return None
        try:
            bldg_part = curr_filename.split("__")[-1]
            curr_bldg_code = os.path.splitext(bldg_part)[0]
            if curr_bldg_code.endswith("-seg"):
                curr_bldg_code = curr_bldg_code[:-4]
            return curr_bldg_code
        except Exception:
            return None

    def _locate_associated_image(self, processed_folder, j_file, orig_img_name):
        """Finds an archived JSON's photo -- by its recorded imagePath first, then by guessing common extensions."""
        if orig_img_name:
            img_path_check = os.path.join(processed_folder, orig_img_name)
            if os.path.exists(img_path_check):
                return img_path_check
        candidate_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for ext in candidate_extensions:
            img_name_fallback = j_file.replace('.json', ext)
            img_path_check = os.path.join(processed_folder, img_name_fallback)
            if os.path.exists(img_path_check):
                return img_path_check
        return None

    def _score_one_candidate_session(self, processed_folder, j_file, curr_bldg_code):
        """Scores one archived JSON as a candidate previous session, or None if it doesn't belong to this group or can't be matched to a photo."""
        if f"__{curr_bldg_code}" not in j_file:
            return None
        j_path = os.path.join(processed_folder, j_file)
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            score = len(meta.get("shapes", []))
            img_ref_path = self._locate_associated_image(processed_folder, j_file, meta.get("imagePath", ""))
            if img_ref_path is None:
                return None
            return score, j_path, img_ref_path
        except Exception:
            return None

    def _find_best_matching_session(self, processed_folder, curr_bldg_code):
        """Scans every archived JSON for the same building group,
        keeping the one with the most shapes."""
        best_json_path, best_score, best_image_name = None, -1, None
        json_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.json')]
        for j_file in json_files:
            result = self._score_one_candidate_session(processed_folder, j_file, curr_bldg_code)
            if result is not None:
                score, j_path, img_ref_path = result
                if score > best_score:
                    best_score, best_json_path, best_image_name = score, j_path, img_ref_path
        return best_json_path, best_score, best_image_name

    def _apply_best_previous_session(self, curr_bldg_code, best_score, best_image_name, best_json_path):
        """Projects the best-matching previous session's fractures
        onto the current photo. Returns True on success."""
        print(f"\n[AUTO-ALIGN] Found a previous session for BUILDING [{curr_bldg_code}] ({best_score} fractures).")
        success = self.warp_and_adapt_json_to_new_image(
            best_image_name, self.cfg.CURRENT_IMAGE_PATH, best_json_path, self.cfg.JSON_OUTPUT_PATH
        )
        if success:
            self.load_labelme_format()
            self.cfg.warp_jitter_triggered = True
            print("[AUTO-ALIGN SUCCESS] Fractures successfully projected onto the new image.")
            return True
        return False

    def auto_import_best_previous_session(self):
        """ Searches the processed-files folder for files sharing the same
        building code (__CODE) as the current image, and imports the best one.
        """
        processed_folder = os.path.join(str(resolve_script_dir()), "already processed images")
        if not os.path.exists(processed_folder):
            return False

        curr_filename = os.path.basename(self.cfg.CURRENT_IMAGE_PATH)
        curr_bldg_code = self._extract_building_code_from_filename(curr_filename)
        if curr_bldg_code is None:
            return False

        best_json_path, best_score, best_image_name = self._find_best_matching_session(processed_folder, curr_bldg_code)

        # Runs the projection only if there's a valid match for the same
        # building with saved geometries.
        if best_json_path and best_score > 0:
            return self._apply_best_previous_session(curr_bldg_code, best_score, best_image_name, best_json_path)
        else:
            print(f"[AUTO-ALIGN] No previous data found for building [{curr_bldg_code}]. Blank canvas.")
            self._signal_import_warning(f"IMPORT NOT PERFORMED: no previous session found for group [{curr_bldg_code}].")
            return False

# SIFT feature cache: avoids re-analyzing the same representative photo
# (or re-extracting the target photo's own features) once per candidate group.

    def _resolve_sift_cache_key(self, img_path, max_dim):
        """Builds the (absolute path, mtime, max_dim) cache key. None
        if the file can't be stat'd."""
        try:
            abs_path = os.path.abspath(img_path)
            mtime = os.path.getmtime(abs_path)
        except OSError:
            return None
        return abs_path, mtime, max_dim

    def _downscale_for_sift(self, img, max_dim):
        """Downscales an image so its longer side is at most max_dim
        pixels, if it isn't already."""
        h, w = img.shape[:2]
        longer_side = max(h, w)
        if longer_side > max_dim:
            scale = max_dim / float(longer_side)
            return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        return img

    def _compute_sift_features(self, abs_path, max_dim):
        """Reads, downscales, and runs SIFT on one image. Returns (keypoints, descriptors), or None if the image is unreadable."""
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = self._downscale_for_sift(img, max_dim)
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
        return sift.detectAndCompute(img, None)

    def _get_cached_sift_features(self, img_path, max_dim=None):
        """Returns (keypoints, descriptors) for img_path, downscaled to at most max_dim pixels and cached by (path, mtime, max_dim). Returns (None, None) if the image can't be read.
        """
        if max_dim is None:
            max_dim = self.cfg.AUTO_GROUP_SIFT_MAX_DIM
        cache_key = self._resolve_sift_cache_key(img_path, max_dim)
        if cache_key is None:
            return None, None
        cached = self.cfg._SIFT_FEATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute_sift_features(cache_key[0], max_dim)
        if result is None:
            return None, None
        kp, des = result

        # Bounded by the number of DISTINCT (path, mtime) photos actually
        # touched during this run -- not unbounded growth within one execution.
        self.cfg._SIFT_FEATURE_CACHE[cache_key] = (kp, des)
        return kp, des

    def _resolve_progress_banner_window_size(self):
        """Same window-size resolution render_scene() uses, so this banner always fills the real window like every other frame."""
        window_rect = cv2.getWindowImageRect("Crack Detector Workspace")
        if window_rect is not None and len(window_rect) >= 4:
            _, _, w_win, h_win = window_rect
            if w_win <= 100 or h_win <= 100:
                return 1200, 900
            return w_win, h_win
        return 1200, 900

    def _draw_progress_banner_text(self, canvas, current_idx, total, group_label, w_win, h_win):
        """Draws the three status lines, scaled to the canvas size."""
        font_scale = max(0.45, min(1.4, h_win / 900.0))
        margin_x = max(20, int(w_win * 0.025))
        line_y = [int(h_win * 0.40), int(h_win * 0.47), int(h_win * 0.54)]

        cv2.putText(canvas, "Automatic building-group analysis in progress...",
                    (margin_x, line_y[0]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        detail = f"Comparing {current_idx} of {total} already-known groups"
        if group_label:
            detail += f" ({group_label})"
        cv2.putText(canvas, detail, (margin_x, line_y[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Please wait, the window is not frozen.", (margin_x, line_y[2]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.78, (150, 220, 150), 1, cv2.LINE_AA)

    def _show_auto_group_progress_banner(self, current_idx, total, group_label=None):
        """Paints a 'please wait' frame, sized to the real window, while auto_assign_building_group_by_similarity() works through its comparison loop, and pumps the display so the OS doesn't mark it unresponsive.
        """
        try:
            w_win, h_win = self._resolve_progress_banner_window_size()
            canvas = np.zeros((h_win, w_win, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)
            self._draw_progress_banner_text(canvas, current_idx, total, group_label, w_win, h_win)
            self._display_frame("Crack Detector Workspace", canvas)
            self._wait_key(1)
        except Exception:
            pass

    def _find_good_sift_matches_for_similarity(self, des_a, des_b):
        """FLANN-matches two descriptor sets, keeping matches passing the Lowe ratio test. None if the FLANN match fails."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = matcher.knnMatch(des_a, des_b, k=2)
        except cv2.error:
            return None

        good_matches = []
        for m_match in matches:
            if len(m_match) == 2:
                m, n = m_match[0], m_match[1]
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches

    def _compute_similarity_homography_inliers(self, kp_a, kp_b, good_matches):
        """RANSAC-fits a homography from the good matches. Returns
        (num_inliers, inlier_ratio), both 0 if the fit itself failed."""
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.5, maxIters=2000, confidence=0.99)
        if H_matrix is None:
            return 0, 0.0
        num_inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = num_inliers / max(1, len(good_matches))
        return num_inliers, inlier_ratio

    def _compute_image_similarity_score(self, img_path_a, img_path_b):
        """Compares two photos with SIFT+FLANN+RANSAC, returning {'good_matches', 'inliers', 'inlier_ratio'} describing how confidently they show the same structure. None if either image is unreadable.
        """
        kp_a, des_a = self._get_cached_sift_features(img_path_a)
        kp_b, des_b = self._get_cached_sift_features(img_path_b)
        if des_a is None or des_b is None:
            return None

        good_matches = self._find_good_sift_matches_for_similarity(des_a, des_b)
        if good_matches is None:
            return None

        if len(good_matches) < 4:
            # Not even enough points to attempt a homography (cv2.findHomography
            # needs at least 4 correspondences) -- clearly not the same photo.
            return {'good_matches': len(good_matches), 'inliers': 0, 'inlier_ratio': 0.0}

        num_inliers, inlier_ratio = self._compute_similarity_homography_inliers(kp_a, kp_b, good_matches)
        return {'good_matches': len(good_matches), 'inliers': num_inliers, 'inlier_ratio': inlier_ratio}

    def _parse_building_index_from_filename(self, fname, candidate_extensions):
        """Extracts the __BLDG index from a qualifying plain photo filename (not JSON, not a -seg overlay). None if it doesn't qualify."""
        if "__BLDG" not in fname:
            return None
        if not fname.lower().endswith(candidate_extensions):
            return None  # skip JSON/other non-image files carrying the same tag
        base_name_only, _ = os.path.splitext(fname)
        if base_name_only.endswith("-seg"):
            return None  # skip the saved colour overlay copy, keep the plain photo
        try:
            return int(base_name_only.split("__BLDG")[-1])
        except ValueError:
            return None

    def _scan_folder_for_building_groups(self, folder, groups, candidate_extensions):
        """Adds every new group found in one folder to `groups`, keeping the first representative image path found for each."""
        if not folder or not os.path.exists(folder):
            return
        for fname in os.listdir(folder):
            idx = self._parse_building_index_from_filename(fname, candidate_extensions)
            if idx is not None and idx not in groups:
                groups[idx] = os.path.join(folder, fname)

    def _list_existing_building_groups(self):
        """Scans IMAGE_FOLDER, OUTPUT_FOLDER, and 'already processed images' for files tagged __BLDG, returning {group_index: representative_image_path}, one photo per distinct group.
        """
        candidate_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        groups = {}
        processed_folder = os.path.join(str(resolve_script_dir()), "already processed images")
        for folder in (self.cfg.IMAGE_FOLDER, self.cfg.OUTPUT_FOLDER, processed_folder):
            self._scan_folder_for_building_groups(folder, groups, candidate_extensions)
        return groups

    def _find_best_matching_building_group(self, target_path, candidates):
        """Scores target_path against every existing group's photo, keeping the best confident match. None if nothing clears the threshold."""
        total_candidates = len(candidates)
        best_idx, best_inliers, best_matches_info = None, -1, None
        for i, (idx, rep_path) in enumerate(candidates.items(), start=1):
            if os.path.abspath(rep_path) == os.path.abspath(target_path):
                continue
            # See _show_auto_group_progress_banner() (05/09/2026, trentunesima
            # consegna): keeps the window visibly alive during this loop.
            self._show_auto_group_progress_banner(i, total_candidates, f"BLDG{idx:03d}")
            score = self._compute_image_similarity_score(rep_path, target_path)
            if score is None:
                continue
            if score['good_matches'] < self.cfg.BUILDING_GROUP_MATCH_MIN_GOOD_MATCHES:
                continue
            if score['inliers'] < self.cfg.BUILDING_GROUP_MATCH_MIN_INLIERS or score['inlier_ratio'] < self.cfg.BUILDING_GROUP_MATCH_MIN_INLIER_RATIO:
                continue
            if score['inliers'] > best_inliers:
                best_idx, best_inliers, best_matches_info = idx, score['inliers'], score
        return best_idx, best_inliers, best_matches_info

    def _assign_to_existing_group(self, best_idx, best_matches_info):
        """Tags the current photo with an already-known building
        group's code."""
        if self._apply_building_tag_to_current_file(best_idx):
            self.cfg.CURRENT_BUILDING_INDEX = max(best_idx, self.cfg.CURRENT_BUILDING_INDEX or 0)
            print(f"[AUTO-GROUP] Similar photo found (same structure, different perspective): "
                  f"assigned to the existing group BLDG{best_idx:03d} "
                  f"({best_matches_info['inliers']}/{best_matches_info['good_matches']} valid matches).")
        else:
            print(f"[AUTO-GROUP ERROR] Match found with BLDG{best_idx:03d} but renaming the file failed.")

    def _create_new_building_group(self, candidates):
        """No existing group matched confidently enough -- creates a new one, numbered one past the highest existing index."""
        existing_max = max(candidates.keys()) if candidates else 0
        next_index = max(existing_max, self.cfg.CURRENT_BUILDING_INDEX or 0) + 1
        if self._apply_building_tag_to_current_file(next_index):
            self.cfg.CURRENT_BUILDING_INDEX = next_index
            print(f"[AUTO-GROUP] No similar photo found among the {len(candidates)} already-known groups: "
                  f"created new group BLDG{next_index:03d}.")
        else:
            print(f"[AUTO-GROUP ERROR] Could not create/apply the new group BLDG{next_index:03d}.")

    def auto_assign_building_group_by_similarity(self, target_path):
        """Called automatically for every freshly opened (untagged) photo: compares it via SIFT+homography against one representative photo per known building group, tagging it with the best match or creating a new group.
        """
        if self.cfg.CURRENT_IMAGE_PATH is None or not os.path.exists(self.cfg.CURRENT_IMAGE_PATH):
            return
        if "__BLDG" in os.path.basename(self.cfg.CURRENT_IMAGE_PATH):
            return  # already tagged (e.g. re-opened mid-session) -- nothing to do

        candidates = self._list_existing_building_groups()
        best_idx, best_inliers, best_matches_info = self._find_best_matching_building_group(target_path, candidates)

        if best_idx is not None:
            self._assign_to_existing_group(best_idx, best_matches_info)
            return

        self._create_new_building_group(candidates)


    def _load_current_image_bytes_and_decode(self, target_path):
        """Sets the session's path bookkeeping, reads and decodes the image; raises if OpenCV can't read the file."""
        self.cfg.CURRENT_IMAGE_PATH = target_path
        if target_path not in self.cfg.historical_processed_queue:
            self.cfg.historical_processed_queue.append(target_path)

        base_name = os.path.basename(self.cfg.CURRENT_IMAGE_PATH)
        img_name, img_ext = os.path.splitext(base_name)
        self.cfg.JSON_OUTPUT_PATH = os.path.join(self.cfg.OUTPUT_FOLDER, f"{img_name}.json")
        self.cfg.EXPORT_IMAGE_PATH = os.path.join(self.cfg.OUTPUT_FOLDER, f"{img_name}-seg{img_ext}")

        with open(self.cfg.CURRENT_IMAGE_PATH, "rb") as _raw_file:
            self.cfg.CURRENT_IMAGE_RAW_BYTES = _raw_file.read()
        self.cfg.img_original = cv2.imdecode(np.frombuffer(self.cfg.CURRENT_IMAGE_RAW_BYTES, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.cfg.img_original is None:
            raise ValueError(f"OpenCV failed loading file target: {self.cfg.CURRENT_IMAGE_PATH}")
        self.cfg.H_img, self.cfg.W_img, _ = self.cfg.img_original.shape

    def _reset_zoom_state(self):
        self.cfg.zoom_factor = 1.0
        self.cfg.zoom_center = [int(self.cfg.W_img // 2), int(self.cfg.H_img // 2)]
        self.cfg.zoom_box = [0, 0, self.cfg.W_img, self.cfg.H_img]

    def _reset_session_editing_state(self):
        """Clears every in-progress interaction/editing state for a fresh image."""
        self.cfg.saved_cracks, self.cfg.saved_detachments = [], []
        self.cfg.action_history, self.cfg.redo_history = [], []
        self.cfg.temp_start, self.cfg.temp_path, self.cfg.temp_nodes = None, [], []
        self.cfg.translate_state["active"], self.cfg.translate_state["start"], self.cfg.translate_state["idx"] = False, None, None
        self.cfg.width_edit_state["active"] = False
        self.cfg.width_edit_state["pending_crack_idx"] = None
        self.cfg.width_edit_state["pending_i0"] = None
        self.cfg.width_edit_state["focused_crack_idx"] = None
        self.cfg.width_edit_state["focused_seg_idx"] = None
        self.cfg.manual_group_entry_state["active"] = False
        self.cfg.manual_group_entry_state["buffer"] = ""
        self.cfg.PIXEL_TO_CM_SCALE = 0.05
        self.cfg.calib_start, self.cfg.calibration_mode = None, False

    def _compute_binary_and_skeleton_masks(self):
        """Rebuilds the adaptive-threshold binary mask and its skeleton
        for the new image, plus fresh (empty) visual overlay masks."""
        gray = cv2.cvtColor(self.cfg.img_original, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.cfg.binary_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        self.cfg.skeleton_mask = (skeletonize(self.cfg.binary_mask > 0) * 255).astype(np.uint8)
        self.cfg.blue_visual_mask = np.zeros((self.cfg.H_img, self.cfg.W_img), dtype=np.uint8)
        self.cfg.green_visual_mask = np.zeros((self.cfg.H_img, self.cfg.W_img), dtype=np.uint8)

    def _build_display_background(self):
        """Builds the background image render_scene() paints onto -- contrast-enhanced (CLAHE) if enabled, otherwise the plain original."""
        if self.cfg.CONTRAST_ENHANCEMENT:
            lab = cv2.cvtColor(self.cfg.img_original, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            self.cfg.img_background = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2BGR)
        else:
            self.cfg.img_background = self.cfg.img_original.copy()

    def _load_or_initialize_shapes_for_mode(self):
        """Mode 2: loads the existing JSON. Mode 1: starts blank and
        auto-assigns a building group by photo similarity."""
        if self.cfg.modalita_scelta == "2":
            self.load_labelme_format()
        else:
            print("[INFO] Blank canvas initialized. Press 'W' (or 'L') to project/align previous fractures.")
            self.auto_assign_building_group_by_similarity(self.cfg.CURRENT_IMAGE_PATH)

    def initialize_image_session(self, target_path, mode="1"):
        """Resets core memory allocations and binds new image matrix arrays to canvas workspace.
        Importing previous fractures is now EXCLUSIVELY manual."""
        self.cfg.SESSION_MODE = str(mode)
        self._load_current_image_bytes_and_decode(target_path)
        self._reset_zoom_state()
        self._reset_session_editing_state()
        self._compute_binary_and_skeleton_masks()
        self._build_display_background()
        self._load_or_initialize_shapes_for_mode()
        # NO AUTOMATIC LOADING HERE.
        # The operator decides when (and whether) to press W.

    def manual_trigger_auto_align(self):
        """Lets the operator manually load the fractures from the previous image
        of the same building by pressing the L key.
        """
        print("\n[MANUAL ALIGN] Starting alignment at the user's request...")
    
        # Calls the script's native SIFT function to look for previous sessions of the same BLDG
        try:
            # Note: make sure this is the exact name of your original SIFT function in the code
            self.auto_import_best_previous_session() 
            print("[MANUAL ALIGN SUCCESS] Alignment completed successfully.")
        except NameError:
            # If the function has a slightly different name in your original script, e.g. without '_import'
            try:
                self.auto_import_best_previous_session()
                print("[MANUAL ALIGN SUCCESS] Alignment completed successfully.")
            except Exception as e:
                print(f"[MANUAL ALIGN ERROR] Could not align: {e}")
        except Exception as e:
            print(f"[MANUAL ALIGN ERROR] Generic error during alignment: {e}")

    def _scroll_help_menu_if_open(self, key, key_clean):
        """[Up]/[Down] scroll the help guide's command list while it's open. Returns True if handled (caller should stop processing)."""
        if not self.cfg.show_help_menu:
            return False
        is_up = (key_clean == 82 or key == 82 or key == 0 or key == 63232)
        is_down = (key_clean == 84 or key == 84 or key == 1 or key == 63233)
        if not (is_up or is_down):
            return False
        self.cfg.help_scroll_offset = max(0, self.cfg.help_scroll_offset + (-1 if is_up else 1))
        self.refresh_zoom_viewport()
        return True

    def _toggle_help_menu(self):
        """[?]: shows/hides the on-screen command guide, always reopening
        scrolled to the top."""
        self.cfg.show_help_menu = not self.cfg.show_help_menu
        self.cfg.help_scroll_offset = 0
        self.refresh_zoom_viewport()

    def _reset_width_edit_focused_tract(self):
        """[Backspace]/[Delete]: while a width-edit tract is focused, removes just that one override. Returns True if a tract was focused (and thus handled here)."""
        if not (self.cfg.width_edit_state["active"] and self.cfg.width_edit_state["focused_crack_idx"] is not None):
            return False
        fc = self.cfg.width_edit_state["focused_crack_idx"]
        fs = self.cfg.width_edit_state["focused_seg_idx"]
        if 0 <= fc < len(self.cfg.saved_cracks):
            segs = self.cfg.saved_cracks[fc].get('width_segments', [])
            if fs is not None and 0 <= fs < len(segs):
                segs.pop(fs)
                print(" [WIDTH] Tract removed.")
        self.cfg.width_edit_state["focused_crack_idx"] = None
        self.cfg.width_edit_state["focused_seg_idx"] = None
        self.refresh_zoom_viewport()
        return True

    def _emergency_unlock(self):
        """[Backspace]/[Delete]: clears every in-progress interaction mode back to a clean slate -- the panic button for a "stuck" tool."""
        print("\n [EMERGENCY UNLOCK] Unlocking system and resetting temporary variables...")
        self._reset_translate_state()
        self._reset_width_edit_state()
        self.cfg.temp_start = None
        self.cfg.temp_path.clear()
        self.cfg.temp_nodes.clear()
        self.cfg.user_clicked_nodes.clear()
        self.cfg.calib_start = None
        self.cfg.calibration_mode = False
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _reset_translate_state(self):
        """Clears the [T] automatic-snap tool's in-progress state."""
        self.cfg.translate_state["active"], self.cfg.translate_state["start"], self.cfg.translate_state["idx"] = False, None, None

    def _reset_width_edit_state(self):
        """Clears the [A] crack-width tool's in-progress state."""
        self.cfg.width_edit_state["active"] = False
        self.cfg.width_edit_state["pending_crack_idx"] = None
        self.cfg.width_edit_state["pending_i0"] = None
        self.cfg.width_edit_state["focused_crack_idx"] = None
        self.cfg.width_edit_state["focused_seg_idx"] = None

    def _toggle_fullscreen(self):
        """[F]: toggles the real OpenCV window between fullscreen and its
        normal 1200x900 size."""
        self.cfg.is_fullscreen = not self.cfg.is_fullscreen
        if self.cfg.is_fullscreen:
            cv2.setWindowProperty("Crack Detector Workspace", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Crack Detector Workspace", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crack Detector Workspace", 1200, 900)
        self.refresh_zoom_viewport()

    def _clear_all(self):
        """[X]: full reset of the canvas AND the exported JSON. Itself undoable: [U] restores everything cleared, [R] re-clears it."""
        self._reset_translate_state()
        self._reset_width_edit_state()
        backup = {'cracks': list(self.cfg.saved_cracks), 'detachments': list(self.cfg.saved_detachments)}
        self.cfg.redo_history.clear()
        self.cfg.action_history.append(('clear_all', backup))
        self.cfg.saved_cracks.clear()
        self.cfg.saved_detachments.clear()
        self.cfg.temp_start, self.cfg.temp_path, self.cfg.temp_nodes = None, [], []
        self.cfg.user_clicked_nodes.clear()
        self.recalculate_masks()
        self.refresh_zoom_viewport()
        self._reset_json_shapes_on_disk()

    def _reset_json_shapes_on_disk(self):
        """[X]: empties the "shapes" list in the already-exported JSON too, so Clear All resets the file on disk, not just the canvas."""
        if not os.path.exists(self.cfg.JSON_OUTPUT_PATH):
            print(" [CLEAR ALL] Canvas cleared.")
            return
        try:
            with open(self.cfg.JSON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            labelme_data["shapes"] = []
            with open(self.cfg.JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
            print(f" [CLEAR ALL] Canvas cleared and JSON reset.")
        except Exception as e:
            print(f" [CLEAR ERROR] Error: {e}")

    def _get_focused_width_segment(self):
        """Returns (segment_dict, crack_index) for the currently width-
        edit-focused tract, or (None, None) if nothing valid is focused."""
        fc = self.cfg.width_edit_state["focused_crack_idx"]
        fs = self.cfg.width_edit_state["focused_seg_idx"]
        if not (0 <= fc < len(self.cfg.saved_cracks)):
            return None, None
        segs = self.cfg.saved_cracks[fc].get('width_segments', [])
        if fs is None or not (0 <= fs < len(segs)):
            return None, None
        return segs[fs], fc

    def _adjust_fill_width(self, seg, fc, delta):
        """Refines an auto-filled tract's detected width by one pixel, signed since it's relative to the detected fill, not an absolute size."""
        if delta > 0:
            seg['width_px'] = min(self.cfg.WIDTH_EDIT_MAX_PX, seg['width_px'] + 1)
        else:
            seg['width_px'] = max(-self.cfg.WIDTH_EDIT_MAX_PX, seg['width_px'] - 1)
        print(f" [WIDTH] Crack tract #{fc}: refinement {seg['width_px']:+d}px "
              f"relative to the automatically detected fill.")

    def _adjust_legacy_width(self, seg, fc, delta):
        """Widens/narrows a legacy (non-fill) tract's symmetric dilation by
        one pixel per side, floored at zero."""
        if delta > 0:
            seg['width_px'] = min(self.cfg.WIDTH_EDIT_MAX_PX, seg['width_px'] + 1)
        else:
            seg['width_px'] = max(0, seg['width_px'] - 1)
        print(f" [WIDTH] Crack tract #{fc}: {seg['width_px']}px per side "
              f"(~{2 * seg['width_px'] + 1}px total).")

    def _width_edit_adjust_symmetric(self, delta):
        """[+]/[-] while a width-edit tract is focused: adjusts that
        tract's width by one pixel (delta=+1 widens, -1 narrows)."""
        seg, fc = self._get_focused_width_segment()
        if seg is not None:
            if seg.get('fill'):
                self._adjust_fill_width(seg, fc, delta)
            else:
                self._adjust_legacy_width(seg, fc, delta)
        self.refresh_zoom_viewport()

    def _apply_directional_width_margin(self, seg, is_widen_left, is_widen_right, is_narrow_left, is_narrow_right):
        """[ / { / ] / } while a tract is focused: nudges one side's margin
        by one pixel, independently of the other side."""
        seg['margin_left_px'] = int(seg.get('margin_left_px', 0) or 0)
        seg['margin_right_px'] = int(seg.get('margin_right_px', 0) or 0)
        if is_widen_left:
            seg['margin_left_px'] = min(self.cfg.WIDTH_EDIT_MAX_PX, seg['margin_left_px'] + 1)
        elif is_narrow_left:
            seg['margin_left_px'] = max(-self.cfg.WIDTH_EDIT_MAX_PX, seg['margin_left_px'] - 1)
        elif is_widen_right:
            seg['margin_right_px'] = min(self.cfg.WIDTH_EDIT_MAX_PX, seg['margin_right_px'] + 1)
        elif is_narrow_right:
            seg['margin_right_px'] = max(-self.cfg.WIDTH_EDIT_MAX_PX, seg['margin_right_px'] - 1)

    def _print_directional_width_status(self, fc, seg):
        print(f" [WIDTH] Crack tract #{fc}: per-side refinement -- "
              f"left ([ / {{) {seg['margin_left_px']:+d}px, "
              f"right (] / }}) {seg['margin_right_px']:+d}px "
              f"(in addition to [+]/[-]: {seg.get('width_px', 0):+d}px on both sides).")

    def _width_edit_adjust_directional(self, is_widen_left, is_widen_right, is_narrow_left, is_narrow_right):
        """[ / { / ] / } while a width-edit tract is focused: independent
        per-side refinement, on top of the symmetric [+]/[-] above."""
        seg, fc = self._get_focused_width_segment()
        if seg is not None:
            self._apply_directional_width_margin(seg, is_widen_left, is_widen_right, is_narrow_left, is_narrow_right)
            self._print_directional_width_status(fc, seg)
        self.refresh_zoom_viewport()

    def _zoom_in(self):
        self.cfg.zoom_factor = min(10.0, self.cfg.zoom_factor + 0.5)
        self.refresh_zoom_viewport()

    def _zoom_out(self):
        self.cfg.zoom_factor = max(1.0, self.cfg.zoom_factor - 0.5)
        self.refresh_zoom_viewport()

    def _zoom_reset(self):
        self.cfg.zoom_factor = 1.0
        self.cfg.zoom_center = [int(self.cfg.W_img // 2), int(self.cfg.H_img // 2)]
        self.refresh_zoom_viewport()

    def _toggle_hud_size(self):
        self.cfg.hud_large_size = not self.cfg.hud_large_size
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _toggle_markers(self):
        self.cfg.show_markers = not self.cfg.show_markers
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _toggle_crack_overlay(self):
        """[N]: hides/shows just the blue crack fill, independent of [Space]/show_markers (endpoint markers and detachment outlines)."""
        self.cfg.show_crack_overlay = not self.cfg.show_crack_overlay
        stato = "visible" if self.cfg.show_crack_overlay else "hidden"
        print(f" [CRACK OVERLAY] Blue crack fill is now {stato}.")
        self.refresh_zoom_viewport()

    def _toggle_info_overlay(self):
        """[J]: shows/hides the FILE/Cracks Length/BUILDING status text --
        see show_info_overlay in config.py."""
        self.cfg.show_info_overlay = not self.cfg.show_info_overlay
        stato = "visible" if self.cfg.show_info_overlay else "hidden"
        print(f" [INFO OVERLAY] Status text is now {stato}.")
        self.refresh_zoom_viewport()

    def _select_crack_tool(self):
        """[C]: switches to the crack tool, cleanly leaving any in-progress T/A mode."""
        self._reset_translate_state()
        self._reset_width_edit_state()
        self.cfg.current_tool, self.cfg.temp_nodes, self.cfg.temp_path = 'crack', [], []

    def _select_detachment_tool(self):
        """[D]: switches to the detachment tool, same cleanup as [C]."""
        self._reset_translate_state()
        self._reset_width_edit_state()
        self.cfg.current_tool, self.cfg.temp_start = 'detachment', None

    def _toggle_save_seg_image(self):
        self.cfg.SAVE_SEG_IMAGE = not self.cfg.SAVE_SEG_IMAGE

    def _toggle_timer_pause(self):
        self.cfg.timer_is_paused = not self.cfg.timer_is_paused

    def _prompt_and_set_pixel_scale(self):
        val = self._prompt_pixel_scale(self.cfg.PIXEL_TO_CM_SCALE)
        if val is not None and val > 0:
            self.cfg.PIXEL_TO_CM_SCALE = val

    def _toggle_calibration_mode(self):
        self.cfg.calibration_mode = not self.cfg.calibration_mode
        self.cfg.calib_start, self.cfg.calib_current_mouse = None, None

    def _toggle_edit_mode(self):
        self.cfg.edit_mode = not self.cfg.edit_mode
        self.cfg.selected_edit_poly = None
        self.cfg.selected_edit_idx = None
        self.recalculate_masks()
        self.refresh_zoom_viewport()
    def _toggle_translate_mode(self):
        """[T]: toggles the automatic-snap tool on/off. Mutually exclusive with width-edit mode (KEY A): only one click-priority mode may be active at a time."""
        self.cfg.translate_state["active"] = not self.cfg.translate_state["active"]
        self.cfg.translate_state["start"] = None
        self.cfg.translate_state["idx"] = None

        if self.cfg.translate_state["active"]:
            # --- DISABLE OTHER OPEN MODES ---
            # Replace these names with your actual segmentation global variables!
            if 'segmentation_mode' in globals(): segmentation_mode = False
            if 'drawing_mode' in globals(): drawing_mode = False
            if 'current_poly_points' in globals(): current_poly_points = []  # Clears any leftover orange squares
            self._reset_width_edit_state()
            print(" [MODE] Automatic Snap Mode ACTIVE. Other modes disabled. Correct multiple points in sequence, press [T] again to exit.")
        else:
            print(" [MODE] Automatic Snap Mode disabled.")

    def _toggle_width_edit_mode(self):
        """[A]: toggles crack-width-tract edit mode on/off. Mutually exclusive with translate mode (KEY T)."""
        self.cfg.width_edit_state["active"] = not self.cfg.width_edit_state["active"]
        self.cfg.width_edit_state["pending_crack_idx"] = None
        self.cfg.width_edit_state["pending_i0"] = None
        self.cfg.width_edit_state["focused_crack_idx"] = None
        self.cfg.width_edit_state["focused_seg_idx"] = None

        if self.cfg.width_edit_state["active"]:
            self._reset_translate_state()
            print(" [MODE] CRACK WIDTH Mode active. Click the STARTING point of the tract to widen, "
                  "then the ENDING point. Use [+]/[-] to adjust the width, [BACKSPACE] to remove the "
                  "active tract, [A] to exit.")
        else:
            print(" [MODE] CRACK WIDTH Mode disabled.")
        self.refresh_zoom_viewport()

    def _is_plausible_ratio(self, ratio, min_ratio, max_ratio):
        """Shared plausibility check used both before and after re-tracing
        a single crack in [V]'s bulk retrace (see _retrace_one_crack)."""
        return min_ratio <= ratio <= max_ratio

    def _retrace_one_crack(self, f, min_ratio, max_ratio):
        """Attempts to re-trace one imported crack onto the real edge via A*, guarded by plausibility checks. Returns a backup dict for Undo if retraced, or None if left untouched."""
        orig = f.get('orig_path')
        if not orig or len(orig) < 2 or 'start' not in f or 'end' not in f:
            return None

        old_start, old_end = orig[0], orig[-1]
        old_span = ((old_end[0] - old_start[0]) ** 2 + (old_end[1] - old_start[1]) ** 2) ** 0.5
        cur_start, cur_end = tuple(f['start']), tuple(f['end'])
        new_span = ((cur_end[0] - cur_start[0]) ** 2 + (cur_end[1] - cur_start[1]) ** 2) ** 0.5
        if old_span < 1e-6 or new_span < 1e-6:
            return None
        # This crack's current endpoint estimate is untrustworthy on its
        # own -- leave it untouched rather than trust a bad estimate.
        if not self._is_plausible_ratio(new_span / old_span, min_ratio, max_ratio):
            return None

        nuovo_path = self.a_star_pathfinding(cur_start, cur_end, 'crack')
        if not nuovo_path:
            return None

        original_length = self._path_length(orig)
        new_length = self._path_length(nuovo_path)
        # A* may have locked onto a different, nearby feature -- reject an implausible length.
        if original_length < 1e-6 or not self._is_plausible_ratio(new_length / original_length, min_ratio, max_ratio):
            return None

        backup = {'start': f['start'], 'end': f['end'], 'path': f['path']}
        f['path'] = nuovo_path
        f['start'] = list(nuovo_path[0])
        f['end'] = list(nuovo_path[-1])
        return backup

    def _print_retrace_summary(self, n_ok, n_skip):
        if n_ok == 0 and n_skip == 0:
            print(" [RETRACE ONTO REAL EDGE] No imported cracks to retrace.")
        else:
            print(f" [RETRACE ONTO REAL EDGE] {n_ok} crack(s) retraced onto the photo's real edge; "
                  f"{n_skip} left unchanged (no credible edge nearby, or implausible position/length) "
                  f"-- correct them individually with [T].")

    def _bulk_retrace_imported_cracks(self):
        """[V]: bulk re-traces every imported crack onto the real edge via A*, rejecting (leaving unchanged) any result that fails a plausibility check against its original shape."""
        PLAUSIBLE_MIN_RATIO, PLAUSIBLE_MAX_RATIO = 0.30, 3.0
        n_ok, n_skip = 0, 0
        # Backups collected here let one Undo revert the whole batch together.
        retrace_backups = []
        for idx, f in enumerate(self.cfg.saved_cracks):
            if not f.get('active', True):
                continue
            backup = self._retrace_one_crack(f, PLAUSIBLE_MIN_RATIO, PLAUSIBLE_MAX_RATIO)
            if backup is None:
                n_skip += 1
            else:
                backup['index'] = idx
                retrace_backups.append(backup)
                n_ok += 1

        if retrace_backups:
            self.cfg.action_history.append(('crack_bulk_retrace', retrace_backups))
        self.recalculate_masks()
        self.refresh_zoom_viewport()
        self._print_retrace_summary(n_ok, n_skip)

    def _close_polygon_final_segment(self):
        """[Y]: traces the final A* segment closing the polygon back to its first point, appending it to temp_nodes."""
        ultimo_punto = self.cfg.temp_nodes[-1]
        primo_punto = self.cfg.temp_nodes[0]
        chiusura_segmento = self.a_star_pathfinding(ultimo_punto, primo_punto, 'detachment')
        if chiusura_segmento:
            self.cfg.temp_nodes.extend(chiusura_segmento[1:])
        else:
            self.cfg.temp_nodes.append(primo_punto)

    def _save_completed_detachment(self):
        """Part of [Y]/close-detachment: commits the just-closed polygon
        (in temp_nodes) to saved_detachments."""
        new_detachment = {
            'path': list(self.cfg.temp_nodes),
            'click_nodes': list(self.cfg.user_clicked_nodes),
            'active': True,
            'session_id': str(time.time())
        }
        self.cfg.saved_detachments.append(new_detachment)
        self.cfg.action_history.append('detachment')

    def _close_detachment_polygon(self):
        """[Y]: closes the in-progress detachment polygon (needs >=3 clicks), saves it, and clears the in-progress state."""
        if not (self.cfg.current_tool == 'detachment' and len(self.cfg.temp_nodes) >= 3):
            return
        self._close_polygon_final_segment()
        self._save_completed_detachment()
        self.cfg.temp_nodes.clear()
        self.cfg.temp_path.clear()
        self.cfg.user_clicked_nodes.clear()
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _undo_in_progress_detachment_node(self):
        """[U]: while clicking a detachment's nodes, removes just the last click. Returns True if this applied."""
        if not (self.cfg.current_tool == 'detachment' and self.cfg.temp_nodes):
            return False
        self.cfg.temp_nodes.pop()
        self.cfg.temp_path = list(self.cfg.temp_nodes)
        if self.cfg.user_clicked_nodes:
            self.cfg.user_clicked_nodes.pop()
        return True

    def _undo_in_progress_crack_trace(self):
        """[U]: while a crack's start point is clicked but not completed, cancels that in-progress trace. Returns True if this applied."""
        if not (self.cfg.current_tool == 'crack' and self.cfg.temp_start is not None):
            return False
        self.cfg.temp_start, self.cfg.temp_path = None, []
        return True

    def _undo_crack_extension(self, backup):
        """Reverts a single crack's path/start/end to the given backup, shared by [T]/[V] undo (both modify an existing crack in place)."""
        idx = backup['index']
        if 0 <= idx < len(self.cfg.saved_cracks):
            self.cfg.saved_cracks[idx]['start'] = backup['start']
            self.cfg.saved_cracks[idx]['end'] = backup['end']
            self.cfg.saved_cracks[idx]['path'] = backup['path']

    def _undo_crack_bulk_retrace(self, backups):
        for backup in backups:
            self._undo_crack_extension(backup)

    def _undo_crack_delete(self, removed_crack):
        """Restores a crack deleted by clicking near its marker, queuing a matching redo entry so [R] re-deletes the same crack."""
        self.cfg.saved_cracks.append(removed_crack)
        self.cfg.redo_history.append(('crack_redelete', removed_crack))

    def _undo_detachment_delete(self, removed_detachment):
        """Mirrors _undo_crack_delete() for detachments -- clicking near a
        detachment's first node deletes it the same way."""
        self.cfg.saved_detachments.append(removed_detachment)
        self.cfg.redo_history.append(('detachment_redelete', removed_detachment))

    def _undo_new_crack(self):
        """Reverts a just-drawn crack ('crack' in action_history): pops it
        off saved_cracks and queues it for [R]/Redo to restore."""
        if self.cfg.saved_cracks and self.cfg.saved_cracks[-1].get('session_id') != 'imported':
            self.cfg.redo_history.append(('crack_new', self.cfg.saved_cracks.pop()))

    def _undo_new_detachment(self):
        if self.cfg.saved_detachments and self.cfg.saved_detachments[-1].get('session_id') != 'imported':
            self.cfg.redo_history.append(('detachment', self.cfg.saved_detachments.pop()))

    def _undo_clear_all(self, backup):
        """Restores everything a [X] Clear All removed, and queues a matching redo entry so [R] can re-clear."""
        self.cfg.saved_cracks[:] = backup['cracks']
        self.cfg.saved_detachments[:] = backup['detachments']
        self.cfg.redo_history.append(('clear_all', backup))
        print("[UNDO] Canvas restored after Clear All.")

    def _undo_from_history(self, last_action):
        """Dispatches one popped action_history entry to the right undo
        handler based on its tag."""
        if isinstance(last_action, tuple) and len(last_action) > 0 and last_action[0] == 'crack_extension':
            self._undo_crack_extension(last_action[1])
        elif isinstance(last_action, tuple) and len(last_action) > 0 and last_action[0] == 'crack_bulk_retrace':
            self._undo_crack_bulk_retrace(last_action[1])
        elif isinstance(last_action, tuple) and len(last_action) > 0 and last_action[0] == 'crack_delete':
            self._undo_crack_delete(last_action[1])
        elif isinstance(last_action, tuple) and len(last_action) > 0 and last_action[0] == 'detachment_delete':
            self._undo_detachment_delete(last_action[1])
        elif isinstance(last_action, tuple) and len(last_action) > 0 and last_action[0] == 'clear_all':
            self._undo_clear_all(last_action[1])
        elif last_action == 'crack':
            self._undo_new_crack()
        elif last_action == 'detachment':
            self._undo_new_detachment()

    def _undo_last_action(self):
        """[U]: the single Undo entry point -- cancels an in-progress trace, or reverses the last completed action_history entry."""
        if self._undo_in_progress_detachment_node():
            pass
        elif self._undo_in_progress_crack_trace():
            pass
        elif self.cfg.action_history:
            self._undo_from_history(self.cfg.action_history.pop())
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _dead_redo_branch(self):
        """Unreachable in normal use: 'R' is already intercepted earlier by process_keypress(). Kept for safety in case that ever changes."""
        if self.cfg.redo_history:
            action_type, restored = self.cfg.redo_history.pop()
            if action_type == 'crack_new':
                self.cfg.saved_cracks.append(restored)
                self.cfg.action_history.append('crack')
            elif action_type == 'detachment':
                self.cfg.saved_detachments.append(restored)
                self.cfg.action_history.append('detachment')
            self.recalculate_masks()
            self.refresh_zoom_viewport()

    def _dispatch_window_and_reset_keys(self, key_clean):
        """[Backspace]/[Delete], [F], [X] -- system-level resets."""
        if key_clean in [8, 127]:
            if not self._reset_width_edit_focused_tract():
                self._emergency_unlock()
        elif key_clean in [ord('f'), ord('F')]:
            self._toggle_fullscreen()
        elif key_clean in [ord('x'), ord('X')]:
            self._clear_all()
        else:
            return False
        return True

    def _dispatch_width_edit_keys(self, key_clean, is_plus_key, is_minus_key,
                                   is_widen_left_key, is_widen_right_key,
                                   is_narrow_left_key, is_narrow_right_key):
        """[+]/[-]/[ /{ /] /} -- only while a width-edit tract is
        focused, ahead of their normal zoom meaning below."""
        focused = self.cfg.width_edit_state["active"] and self.cfg.width_edit_state["focused_crack_idx"] is not None
        if not focused:
            return False
        if is_plus_key:
            self._width_edit_adjust_symmetric(+1)
        elif is_minus_key:
            self._width_edit_adjust_symmetric(-1)
        elif is_widen_left_key or is_widen_right_key or is_narrow_left_key or is_narrow_right_key:
            self._width_edit_adjust_directional(is_widen_left_key, is_widen_right_key, is_narrow_left_key, is_narrow_right_key)
        else:
            return False
        return True

    def _dispatch_view_toggle_keys(self, key_clean, is_plus_key, is_minus_key):
        """Zoom, HUD size, markers, blue/info overlays -- plain view
        toggles."""
        if is_plus_key:
            self._zoom_in()
        elif is_minus_key:
            self._zoom_out()
        elif key_clean in [ord('0'), ord('z'), ord('Z'), 48]:
            self._zoom_reset()
        elif key_clean in [ord('h'), ord('H')]:
            self._toggle_hud_size()
        elif key_clean == ord(' '):
            self._toggle_markers()
        elif key_clean in [ord('n'), ord('N')]:
            self._toggle_crack_overlay()
        elif key_clean in [ord('j'), ord('J')]:
            self._toggle_info_overlay()
        else:
            return False
        return True

    def _dispatch_tool_and_misc_keys(self, key_clean):
        """[C]/[D] tool selection plus the single-purpose M/P/K/L keys."""
        if key_clean in [ord('c'), ord('C')]:
            self._select_crack_tool()
        elif key_clean in [ord('d'), ord('D')]:
            self._select_detachment_tool()
        elif key_clean in [ord('m'), ord('M')]:
            self._toggle_save_seg_image()
        elif key_clean in [ord('p'), ord('P')]:
            self._toggle_timer_pause()
        elif key_clean in [ord('k'), ord('K')]:
            self._prompt_and_set_pixel_scale()
        elif key_clean in [ord('l'), ord('L')]:
            self._toggle_calibration_mode()
        else:
            return False
        return True

    def _dispatch_action_mode_keys(self, key_clean):
        """[E]/[T]/[A]/[V]/[G]/[Y]/[U]/[R] -- edit modes and crack/
        detachment actions."""
        if key_clean in [ord('e'), ord('E')]:
            self._toggle_edit_mode()
        elif key_clean in [ord('t'), ord('T')]:
            self._toggle_translate_mode()
        elif key_clean in [ord('a'), ord('A')]:
            self._toggle_width_edit_mode()
        elif key_clean in [ord('v'), ord('V')]:
            self._bulk_retrace_imported_cracks()
        elif key_clean in [ord('g'), ord('G')]:
            self.filter_incompatible_cracks_for_current_group()
        elif key_clean in [ord('y'), ord('Y')]:
            self._close_detachment_polygon()
        elif key_clean in [ord('u'), ord('U')]:
            self._undo_last_action()
        elif key_clean in [ord('r'), ord('R')]:
            self._dead_redo_branch()

    def _handle_keyboard_dispatch(self, key, key_clean, is_plus_key, is_minus_key,
                                   is_widen_left_key, is_widen_right_key,
                                   is_narrow_left_key, is_narrow_right_key):
        """Tries each dispatch group in the original if/elif chain's
        order, stopping at the first one that handles the key."""
        if key_clean == ord('?'):
            self._toggle_help_menu()
            return
        if self._dispatch_window_and_reset_keys(key_clean):
            return
        if self._dispatch_width_edit_keys(key_clean, is_plus_key, is_minus_key,
                                           is_widen_left_key, is_widen_right_key,
                                           is_narrow_left_key, is_narrow_right_key):
            return
        if self._dispatch_view_toggle_keys(key_clean, is_plus_key, is_minus_key):
            return
        if self._dispatch_tool_and_misc_keys(key_clean):
            return
        self._dispatch_action_mode_keys(key_clean)

    def _pan_hover_crack(self, idx, dx, dy):
        """Arrow keys: nudges one specific crack's path/start/end by (dx, dy) while hovering its extension marker."""
        if not (0 <= idx < len(self.cfg.saved_cracks)):
            return
        f = self.cfg.saved_cracks[idx]
        if 'path' in f:
            f['path'] = [(int(pt[0] + dx), int(pt[1] + dy)) for pt in f['path']]
        if 'start' in f:
            f['start'] = (int(f['start'][0] + dx), int(f['start'][1] + dy))
        if 'end' in f:
            f['end'] = (int(f['end'][0] + dx), int(f['end'][1] + dy))

    def _pan_zoom_viewport(self, is_up, is_down, is_left, is_right, step):
        """Part of arrow-key handling: pans the zoomed viewport by `step`
        pixels in the given direction."""
        if is_up:
            self.cfg.zoom_center[1] = max(0, self.cfg.zoom_center[1] - step)
        elif is_down:
            self.cfg.zoom_center[1] = min(self.cfg.H_img, self.cfg.zoom_center[1] + step)
        elif is_left:
            self.cfg.zoom_center[0] = max(0, self.cfg.zoom_center[0] - step)
        elif is_right:
            self.cfg.zoom_center[0] = min(self.cfg.W_img, self.cfg.zoom_center[0] + step)

    def _handle_arrow_keys(self, key, key_clean, step):
        """Pans the viewport (or nudges a hovered crack's endpoint) on an arrow-key press, checked after the main dispatch table."""
        is_arrow_up = (key_clean == 82 or key == 82 or key == 0 or key == 63232)
        is_arrow_down = (key_clean == 84 or key == 84 or key == 1 or key == 63233)
        is_arrow_left = (key_clean == 81 or key == 81 or key == 2 or key == 63234)
        is_arrow_right = (key_clean == 83 or key == 83 or key == 3 or key == 63235)

        if not (is_arrow_up or is_arrow_down or is_arrow_left or is_arrow_right):
            return

        dx, dy = 0, 0
        if is_arrow_up: dy = -1
        elif is_arrow_down: dy = 1
        elif is_arrow_left: dx = -1
        elif is_arrow_right: dx = 1

        if self.cfg.current_tool == 'crack' and self.cfg.temp_start is None and self.cfg.hover_extension_index is not None:
            self._pan_hover_crack(self.cfg.hover_extension_index, dx, dy)
            self.recalculate_masks()
        else:
            self._pan_zoom_viewport(is_arrow_up, is_arrow_down, is_arrow_left, is_arrow_right, step)
        self.refresh_zoom_viewport()

    def handle_keyboard(self, key):
        """Processes typed characters matching them onto system variable adjustments,
        fully optimized for trackpads and protected against multi-tier index errors.
        """

        try:
            key_clean = key & 0xFF

            # While the [?] guide is open, Up/Down scroll its command list instead.
            if self._scroll_help_menu_if_open(key, key_clean):
                return

            step = int(100 / self.cfg.zoom_factor) if self.cfg.zoom_factor > 1.0 else 100
            # Shared by the zoom handlers and the width-edit [+]/[-] handlers (KEY A).
            is_plus_key = key_clean in [ord('+'), ord('='), ord('i'), ord('I'), 43, 61] or key == 65451
            is_minus_key = key_clean in [ord('-'), ord('o'), ord('O'), 45] or key == 65453
            # [ / { / ] / } refine one side at a time of a focused [A] tract.
            is_widen_left_key = key_clean == ord('[')
            is_widen_right_key = key_clean == ord(']')
            is_narrow_left_key = key_clean == ord('{')
            is_narrow_right_key = key_clean == ord('}')

            self._handle_keyboard_dispatch(key, key_clean, is_plus_key, is_minus_key,
                                            is_widen_left_key, is_widen_right_key,
                                            is_narrow_left_key, is_narrow_right_key)
            self._handle_arrow_keys(key, key_clean, step)

        except Exception as e: 
            print(f"[KEYBOARD EXCEPTION] Error: {e}", file=sys.stderr)


    def _strip_existing_building_tag(self, curr_filename):
        """If the file already carries a __BLDG tag, renames it back to its clean name first. Returns False only if the rename fails."""
        if "__BLDG" not in curr_filename:
            return True
        img_base_clean = curr_filename.split("__BLDG")[0]
        _, img_ext = os.path.splitext(curr_filename)
        img_dir = os.path.dirname(self.cfg.CURRENT_IMAGE_PATH)
        clean_img_path = os.path.join(img_dir, f"{img_base_clean}{img_ext}")

        if self.cfg.CURRENT_IMAGE_PATH != clean_img_path and os.path.exists(self.cfg.CURRENT_IMAGE_PATH):
            try:
                os.rename(self.cfg.CURRENT_IMAGE_PATH, clean_img_path)
            except Exception as e:
                print(f"[RENAME ERROR] Could not clean up the previous filename: {e}")
                return False

        self.cfg.CURRENT_IMAGE_PATH = clean_img_path
        # A "clean" path that never has a file on disk -- the real JSON is
        # tracked via old_json_path (captured before this call) and renamed later.
        self.cfg.JSON_OUTPUT_PATH = os.path.join(self.cfg.OUTPUT_FOLDER, f"{img_base_clean}.json")
        return True

    def _rename_image_with_building_tag(self, new_index):
        """Renames the (now untagged) image on disk with the requested group code. Returns (path, filename, base, suffix), or None on failure."""
        building_suffix = f"__BLDG{new_index:03d}"
        img_dir, img_filename = os.path.split(self.cfg.CURRENT_IMAGE_PATH)
        img_base, img_ext = os.path.splitext(img_filename)
        # Safety cleanup of leftover extensions in the base name
        if img_base.endswith(img_ext):
            img_base = img_base[:-len(img_ext)]
        new_img_filename = f"{img_base}{building_suffix}{img_ext}"
        new_img_path = os.path.join(img_dir, new_img_filename)

        try:
            os.rename(self.cfg.CURRENT_IMAGE_PATH, new_img_path)
            print(f"[RENAME SUCCESS] Image associated with the group: {new_img_filename}")
        except Exception as e:
            print(f"[RENAME ERROR] Could not rename the file: {e}")
            return None
        return new_img_path, new_img_filename, img_base, building_suffix

    def _sync_renamed_json(self, old_json_path, new_img_filename, img_base, building_suffix):
        """Renames the already-exported JSON (if any) to match, updating its imagePath. Always returns the new JSON path."""
        new_json_filename = f"{img_base}{building_suffix}.json"
        new_json_path = os.path.join(os.path.dirname(self.cfg.JSON_OUTPUT_PATH), new_json_filename)
        if os.path.exists(old_json_path):
            try:
                with open(old_json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta["imagePath"] = new_img_filename
                with open(old_json_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
                os.rename(old_json_path, new_json_path)
                print(f"[RENAME SUCCESS] Group JSON synced: {new_json_filename}")
            except Exception as e:
                print(f"[RENAME WARNING] Error updating JSON metadata: {e}")
        return new_json_path

    def _apply_building_tag_to_current_file(self, new_index):
        """Strips any existing __BLDG tag and renames the image + its JSON to carry new_index's tag instead. Returns True on success, False if the rename failed.
        """
        # The already-saved JSON under the previous tag (if any) is captured
        # here, before stripping, and synced to the new tag at the end.
        old_json_path = self.cfg.JSON_OUTPUT_PATH
        curr_filename = os.path.basename(self.cfg.CURRENT_IMAGE_PATH)

        if not self._strip_existing_building_tag(curr_filename):
            return False

        rename_result = self._rename_image_with_building_tag(new_index)
        if rename_result is None:
            return False
        new_img_path, new_img_filename, img_base, building_suffix = rename_result

        new_json_path = self._sync_renamed_json(old_json_path, new_img_filename, img_base, building_suffix)

        self.cfg.CURRENT_IMAGE_PATH = new_img_path
        self.cfg.JSON_OUTPUT_PATH = new_json_path
        return True

    def _collect_known_building_indices(self, processed_folder):
        """Every distinct building-group number already tagged on a file in the working or archived folder -- a hint list for the [B] operator."""
        known_indices = set()
        for folder in (processed_folder, self.cfg.OUTPUT_FOLDER):
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if "__BLDG" in f:
                        try:
                            base_name_only, _ = os.path.splitext(f)
                            if base_name_only.endswith("-seg"):
                                base_name_only = base_name_only[:-4]
                            known_indices.add(int(base_name_only.split("__BLDG")[-1]))
                        except ValueError:
                            continue
        return known_indices

    def _print_known_building_groups_hint(self, known_indices):
        if known_indices:
            print(f"[MANUAL GROUP] Building groups already seen in this folder: "
                  f"{', '.join(f'BLDG{i:03d}' for i in sorted(known_indices))}")
        else:
            print("[MANUAL GROUP] No building group present on disk yet.")

    def start_manual_building_group_entry(self):
        """[B] key handler: opens on-screen digit entry to manually (re)assign the building group of the current image, overriding the automatic on-load assignment.
        """

        if self.cfg.CURRENT_IMAGE_PATH is None or not os.path.exists(self.cfg.CURRENT_IMAGE_PATH):
            print("[MANUAL GROUP ERROR] No active image.")
            return

        # Hint: list the building groups already seen on disk, so the operator
        # doesn't have to remember the exact number by heart.
        processed_folder = os.path.join(str(resolve_script_dir()), "already processed images")
        known_indices = self._collect_known_building_indices(processed_folder)
        self._print_known_building_groups_hint(known_indices)

        print(" [MANUAL GROUP] Type the number (with the tool's window active/in the foreground), "
              "ENTER to confirm, BACKSPACE to delete a digit (or to cancel if it's already empty).")
        self.cfg.manual_group_entry_state["active"] = True
        self.cfg.manual_group_entry_state["buffer"] = ""

    def _confirm_manual_building_group_entry(self):
        """Validates and applies the digits typed via the on-screen [B] entry, then closes the entry mode. Empty/non-positive input cancels without changing anything."""

        digits = self.cfg.manual_group_entry_state["buffer"]
        self.cfg.manual_group_entry_state["active"] = False
        self.cfg.manual_group_entry_state["buffer"] = ""

        if not digits:
            print("[MANUAL GROUP] Invalid value, operation cancelled.")
            return
        new_index = int(digits)
        if new_index <= 0:
            print("[MANUAL GROUP] The group number must be positive, operation cancelled.")
            return

        if self._apply_building_tag_to_current_file(new_index):
            self.cfg.CURRENT_BUILDING_INDEX = new_index
            print(f"[MANUAL GROUP] Current image manually assigned to BLDG{new_index:03d}.")




    def _collect_candidate_json_files(self, processed_folder):
        """Every JSON in either the working or archived folder, excluding the current image's own JSON. None if nothing is found."""
        # Scans both the working AND archived folders and merges the results.
        search_folders = [self.cfg.OUTPUT_FOLDER, processed_folder]
        json_files = []
        for folder in search_folders:
            if os.path.exists(folder):
                json_files.extend(
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.lower().endswith('.json')
                )
        if not json_files:
            print("[ALIGNMENT] No JSON file found in 'segmentated images' or 'already processed images'.")
            return None
        json_files = [p for p in json_files if os.path.abspath(p) != os.path.abspath(self.cfg.JSON_OUTPUT_PATH)]
        return json_files or None

    def _pick_best_matching_json(self, json_files):
        """Most-recent JSON belonging to the current image's building group if it has one, otherwise the plain most-recent JSON."""
        json_files.sort(key=os.path.getmtime, reverse=True)
        curr_filename = os.path.basename(self.cfg.CURRENT_IMAGE_PATH) if self.cfg.CURRENT_IMAGE_PATH else ""
        latest_json_path = json_files[0]
        if "__BLDG" in curr_filename:
            curr_bldg_tag = "__" + curr_filename.split("__")[-1].split(".")[0].split("-seg")[0]
            for candidate in json_files:
                if curr_bldg_tag in os.path.basename(candidate):
                    latest_json_path = candidate
                    break
        return latest_json_path

    def _find_image_by_recorded_path(self, original_filename, cartelle_ricerca):
        """Searches for the source image using the JSON's own recorded
        imagePath field."""
        if not original_filename:
            return None
        for cartella in cartelle_ricerca:
            candidate = os.path.join(cartella, original_filename)
            if os.path.exists(candidate):
                return candidate
        return None

    def _find_image_by_guessed_extension(self, latest_json_path, cartelle_ricerca):
        """Fallback when the recorded imagePath doesn't match: tries the JSON's own base name with common image extensions."""
        VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        base_name, _ = os.path.splitext(os.path.basename(latest_json_path))
        if base_name.endswith('-seg'):
            base_name = base_name[:-4]
        for cartella in cartelle_ricerca:
            for ext in VALID_EXTENSIONS:
                candidate = os.path.join(cartella, f"{base_name}{ext}")
                if os.path.exists(candidate):
                    return candidate
        return None

    def find_last_processed_session_asset(self):
        """Scans 'segmentated images' and 'already processed images' to locate the latest JSON and its original image.
        """
        try:
            CARTELLA_BASE = str(resolve_script_dir())
            processed_folder = os.path.join(CARTELLA_BASE, "already processed images")

            json_files = self._collect_candidate_json_files(processed_folder)
            if json_files is None:
                return None, None

            latest_json_path = self._pick_best_matching_json(json_files)

            with open(latest_json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                original_filename = meta.get("imagePath")

            # List of folders to search for the source image
            cartelle_ricerca = [
                self.cfg.IMAGE_FOLDER,                          # Current active folder
                processed_folder,                               # Historical archive
                os.path.join(CARTELLA_BASE, "Images"),          # Standard input folder
                self.cfg.OUTPUT_FOLDER                          # Same folder as the JSON
            ]

            found = self._find_image_by_recorded_path(original_filename, cartelle_ricerca)
            if found is None:
                found = self._find_image_by_guessed_extension(latest_json_path, cartelle_ricerca)

            if found is not None:
                return found, latest_json_path

            print(f"[ALIGNMENT WARNING] Found JSON {os.path.basename(latest_json_path)} but the original image cannot be found in the system paths.")
        except Exception as e:
            print(f"[RECOVERY ERROR] Error scanning the session database: {e}", file=sys.stderr)
        return None, None

    def _import_crack_comparator(self):
        """Imports the standalone comparator utility (reused, not
        duplicated) -- (None, None) if it can't be found."""
        try:
            from compare_and_filter_cracks import CrackComparator, LoadedJson
        except ImportError:
            print(" [GROUP ERROR] compare_and_filter_cracks.py not found in the same folder as this script.")
            return None, None
        return CrackComparator(self.cfg), LoadedJson

    def _collect_group_json_paths(self, bldg_tag, processed_folder):
        """Every already-exported JSON (working + archived folders)
        tagged with this building group, deduplicated by path."""
        json_paths, seen = [], set()
        for folder in (self.cfg.OUTPUT_FOLDER, processed_folder):
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith('.json') and bldg_tag in fname:
                        candidate = os.path.join(folder, fname)
                        ap = os.path.abspath(candidate)
                        if ap not in seen:
                            seen.add(ap)
                            json_paths.append(candidate)
        return json_paths

    def _load_group_jsons(self, json_paths, LoadedJson):
        """Loads every group JSON; None (error already printed) if any
        of them can't be read."""
        try:
            return [LoadedJson(p) for p in json_paths]
        except Exception as e:
            print(f" [GROUP ERROR] Could not read the group's JSON files: {e}", file=sys.stderr)
            return None

    def _reload_current_image_if_affected(self, loaded_jsons):
        """Keeps the on-screen session in sync if the just-rewritten group files included the currently open image. Call AFTER the rewrite."""
        affected_current_image = any(
            os.path.abspath(lj.path) == os.path.abspath(self.cfg.JSON_OUTPUT_PATH) for lj in loaded_jsons
        )
        if affected_current_image:
            self.load_labelme_format()
            self.recalculate_masks()
            self.refresh_zoom_viewport()
            print(" [GROUP] Current image reloaded to reflect the discarded cracks.")

    def filter_incompatible_cracks_for_current_group(self):
        """[G] key handler: cross-checks every crack of the current building group across its exported photos by sinuosity, discarding any that disagree too much -- from every photo of the group, with confirmation first.
        """
        if self.cfg.CURRENT_BUILDING_INDEX is None:
            print(" [GROUP] No active building group on this image "
                  "(automatic assignment hasn't run on any photo yet -- you can assign one with [B]).")
            return

        comparator, LoadedJson = self._import_crack_comparator()
        if comparator is None:
            return

        CARTELLA_BASE = str(resolve_script_dir())
        processed_folder = os.path.join(CARTELLA_BASE, "already processed images")
        bldg_tag = f"__BLDG{self.cfg.CURRENT_BUILDING_INDEX:03d}"

        json_paths = self._collect_group_json_paths(bldg_tag, processed_folder)
        if len(json_paths) < 2:
            print(f" [GROUP] Found only {len(json_paths)} saved photo(s) for group {bldg_tag[2:]} "
                  f"-- at least 2 photos of the same group are needed for a comparison.")
            return

        loaded_jsons = self._load_group_jsons(json_paths, LoadedJson)
        if loaded_jsons is None:
            return

        rows, incompatible, extra_note = comparator.compare(
            loaded_jsons, self.cfg.CRACK_COMPAT_SINUOSITY_THRESHOLD, self.cfg.CRACK_COMPAT_LENGTH_THRESHOLD
        )
        comparator.print_report(rows, self.cfg.CRACK_COMPAT_SINUOSITY_THRESHOLD, self.cfg.CRACK_COMPAT_LENGTH_THRESHOLD, extra_note)

        if not incompatible:
            print(f" [GROUP] Every crack in group {bldg_tag[2:]} is compatible across the {len(json_paths)} photos -- none discarded.")
            return

        # This action deletes shapes from disk, so confirm first.
        if not self._confirm_crack_filter_removal(bldg_tag, json_paths, incompatible):
            print(" [GROUP] Filter cancelled by the operator.")
            return

        comparator.remove_incompatible_and_save(loaded_jsons, incompatible, dry_run=False, make_backup=True, images_dir=None,
                                                 mask_dilation_px=self.cfg.CRACK_MASK_DILATION_PX)
        self._reload_current_image_if_affected(loaded_jsons)

        print(f" [GROUP] Done: {len(incompatible)} crack(s) discarded from {len(json_paths)} photo(s) of group {bldg_tag[2:]}.")

    def run_post_save_group_crack_check(self):
        """Automatic post-save trigger for filter_incompatible_cracks_for_current_group() -- runs after every save so the operator no longer has to press [G] by hand; still asks for confirmation before discarding anything.
        """
        print("\n[AUTO] Save completed -- automatically checking crack "
              "compatibility with the other photos in the building group (if available)...")
        self.filter_incompatible_cracks_for_current_group()


    def _set_help_scroll_from_y(self, y, ry0, track_h, max_off):
        """Maps a y coordinate onto a scroll offset within the track."""
        frac = (y - ry0) / track_h
        self.cfg.help_scroll_offset = max(0, min(max_off, round(frac * max_off)))
        self.refresh_zoom_viewport()

    def _drag_help_scrollbar(self, event, x, y):
        """Starts or continues dragging the help guide's scrollbar
        thumb."""
        rx0, ry0, rx1, ry1 = self.cfg.help_scrollbar_rect
        track_h = max(1, ry1 - ry0)
        max_off = self.cfg.help_scroll_max_offset
        if event == cv2.EVENT_LBUTTONDOWN and rx0 <= x <= rx1 and ry0 <= y <= ry1:
            self.cfg.help_scrollbar_dragging = True
            self._set_help_scroll_from_y(y, ry0, track_h, max_off)
        elif event == cv2.EVENT_MOUSEMOVE and self.cfg.help_scrollbar_dragging:
            self._set_help_scroll_from_y(y, ry0, track_h, max_off)

    def _handle_help_scrollbar_mouse(self, event, x, y):
        """Handles mouse input while the [?] guide is open: drags its
        scrollbar, ignores clicks anywhere else on the photo."""
        if self.cfg.help_scrollbar_rect is not None:
            self._drag_help_scrollbar(event, x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.cfg.help_scrollbar_dragging = False

    def _complete_calibration(self, real_x, real_y):
        """Second calibration click: computes and applies the new
        pixel-to-cm scale from the traced distance."""
        x1, y1 = self.cfg.calib_start
        dist_pixels = np.sqrt((real_x - x1)**2 + (real_y - y1)**2)
        if dist_pixels > 0:
            print(f"\n[CALIBRATION] Pixels traced: {dist_pixels:.2f} px")
            real_dist_cm = self._prompt_calibration_distance()
            if real_dist_cm is None:
                print("[CALIBRATION ERROR] Invalid input.")
            elif real_dist_cm > 0:
                self.cfg.PIXEL_TO_CM_SCALE = real_dist_cm / dist_pixels
                print(f"[CALIBRATION SUCCESS] New scale: {self.cfg.PIXEL_TO_CM_SCALE:.6f} cm/pixel")
        self.cfg.calib_start = None
        self.cfg.calibration_mode = False
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _handle_calibration_mouse(self, event, x, y):
        """Handles the two-click distance measurement for [L]/
        calibration mode."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self.cfg.calib_start is None:
            self.cfg.calib_start = (real_x, real_y)
        else:
            self._complete_calibration(real_x, real_y)

    def _find_nearest_crack_point(self, real_x, real_y, tolerance=None):
        """Nearest point across all saved cracks, optionally within a
        tolerance. Returns (None, None) if nothing qualifies."""
        best_dist = float('inf')
        best_crack_idx, best_pt_idx = None, None
        for idx, f in enumerate(self.cfg.saved_cracks):
            path = f.get('path')
            if not path:
                continue
            for p_idx, pt in enumerate(path):
                dist = np.sqrt((real_x - pt[0]) ** 2 + (real_y - pt[1]) ** 2)
                if (tolerance is None or dist <= tolerance) and dist < best_dist:
                    best_dist = dist
                    best_crack_idx, best_pt_idx = idx, p_idx
        return best_crack_idx, best_pt_idx

    def _select_width_tract_start(self, real_x, real_y):
        """First [A] click: nearest point on ANY crack, within
        tolerance."""
        best_crack_idx, best_pt_idx = self._find_nearest_crack_point(real_x, real_y, tolerance=22)
        if best_crack_idx is not None:
            self.cfg.width_edit_state["pending_crack_idx"] = best_crack_idx
            self.cfg.width_edit_state["pending_i0"] = best_pt_idx
            self.cfg.width_edit_state["focused_crack_idx"] = None
            self.cfg.width_edit_state["focused_seg_idx"] = None
            print(f" [WIDTH] Starting point selected (crack #{best_crack_idx}). Click the ending point of the tract.")
        else:
            print(" [WIDTH] No crack found near the click.")

    def _find_nearest_point_on_crack(self, path, real_x, real_y):
        """Nearest point index on a SPECIFIC crack's path, no tolerance
        limit."""
        best_dist = float('inf')
        best_pt_idx = None
        for p_idx, pt in enumerate(path):
            dist = np.sqrt((real_x - pt[0]) ** 2 + (real_y - pt[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_pt_idx = p_idx
        return best_pt_idx

    def _create_width_segment(self, fc, i0, i1):
        """Builds and focuses a new width_segments entry, auto-filled
        from the crack's real pixels."""
        auto_fill = self._autofill_tract_mask(self.cfg.saved_cracks[fc]['path'], i0, i1)
        new_seg = {'i0': i0, 'i1': i1, 'width_px': 0, 'fill': auto_fill,
                   'margin_left_px': 0, 'margin_right_px': 0}
        self.cfg.saved_cracks[fc].setdefault('width_segments', []).append(new_seg)
        self.cfg.width_edit_state["focused_crack_idx"] = fc
        self.cfg.width_edit_state["focused_seg_idx"] = len(self.cfg.saved_cracks[fc]['width_segments']) - 1
        return auto_fill

    def _print_width_tract_selected(self, fc, i0, i1, auto_fill):
        """Prints the tract-selected status message, worded differently
        for an auto-filled vs. a faint (unfilled) tract."""
        if auto_fill:
            print(f" [WIDTH] Tract selected: crack #{fc}, {i1 - i0 + 1} points. "
                  f"Automatic fill applied from the crack's pixels in the photo. "
                  f"Use [+]/[-] to refine it (widen/narrow, current: 0px). "
                  f"Click elsewhere for a new tract, [A] to exit.")
        else:
            print(f" [WIDTH] Tract selected: crack #{fc}, {i1 - i0 + 1} points. "
                  f"No fill automatically detected in this tract (crack too faint). "
                  f"Use [+]/[-] to widen it manually (current: 0px per side). "
                  f"Click elsewhere for a new tract, [A] to exit.")

    def _select_width_tract_end(self, real_x, real_y):
        """Second [A] click: nearest point on the SAME crack, then
        creates the width segment between the two indices."""
        fc = self.cfg.width_edit_state["pending_crack_idx"]
        path = self.cfg.saved_cracks[fc].get('path') if 0 <= fc < len(self.cfg.saved_cracks) else None
        if path:
            best_pt_idx = self._find_nearest_point_on_crack(path, real_x, real_y)
            i0, i1 = self.cfg.width_edit_state["pending_i0"], best_pt_idx
            if i0 is not None and i1 is not None:
                if i0 > i1:
                    i0, i1 = i1, i0
                auto_fill = self._create_width_segment(fc, i0, i1)
                self._print_width_tract_selected(fc, i0, i1, auto_fill)
        else:
            print(" [WIDTH] The selected crack is no longer available, try again.")
        self.cfg.width_edit_state["pending_crack_idx"] = None
        self.cfg.width_edit_state["pending_i0"] = None

    def _handle_width_edit_mouse(self, event, x, y):
        """Handles the two-click tract selection for [A]/width-edit
        mode."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self.cfg.width_edit_state["pending_crack_idx"] is None:
            self._select_width_tract_start(real_x, real_y)
        else:
            self._select_width_tract_end(real_x, real_y)
        self.refresh_zoom_viewport()

    def _anchor_translate_crack(self, real_x, real_y):
        """First [T] click: anchors the nearest point on any crack,
        within tolerance."""
        best_dist = float('inf')
        for idx, f in enumerate(self.cfg.saved_cracks):
            if 'path' in f:
                for pt in f['path']:
                    dist = np.sqrt((real_x - pt[0])**2 + (real_y - pt[1])**2)
                    if dist <= 22 and dist < best_dist:
                        best_dist = dist
                        self.cfg.translate_state["idx"] = idx
                        self.cfg.translate_state["start"] = (pt[0], pt[1])
        if self.cfg.translate_state["idx"] is not None:
            print(f" [TRANSLATE] Crack #{self.cfg.translate_state['idx']} anchored. Now click on the REAL crack in the image.")

    def _apply_translate_snap(self, f, shifted_start, shifted_end, nuovo_path):
        """Commits a successful [T] snap: backs up the old shape (for
        Undo), then overwrites it with the new A* path."""
        backup = {'index': self.cfg.translate_state["idx"], 'start': f['start'], 'end': f['end'], 'path': f['path']}
        f['path'] = nuovo_path
        f['start'] = list(shifted_start)
        f['end'] = list(shifted_end)
        print(f" [AUTOMATIC SNAP COMPLETE] Crack #{self.cfg.translate_state['idx']} realigned to the real fracture. Click another point for another correction, or press [T] to exit.")
        self.cfg.action_history.append(('crack_extension', backup))
        self.recalculate_masks()

    def _snap_translate_crack(self, real_x, real_y):
        """Second [T] click: re-traces the anchored crack via A* against
        the real edge near the clicked point."""
        f = self.cfg.saved_cracks[self.cfg.translate_state["idx"]]
        dx = real_x - self.cfg.translate_state["start"][0]
        dy = real_y - self.cfg.translate_state["start"][1]
        if 'start' not in f or 'end' not in f:
            print(f" [AUTOMATIC SNAP FAILED] Crack #{self.cfg.translate_state['idx']} has no valid start/end points.")
            return
        shifted_start = (f['start'][0] + dx, f['start'][1] + dy)
        shifted_end = (f['end'][0] + dx, f['end'][1] + dy)
        nuovo_path = self.a_star_pathfinding(shifted_start, shifted_end, 'crack')
        if nuovo_path:
            self._apply_translate_snap(f, shifted_start, shifted_end, nuovo_path)
        else:
            print(f" [AUTOMATIC SNAP FAILED] No credible edge found near the clicked point: crack #{self.cfg.translate_state['idx']} was not modified. Try again clicking closer to the real fracture.")

    def _handle_translate_mouse(self, event, x, y):
        """Handles the two-click re-snap for [T]/translate mode."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self.cfg.translate_state["start"] is None:
            self._anchor_translate_crack(real_x, real_y)
        else:
            self._snap_translate_crack(real_x, real_y)
            self.cfg.translate_state["start"] = None
            self.cfg.translate_state["idx"] = None
        self.refresh_zoom_viewport()

    def _edit_mode_select_node(self, x, y):
        """[E] click: selects the nearest detachment click-node within
        tolerance, if any."""
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        for p_idx, d in enumerate(self.cfg.saved_detachments):
            if 'click_nodes' in d:
                for n_idx, pt in enumerate(d['click_nodes']):
                    if abs(real_x - pt[0]) <= 22 and abs(real_y - pt[1]) <= 22:
                        self.cfg.selected_edit_poly = p_idx
                        self.cfg.selected_edit_idx = n_idx
                        print(f"[EDIT] Real node {n_idx} of detachment #{p_idx} anchored")
                        return

    def _rebuild_detachment_path_from_nodes(self, d):
        """Regenerates a detachment's whole A* path from its (possibly
        just-moved) click_nodes, including the closing segment."""
        nuovo_path = [d['click_nodes'][0]]
        for i in range(len(d['click_nodes']) - 1):
            segmento = self.a_star_pathfinding(d['click_nodes'][i], d['click_nodes'][i+1], 'detachment')
            if segmento:
                nuovo_path.extend(segmento[1:])
            else:
                nuovo_path.append(d['click_nodes'][i+1])
        segmento_chiusura = self.a_star_pathfinding(d['click_nodes'][-1], d['click_nodes'][0], 'detachment')
        if segmento_chiusura:
            nuovo_path.extend(segmento_chiusura[1:])
        d['path'] = nuovo_path

    def _edit_mode_drag_node(self, x, y):
        """[E] drag: moves the selected node and live-recomputes the
        whole polygon's A* outline."""
        if self.cfg.selected_edit_poly is None or self.cfg.selected_edit_idx is None:
            return
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        d = self.cfg.saved_detachments[self.cfg.selected_edit_poly]
        d['click_nodes'][self.cfg.selected_edit_idx] = (real_x, real_y)
        self._rebuild_detachment_path_from_nodes(d)
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _edit_mode_release_node(self):
        """[E] release: finalizes the drag."""
        if self.cfg.selected_edit_poly is not None:
            print("[EDIT] Outline recalculated and saved along the mask.")
            self.cfg.selected_edit_poly = None
            self.cfg.selected_edit_idx = None
            self.recalculate_masks()
            self.refresh_zoom_viewport()

    def _handle_edit_mode_mouse(self, event, x, y):
        """Handles node-dragging for [E]/edit mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._edit_mode_select_node(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._edit_mode_drag_node(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._edit_mode_release_node()

    def _delete_crack_near(self, real_x, real_y):
        """Deletes the first crack with a start/end marker within click
        tolerance. Returns True if one was deleted."""
        tolleranza_click = 22
        for idx, f in enumerate(self.cfg.saved_cracks):
            for k_p in ['start', 'end']:
                if k_p in f:
                    kp_x, kp_y = f[k_p]
                    if abs(real_x - kp_x) <= tolleranza_click and abs(real_y - kp_y) <= tolleranza_click:
                        removed_crack = self.cfg.saved_cracks.pop(idx)
                        self.cfg.action_history.append(('crack_delete', removed_crack))
                        self.recalculate_masks()
                        self.refresh_zoom_viewport()
                        return True
        return False

    def _start_new_crack_trace(self, real_x, real_y):
        """First click of a new crack trace: anchors the start point."""
        self.cfg.redo_history.clear()
        self.cfg.temp_start = (real_x, real_y)
        self.cfg.temp_path = [self.cfg.temp_start]
        if self.cfg.hover_extension_index is not None:
            self.cfg.action_history.append(('crack_extension_start', self.cfg.hover_extension_index, self.cfg.hover_extension_mode))
        else:
            self.cfg.action_history.append('crack_start')

    def _complete_new_crack_trace(self, end_point):
        """Second click of a new crack trace: traces via A* and, if
        successful, saves the new crack."""
        computed_path = self.a_star_pathfinding(self.cfg.temp_start, end_point, 'crack')
        if computed_path:
            new_crack = {
                'start': self.cfg.temp_start,
                'end': end_point,
                'path': computed_path,
                'active': True,
                'session_id': str(time.time())
            }
            self.cfg.saved_cracks.append(new_crack)
            self.cfg.action_history.append('crack')
            self.cfg.temp_start = None
            self.cfg.temp_path.clear()
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _crack_tool_click(self, x, y):
        """Dispatches a crack-tool click: delete-by-proximity (only
        with no trace in progress), else start/complete a trace."""
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self.cfg.temp_start is None and self._delete_crack_near(real_x, real_y):
            return
        if self.cfg.temp_start is None:
            self._start_new_crack_trace(real_x, real_y)
        else:
            self._complete_new_crack_trace((real_x, real_y))

    def _crack_tool_preview(self, x, y):
        """Live-updates the in-progress trace's preview line."""
        if self.cfg.temp_start is not None:
            real_x, real_y = self.transform_window_to_real_coords(x, y)
            self.cfg.temp_path = [self.cfg.temp_start, (real_x, real_y)]
            self.refresh_zoom_viewport()

    def _handle_crack_tool_mouse(self, event, x, y):
        """Handles the crack tool's clicks/drag preview."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._crack_tool_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._crack_tool_preview(x, y)

    def _delete_detachment_near(self, real_x, real_y):
        """Deletes the first detachment whose start marker is within
        click tolerance. Returns True if one was deleted."""
        tolleranza_click = 22
        for idx, d in enumerate(self.cfg.saved_detachments):
            if 'path' in d and len(d['path']) > 0:
                d_sx, d_sy = d['path'][0]
                if abs(real_x - d_sx) <= tolleranza_click and abs(real_y - d_sy) <= tolleranza_click:
                    removed_detachment = self.cfg.saved_detachments.pop(idx)
                    self.cfg.action_history.append(('detachment_delete', removed_detachment))
                    self.recalculate_masks()
                    self.refresh_zoom_viewport()
                    return True
        return False

    def _add_detachment_node(self, real_x, real_y):
        """Adds one click-node to the in-progress polygon, tracing an
        A* segment from the previous node."""
        if len(self.cfg.temp_nodes) == 0:
            self.cfg.redo_history.clear()
            self.cfg.user_clicked_nodes = [(real_x, real_y)]
            self.cfg.temp_nodes.append((real_x, real_y))
        else:
            ultimo_punto = self.cfg.temp_nodes[-1]
            nuovo_punto = (real_x, real_y)
            self.cfg.user_clicked_nodes.append(nuovo_punto)
            computed_segment = self.a_star_pathfinding(ultimo_punto, nuovo_punto, 'detachment')
            if computed_segment:
                self.cfg.temp_nodes.extend(computed_segment[1:])
            else:
                self.cfg.temp_nodes.append(nuovo_punto)
        self.cfg.action_history.append('detachment_node')
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _detachment_tool_click(self, x, y):
        """Dispatches a detachment-tool click: delete-by-proximity, or
        add a new polygon node."""
        real_x, real_y = self.transform_window_to_real_coords(x, y)
        if self._delete_detachment_near(real_x, real_y):
            return
        self._add_detachment_node(real_x, real_y)

    def _detachment_tool_preview(self, x, y):
        """Live-updates the in-progress polygon's preview segment."""
        if len(self.cfg.temp_nodes) > 0:
            real_x, real_y = self.transform_window_to_real_coords(x, y)
            ultimo_punto = self.cfg.temp_nodes[-1]
            nuovo_punto = (real_x, real_y)
            preview_segment = self.a_star_pathfinding(ultimo_punto, nuovo_punto, 'detachment')
            if preview_segment:
                self.cfg.temp_path = list(self.cfg.temp_nodes) + preview_segment[1:]
            else:
                self.cfg.temp_path = list(self.cfg.temp_nodes) + [nuovo_punto]
            self.refresh_zoom_viewport()

    def _handle_detachment_tool_mouse(self, event, x, y):
        """Handles the detachment tool's clicks/drag preview."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._detachment_tool_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._detachment_tool_preview(x, y)

    def mouse_callback(self, event, x, y, flags, param):
        """Handles mouse clicks, transforming window coordinates into real image pixels with A* edge/user-node edit support."""
        try:
            if self.cfg.show_help_menu:
                self._handle_help_scrollbar_mouse(event, x, y)
                return

            if self.cfg.calibration_mode:
                self._handle_calibration_mouse(event, x, y)
                return

            if self.cfg.width_edit_state["active"]:
                self._handle_width_edit_mouse(event, x, y)
                return

            if self.cfg.translate_state["active"]:
                self._handle_translate_mouse(event, x, y)
                return

            if self.cfg.edit_mode:
                self._handle_edit_mode_mouse(event, x, y)
                return

            if self.cfg.current_tool == 'crack':
                self._handle_crack_tool_mouse(event, x, y)
            elif self.cfg.current_tool == 'detachment':
                self._handle_detachment_tool_mouse(event, x, y)

        except Exception as e:
            print(f"[MOUSE EXCEPTION] Error: {e}", file=sys.stderr)

    def _compute_total_hud_elements(self):
        """Real-time count of every saved crack + detachment in this
        session."""
        return len(self.cfg.saved_cracks) + len(self.cfg.saved_detachments)

    def _resolve_building_hud_text(self, filename, total_elements):
        """Builds the BUILDING status line's text and color, based on
        whether the filename carries a group tag."""
        if "__BLDG" not in filename:
            return "BUILDING: NOT ASSIGNED (Press 'R')", (80, 175, 235)  # Softened orange -- needs attention
        try:
            bldg_text = filename.split("__")[-1]
            if bldg_text.lower().endswith('.jpg') or bldg_text.lower().endswith('.png') or bldg_text.lower().endswith('.jpeg'):
                bldg_text = os.path.splitext(bldg_text)[0]
            if bldg_text.endswith("-seg"):
                bldg_text = bldg_text[:-4]
            return f"BUILDING: {bldg_text} | ELEMENTS: {total_elements}", (110, 210, 110)  # Softened green -- assigned OK
        except Exception:
            return "BUILDING: NAME PARSING ERROR", (90, 90, 235)  # Softened red -- error

    def _draw_hud_text_with_shadow(self, win_out_canvas, hud_text, text_color, y_pos=None):
        """Draws the HUD line with a black contrast shadow behind it."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 1.1 if self.cfg.hud_large_size else 0.60
        f_thick = 3 if self.cfg.hud_large_size else 2
        if y_pos is None:
            # Fallback position matching the original fixed layout, for any
            # caller not passing the dynamic one _draw_hud_stack computes.
            y_pos = 180 if self.cfg.hud_large_size else 140
        position = (15, y_pos)
        cv2.putText(win_out_canvas, hud_text, (position[0] + 1, position[1] + 1), font, f_scale, (0, 0, 0), f_thick + 1, cv2.LINE_AA)
        cv2.putText(win_out_canvas, hud_text, position, font, f_scale, text_color, f_thick, cv2.LINE_AA)

    def draw_building_hud_overlay(self, win_out_canvas, y_pos=None):
        """Draws, overlaid on the window canvas, the current building code,
        the tracing status, and the total number of elements currently loaded.
        """
        if self.cfg.CURRENT_IMAGE_PATH is None:
            return
        # Shares the [J]/Info toggle with the status/metrics panel above.
        if not self.cfg.show_info_overlay:
            return

        filename = os.path.basename(self.cfg.CURRENT_IMAGE_PATH)
        total_elements = self._compute_total_hud_elements()
        hud_text, text_color = self._resolve_building_hud_text(filename, total_elements)
        self._draw_hud_text_with_shadow(win_out_canvas, hud_text, text_color, y_pos)

    def _paint_temp_crack_preview(self, scene):
        """Paints the in-progress crack trace onto the scene, before
        any zoom/crop."""
        if len(self.cfg.temp_path) > 1 and self.cfg.current_tool == 'crack':
            for pt in self.cfg.temp_path:
                pt_x, pt_y = int(pt[0]), int(pt[1])
                if 0 <= pt_x < self.cfg.W_img and 0 <= pt_y < self.cfg.H_img:
                    scene[pt_y, pt_x] = (0, 165, 255)

    def _ensure_visual_mask_shape(self, mask, h_actual, w_active):
        """Creates or resizes a visual mask (blue/green fill) to match
        the scene's current size. Returns the (possibly new) mask."""
        if mask is None or mask.size == 0 or len(mask.shape) < 2:
            return np.zeros((h_actual, w_active), dtype=np.uint8)
        elif mask.shape[:2] != (h_actual, w_active):
            return cv2.resize(mask, (w_active, h_actual), interpolation=cv2.INTER_NEAREST)
        return mask

    def _paint_width_segment_fills(self, scene, h_actual, w_active):
        """Paints every width-edited tract's real fill shape, for every
        saved crack -- not just the one currently focused."""
        for _wf in self.cfg.saved_cracks:
            if not _wf.get('active', True):
                continue
            _wf_path = _wf.get('path')
            if not _wf_path:
                continue
            for _wseg in (_wf.get('width_segments') or []):
                try:
                    _wi0, _wi1 = int(_wseg.get('i0', 0)), int(_wseg.get('i1', 0))
                except (TypeError, ValueError):
                    continue
                if _wi0 > _wi1:
                    _wi0, _wi1 = _wi1, _wi0
                _wi0 = max(0, _wi0)
                _wi1 = min(len(_wf_path) - 1, _wi1)
                if _wi0 > _wi1:
                    continue
                _wseg_mask = self._render_width_segment_mask(_wf_path, _wi0, _wi1, _wseg, h_actual, w_active)
                if np.any(_wseg_mask):
                    scene[_wseg_mask == 255] = (255, 0, 0)

    def _paint_blue_crack_overlay(self, scene, k, h_actual, w_active):
        """Paints the blue crack fill, gated by [N]. Detachments (green)
        are intentionally not gated by this same flag."""
        if not self.cfg.show_crack_overlay:
            return
        mb = cv2.dilate(self.cfg.blue_visual_mask, k)
        scene[mb == 255] = (255, 0, 0)
        self._paint_width_segment_fills(scene, h_actual, w_active)

    def _paint_green_detachment_overlay(self, scene, k):
        """Alpha-blends the green detachment fill onto the scene --
        always shown, never gated by [N]."""
        mv = cv2.dilate(self.cfg.green_visual_mask, k)
        if np.any(mv == 255):
            overlay = scene.copy()
            overlay[mv == 255] = (0, 200, 0)
            alpha_blend = 0.35
            cv2.addWeighted(overlay, alpha_blend, scene, 1 - alpha_blend, 0, scene)

    def _paint_width_edit_live_preview(self, scene, h_actual, w_active):
        """Highlights the focused width-edit tract in yellow, at its
        current width, for live comparison against the real photo."""
        if not (self.cfg.width_edit_state["active"] and self.cfg.width_edit_state["focused_crack_idx"] is not None):
            return
        fc = self.cfg.width_edit_state["focused_crack_idx"]
        fs = self.cfg.width_edit_state["focused_seg_idx"]
        if not (0 <= fc < len(self.cfg.saved_cracks)):
            return
        segs = self.cfg.saved_cracks[fc].get('width_segments', [])
        if fs is None or not (0 <= fs < len(segs)):
            return
        seg = segs[fs]
        seg_path = self.cfg.saved_cracks[fc].get('path', [])
        i0, i1 = seg.get('i0', 0), seg.get('i1', -1)
        if not (0 <= i0 <= i1 < len(seg_path)):
            return
        seg_mask = self._render_width_segment_mask(seg_path, i0, i1, seg, h_actual, w_active)
        if np.any(seg_mask):
            preview_overlay = scene.copy()
            preview_overlay[seg_mask == 255] = (0, 255, 255)  # bright yellow highlight
            cv2.addWeighted(preview_overlay, 0.55, scene, 0.45, 0, scene)

    def _compute_zoom_roi(self, scene, h_actual, w_active):
        """Crops the scene to the current zoom box, resetting it to the
        full frame if it's ever invalid."""
        z_xmin, z_ymin, z_xmax, z_ymax = self.cfg.zoom_box
        x0, y0, x1, y1 = int(z_xmin), int(z_ymin), int(z_xmax), int(z_ymax)

        if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0 or x1 > w_active or y1 > h_actual:
            x0, y0, x1, y1 = 0, 0, w_active, h_actual
            self.cfg.zoom_box = [0, 0, w_active, h_actual]

        roi = scene[y0:y1, x0:x1]
        if roi.size == 0 or len(roi.shape) < 2:
            roi = scene.copy()
        return roi

    def _resize_roi_to_window(self, roi):
        """Resizes the (zoomed) ROI to the real window's current size,
        falling back to 1200x900 if the window isn't available."""
        window_rect = cv2.getWindowImageRect("Crack Detector Workspace")
        if window_rect is not None and len(window_rect) >= 4:
            _, _, w_win, h_win = window_rect
            if w_win > 100 and h_win > 100:
                return cv2.resize(roi, (w_win, h_win), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(roi, (1200, 900), interpolation=cv2.INTER_LINEAR)

    def _draw_detachment_outline_and_marker(self, win_out, d, marker_dim):
        """Draws one detachment's red outline plus its start-node
        triangle marker (filled if active, hollow if soft-deleted)."""
        window_poly_pts = []
        for pt_real in d['path']:
            wx_p, wy_p = self.transform_real_to_window_coords(int(pt_real[0]), int(pt_real[1]))
            window_poly_pts.append([wx_p, wy_p])
        cv2.polylines(win_out, [np.array(window_poly_pts, dtype=np.int32)], True, (0, 0, 255), 2)

        d_sx, d_sy = d['path'][0][0], d['path'][0][1]
        wx, wy = self.transform_real_to_window_coords(int(d_sx), int(d_sy))
        if not (0 <= wx < 1200 and 0 <= wy < 900):
            return
        triangle_pts = np.array([[wx, wy-marker_dim], [wx-marker_dim, wy+marker_dim], [wx+marker_dim, wy+marker_dim]], dtype=np.int32)
        cv2.polylines(win_out, [triangle_pts], True, (0, 0, 255), 2)
        if d.get('active', True):
            inner_dim = int(marker_dim * 0.4)
            inner_triangle_pts = np.array([[wx, wy-inner_dim], [wx-inner_dim, wy+inner_dim], [wx+inner_dim, wy+inner_dim]], dtype=np.int32)
            cv2.fillPoly(win_out, [inner_triangle_pts], (0, 0, 255))

    def _draw_detachment_markers(self, win_out, marker_dim):
        """Draws every saved detachment's outline/marker."""
        for d in self.cfg.saved_detachments:
            if 'path' in d and len(d['path']) > 0:
                self._draw_detachment_outline_and_marker(win_out, d, marker_dim)

    def _draw_in_progress_detachment_trace(self, win_out, marker_dim):
        """Draws the polygon currently being clicked out for a new
        detachment, not yet closed with [Y]."""
        if not (self.cfg.current_tool == 'detachment' and (self.cfg.temp_nodes or self.cfg.temp_path)):
            return
        tracciato_corrente = self.cfg.temp_path if len(self.cfg.temp_path) > 0 else self.cfg.temp_nodes
        window_temp_pts = []
        for idx, n in enumerate(tracciato_corrente):
            wx, wy = self.transform_real_to_window_coords(int(n[0]), int(n[1]))
            window_temp_pts.append([wx, wy])

            if idx == 0 and len(self.cfg.temp_nodes) > 0:
                if 0 <= wx < 1200 and 0 <= wy < 900:
                    cv2.polylines(win_out, [np.array([[wx, wy-marker_dim], [wx-marker_dim, wy+marker_dim], [wx+marker_dim, wy+marker_dim]], dtype=np.int32)], True, (0, 0, 255), 2)
            elif idx > 0 and len(self.cfg.temp_nodes) > idx:
                if 0 <= wx < 1200 and 0 <= wy < 900 and n in self.cfg.temp_nodes:
                    cv2.circle(win_out, (wx, wy), 4, (0, 255, 255), -1)

        if len(window_temp_pts) > 1:
            cv2.polylines(win_out, [np.array(window_temp_pts, dtype=np.int32)], False, (0, 0, 255), 2)

    def _draw_edit_mode_indicator(self, win_out):
        """Draws the "EDIT MODE ACTIVE" text while [E] is on."""
        if not self.cfg.edit_mode:
            return
        e_scale = 1.2 if self.cfg.hud_large_size else 0.65
        e_thick = 3 if self.cfg.hud_large_size else 2
        cv2.putText(win_out, "EDIT MODE ACTIVE", (15, 135 if self.cfg.hud_large_size else 120), cv2.FONT_HERSHEY_SIMPLEX, e_scale, (0, 165, 255), e_thick, cv2.LINE_AA)

    def _draw_translate_click_nodes(self, win_out):
        """Draws every detachment click-node, plus a green ring around
        the one currently selected for editing."""
        for d in self.cfg.saved_detachments:
            if 'click_nodes' in d:
                for pt in d['click_nodes']:
                    wx, wy = self.transform_real_to_window_coords(int(pt[0]), int(pt[1]))
                    if 0 <= wx < 1200 and 0 <= wy < 900:
                        cv2.circle(win_out, (wx, wy), 5, (0, 165, 255), -1)

        if self.cfg.selected_edit_poly is not None and self.cfg.selected_edit_idx is not None:
            pt_active = self.cfg.saved_detachments[self.cfg.selected_edit_poly]['click_nodes'][self.cfg.selected_edit_idx]
            wx, wy = self.transform_real_to_window_coords(int(pt_active[0]), int(pt_active[1]))
            if 0 <= wx < 1200 and 0 <= wy < 900:
                cv2.circle(win_out, (wx, wy), 10, (0, 255, 0), 2)

    def _draw_translate_mode_indicator(self, win_out):
        """Draws the [T] automatic-snap tool's status text and anchor
        guide, reading the real translate_state (not a dead variable)."""
        if not self.cfg.translate_state["active"]:
            return
        t_scale = 1.2 if self.cfg.hud_large_size else 0.65
        t_thick = 3 if self.cfg.hud_large_size else 2
        msg = "AUTOMATIC SNAP MODE (T): Select crack" if self.cfg.translate_state["start"] is None else "AUTOMATIC SNAP MODE: Click on the real crack"
        cv2.putText(win_out, msg, (15, 135 if self.cfg.hud_large_size else 120), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (255, 0, 255), t_thick, cv2.LINE_AA)

        if self.cfg.translate_state["start"] is not None:
            wx_start, wy_start = self.transform_real_to_window_coords(int(self.cfg.translate_state["start"][0]), int(self.cfg.translate_state["start"][1]))
            cv2.circle(win_out, (wx_start, wy_start), 6, (255, 0, 255), -1)
            self._draw_translate_click_nodes(win_out)

    def _compute_width_edit_hud_messages(self):
        """Builds the status line(s) for the focused/pending width-edit
        tract. Returns (msg_a, msg_a2), msg_a2 may be None."""
        msg_a2 = None
        if self.cfg.width_edit_state["focused_crack_idx"] is not None:
            fc = self.cfg.width_edit_state["focused_crack_idx"]
            fs = self.cfg.width_edit_state["focused_seg_idx"]
            msg_a = f"CRACK WIDTH (A): crack #{fc}"
            if 0 <= fc < len(self.cfg.saved_cracks):
                segs = self.cfg.saved_cracks[fc].get('width_segments', [])
                if fs is not None and 0 <= fs < len(segs):
                    msg_a, msg_a2 = self._describe_focused_width_segment(fc, segs[fs])
        elif self.cfg.width_edit_state["pending_crack_idx"] is not None:
            msg_a = f"CRACK WIDTH (A): crack #{self.cfg.width_edit_state['pending_crack_idx']} -- click the ending point of the tract"
        else:
            msg_a = "CRACK WIDTH (A): click the starting point of the tract to widen"
        return msg_a, msg_a2

    def _describe_focused_width_segment(self, fc, seg):
        """Describes a focused tract: measured-average wording for an
        auto-filled tract, fixed-width wording for a legacy one."""
        cur_w = seg.get('width_px', 0)
        if seg.get('fill'):
            seg_path = self.cfg.saved_cracks[fc].get('path', [])
            i0, i1 = seg.get('i0', 0), seg.get('i1', -1)
            measured_px = 0
            if 0 <= i0 <= i1 < len(seg_path):
                seg_mask = self._render_width_segment_mask(seg_path, i0, i1, seg, self.cfg.H_img, self.cfg.W_img)
                tract_len = max(1, i1 - i0 + 1)
                measured_px = int(round(np.count_nonzero(seg_mask) / tract_len))
            mgn_l = int(seg.get('margin_left_px', 0) or 0)
            mgn_r = int(seg.get('margin_right_px', 0) or 0)
            msg_a = (f"CRACK WIDTH (A): crack #{fc} -- automatic fill, "
                     f"~{measured_px}px average width, refinement {cur_w:+d}px.")
            msg_a2 = (f"[+]/[-] both sides | "
                      f"[/{{ left {mgn_l:+d}px, ]/}} right {mgn_r:+d}px | click = new tract")
        else:
            mgn_l = int(seg.get('margin_left_px', 0) or 0)
            mgn_r = int(seg.get('margin_right_px', 0) or 0)
            msg_a = (f"CRACK WIDTH (A): crack #{fc} -- {cur_w}px per side (~{2*cur_w+1}px total). "
                     f"[+]/[-] adjusts, click = new tract")
            msg_a2 = (f"[/{{ left {mgn_l:+d}px, ]/}} right {mgn_r:+d}px | click = new tract")
        return msg_a, msg_a2

    def _draw_width_edit_hud(self, win_out):
        """Draws the [A] crack-width tool's status text, on its own
        dedicated HUD row."""
        if not self.cfg.width_edit_state["active"]:
            return
        a_scale = 1.2 if self.cfg.hud_large_size else 0.65
        a_thick = 3 if self.cfg.hud_large_size else 2
        a_y = 180 if self.cfg.hud_large_size else 148
        msg_a, msg_a2 = self._compute_width_edit_hud_messages()
        cv2.putText(win_out, msg_a, (15, a_y), cv2.FONT_HERSHEY_SIMPLEX, a_scale, (0, 255, 255), a_thick, cv2.LINE_AA)
        if msg_a2 is not None:
            a_y2 = a_y + (32 if self.cfg.hud_large_size else 22)
            cv2.putText(win_out, msg_a2, (15, a_y2), cv2.FONT_HERSHEY_SIMPLEX, a_scale, (0, 255, 255), a_thick, cv2.LINE_AA)

    def _draw_manual_group_entry_banner(self, win_out):
        """Draws the [B] on-screen digit-entry banner -- replaces what
        used to be a blocking terminal prompt."""
        if not self.cfg.manual_group_entry_state["active"]:
            return
        cv2.rectangle(win_out, (15, 250), (1185, 310), (0, 140, 255), -1)
        typed = self.cfg.manual_group_entry_state["buffer"]
        if typed:
            msg_b = f"BUILDING GROUP (B): type the number -- {typed}  (BLDG{int(typed):03d})"
        else:
            msg_b = "BUILDING GROUP (B): type the number -- _"
        cv2.putText(win_out, msg_b, (30, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(win_out, "ENTER confirms -- BACKSPACE deletes digit/exits -- [B] cancels",
                    (30, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_one_crack_marker(self, win_out, f, marker_dim):
        """Draws one crack's two endpoint markers (filled if soft-
        deleted, crosshair if active)."""
        symbol = "-" if f.get('active', True) else "+"
        for k_p in ['start', 'end']:
            if k_p not in f:
                continue
            kp_x, kp_y = f.get(k_p)[0], f.get(k_p)[1]
            wx, wy = self.transform_real_to_window_coords(int(kp_x), int(kp_y))
            if not (0 <= wx < 1200 and 0 <= wy < 900):
                continue
            cv2.rectangle(win_out, (wx-marker_dim, wy-marker_dim), (wx+marker_dim, wy+marker_dim), (0, 255, 255), 2)
            if symbol == "-":
                inner_dim = int(marker_dim * 0.4)
                cv2.rectangle(win_out, (wx-inner_dim, wy-inner_dim), (wx+inner_dim, wy+inner_dim), (0, 255, 255), -1)
            elif symbol == "+":
                cv2.line(win_out, (wx-marker_dim+5, wy), (wx+marker_dim-5, wy), (0, 255, 255), 2)
                cv2.line(win_out, (wx, wy-marker_dim+5), (wx, wy+marker_dim-5), (0, 255, 255), 2)

    def _draw_crack_markers(self, win_out, marker_dim):
        """Draws every saved crack's start/end markers."""
        for f in self.cfg.saved_cracks:
            self._draw_one_crack_marker(win_out, f, marker_dim)

    def _draw_temp_crack_start_marker(self, win_out, marker_dim):
        """Draws the marker at an in-progress crack's start point,
        before its second click."""
        if not (self.cfg.temp_start is not None and self.cfg.current_tool == 'crack'):
            return
        ts_x, ts_y = self.cfg.temp_start[0], self.cfg.temp_start[1]
        wx, wy = self.transform_real_to_window_coords(int(ts_x), int(ts_y))
        if 0 <= wx < 1200 and 0 <= wy < 900:
            cv2.rectangle(win_out, (wx-marker_dim, wy-marker_dim), (wx+marker_dim, wy+marker_dim), (0, 165, 255), 2)

    def _draw_hover_snap_indicator(self, win_out):
        """Draws the pulsing crosshair over a crack's end while it's the
        target of arrow-key nudging."""
        if not (self.cfg.current_tool == 'crack' and self.cfg.temp_start is None and self.cfg.hover_extension_index is not None):
            return
        target_f = self.cfg.saved_cracks[self.cfg.hover_extension_index]
        pt_real = target_f['path'][0] if self.cfg.hover_extension_mode == 'start' else target_f['path'][-1]
        wx_snap, wy_snap = self.transform_real_to_window_coords(int(pt_real[0]), int(pt_real[1]))
        if not (0 <= wx_snap < 1200 and 0 <= wy_snap < 900):
            return
        pulsing_color = (0, 255, 0) if int(time.time() * 4) % 2 == 0 else (0, 165, 255)
        cv2.circle(win_out, (wx_snap, wy_snap), 12, pulsing_color, 2, cv2.LINE_AA)
        cv2.circle(win_out, (wx_snap, wy_snap), 3, pulsing_color, -1)
        cv2.line(win_out, (wx_snap - 20, wy_snap), (wx_snap - 6, wy_snap), pulsing_color, 1)
        cv2.line(win_out, (wx_snap + 6, wy_snap), (wx_snap + 20, wy_snap), pulsing_color, 1)
        cv2.line(win_out, (wx_snap, wy_snap - 20), (wx_snap, wy_snap - 6), pulsing_color, 1)
        cv2.line(win_out, (wx_snap, wy_snap + 6), (wx_snap, wy_snap + 20), pulsing_color, 1)

        s_scale = 0.9 if self.cfg.hud_large_size else 0.45
        s_thick = 2 if self.cfg.hud_large_size else 1
        cv2.putText(win_out, f"SNAPPED: Crack #{self.cfg.hover_extension_index}", (wx_snap + 15, wy_snap - 10), cv2.FONT_HERSHEY_SIMPLEX, s_scale, pulsing_color, s_thick, cv2.LINE_AA)

    def _draw_tool_markers(self, win_out, marker_dim):
        """Draws every marker/indicator gated by [Space]/show_markers,
        in their original stacking order."""
        if not self.cfg.show_markers:
            return
        self._draw_detachment_markers(win_out, marker_dim)
        self._draw_in_progress_detachment_trace(win_out, marker_dim)
        self._draw_edit_mode_indicator(win_out)
        self._draw_translate_mode_indicator(win_out)
        self._draw_width_edit_hud(win_out)
        self._draw_manual_group_entry_banner(win_out)
        self._draw_crack_markers(win_out, marker_dim)
        self._draw_temp_crack_start_marker(win_out, marker_dim)
        self._draw_hover_snap_indicator(win_out)

    def _update_total_cracks_length(self):
        """Recomputes the total crack length (cm) fresh from the current saved_cracks -- NOT an accumulator, since this runs every frame.
        """
        total_crack_points = sum(len(f['path']) for f in self.cfg.saved_cracks if f.get('active', True) and 'path' in f)
        self.cfg.total_cracks_length_cm = total_crack_points * self.cfg.PIXEL_TO_CM_SCALE

    def _compute_hud_font_metrics(self):
        """Returns the font scale/thickness/y-position constants shared
        by most HUD text, doubled in large-HUD-size mode."""
        large = self.cfg.hud_large_size
        return {
            'f_scale': 1.2 if large else 0.65,
            'f_scale_btm': 0.9 if large else 0.55,
            'f_thick': 3 if large else 2,
            'f_thick_btm': 2 if large else 1,
            'y_pos1': 45 if large else 30,
            'y_pos2': 90 if large else 60,
            'y_pos3': 135 if large else 90,
            'line_h': 45 if large else 30,
        }

    def _compute_crack_reliability(self, path):
        """Fraction of this crack's path points where the automatic edge
        detector (see _auto_detect_crack_edge_offsets) found real
        crack-like texture in binary_mask, as a percentage -- how much
        of the traced line is corroborated by the underlying photo,
        rather than just drawn freehand by the operator. None if there's
        no path or no binary_mask to check against."""
        if not path or self.cfg.binary_mask is None:
            return None
        n = len(path)
        if n == 0:
            return None
        height, width = self.cfg.binary_mask.shape[:2]
        max_offset = self.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX
        left_offsets, right_offsets = self._auto_detect_crack_edge_offsets(path, self.cfg.binary_mask, height, width, max_offset)
        corroborated = sum(1 for i in range(n) if left_offsets[i] > 0 or right_offsets[i] > 0)
        return 100.0 * corroborated / n

    def _build_crack_reliability_lines(self):
        """One "Crack N: XX%" entry per active crack with a computable
        reliability score, grouped a few per line for the info panel."""
        entries = []
        for i, f in enumerate(self.cfg.saved_cracks, start=1):
            if not f.get('active', True):
                continue
            pct = self._compute_crack_reliability(f.get('path'))
            if pct is not None:
                entries.append(f"Crack {i}: {pct:.0f}%")
        per_line = 5
        return [" | ".join(entries[i:i + per_line]) for i in range(0, len(entries), per_line)]

    def _draw_info_overlay(self, win_out, status, metrics_str, hm):
        """Draws the FILE/Cracks-Length status panel, toggled by [J], off
        by default -- reserving a line for the Building status (drawn
        separately, right after this, by draw_building_hud_overlay())
        before the "Crack Reliability:" label and per-crack score lines."""
        if not self.cfg.show_info_overlay:
            return
        line_h = hm['line_h']
        building_line = 1 if self.cfg.CURRENT_IMAGE_PATH is not None else 0
        reliability_lines = self._build_crack_reliability_lines()
        extra_lines = building_line + len(reliability_lines) + (1 if reliability_lines else 0)  # +1 for the label line
        base_bottom = 205 if self.cfg.hud_large_size else 160
        panel_bottom = base_bottom + line_h * extra_lines
        overlay_info = win_out.copy()
        cv2.rectangle(overlay_info, (8, 6), (760, panel_bottom), (35, 30, 25), -1)
        cv2.addWeighted(overlay_info, 0.55, win_out, 0.45, 0, win_out)
        cv2.putText(win_out, status, (15, hm['y_pos1']), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (235, 235, 235), hm['f_thick'], cv2.LINE_AA)
        cv2.putText(win_out, metrics_str, (15, hm['y_pos2']), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (170, 220, 255), hm['f_thick'], cv2.LINE_AA)
        if reliability_lines:
            label_y = hm['y_pos2'] + line_h * (building_line + 1)
            cv2.putText(win_out, "Crack Reliability:", (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (170, 255, 200), hm['f_thick'], cv2.LINE_AA)
            for i, line in enumerate(reliability_lines):
                y = label_y + line_h * (i + 1)
                cv2.putText(win_out, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (170, 255, 200), hm['f_thick'], cv2.LINE_AA)

    def _resolve_building_hud_y(self, hm):
        """Y-position of the Building status line: right after metrics_str,
        before the Crack Reliability section (see _draw_info_overlay)."""
        return hm['y_pos2'] + hm['line_h']

    def _draw_calibration_indicator(self, win_out, hm):
        """Draws the [L] calibration-mode status text and anchor dot."""
        if not self.cfg.calibration_mode:
            return
        cv2.putText(win_out, "L-MODE: CALIBRATION ACTIVE", (15, hm['y_pos3']), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (0, 165, 255), hm['f_thick'], cv2.LINE_AA)
        if self.cfg.calib_start is not None:
            wx1, wy1 = self.transform_real_to_window_coords(int(self.cfg.calib_start[0]), int(self.cfg.calib_start[1]))
            cv2.circle(win_out, (wx1, wy1), 6, (0, 165, 255), -1)

    def _draw_session_timer(self, win_out, hm):
        """Draws the session timer, or a blinking paused banner.
        Returns the time string shown (None while paused)."""
        if not self.cfg.timer_is_paused:
            time_str = f"Session time: {int(self.cfg.total_elapsed_paused_time + (time.time() - self.cfg.image_load_time))}s"
            time_x = 950 if self.cfg.hud_large_size else 1000
            overlay_time = win_out.copy()
            cv2.rectangle(overlay_time, (time_x - 10, 8), (1190, 40), (30, 28, 25), -1)
            cv2.addWeighted(overlay_time, 0.55, win_out, 0.45, 0, win_out)
            cv2.putText(win_out, time_str, (time_x, 30), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (235, 235, 235), hm['f_thick_btm'], cv2.LINE_AA)
            return time_str
        else:
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(win_out, (850 if self.cfg.hud_large_size else 930, 12), (1185, 42), (20, 20, 160), -1)
                cv2.putText(win_out, "STOPWATCH PAUSED", (870 if self.cfg.hud_large_size else 945, 32), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (255, 255, 255), hm['f_thick_btm'], cv2.LINE_AA)
            return None

    def _draw_timeout_and_warp_banners(self, win_out, hm):
        """Draws the A* timeout and homography-warp status banners."""
        if self.cfg.pathfinding_timeout_triggered:
            cv2.rectangle(win_out, (720, 15), (1180, 75), (20, 20, 160), -1)
            cv2.putText(win_out, "TIMEOUT: Path Unreachable", (745, 52), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale'], (255, 255, 255), hm['f_thick'], cv2.LINE_AA)
        if self.cfg.warp_jitter_triggered:
            cv2.rectangle(win_out, (720, 85), (1180, 145), (30, 105, 210), -1)
            cv2.putText(win_out, "WARP SYNC: Prospect Matched", (735, 122), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (255, 255, 255), hm['f_thick_btm'], cv2.LINE_AA)

    def _draw_import_warning_banner(self, win_out, hm):
        """Draws a failed W/L import banner for 6 seconds, then
        auto-dismisses it."""
        corrente_time = time.time()
        if self.cfg.import_warning_triggered and (corrente_time - self.cfg.import_warning_timestamp < 6.0):
            cv2.rectangle(win_out, (15, 180), (1185, 240), (0, 60, 220), -1)
            cv2.putText(win_out, self.cfg.import_warning_message, (35, 218), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (255, 255, 255), hm['f_thick_btm'], cv2.LINE_AA)
        elif self.cfg.import_warning_triggered:
            self.cfg.import_warning_triggered = False

    def _draw_save_status_banner(self, win_out, show_error_banner, error_time, hm):
        """Draws the save-error or export-complete banner, whichever is
        currently within its display window."""
        corrente_time = time.time()
        if show_error_banner and (corrente_time - error_time < 4.0):
            cv2.rectangle(win_out, (15, 110), (1185, 170), (20, 20, 200), -1)
            error_msg = "Cannot save file: To modify segmented images, use Option 2 at application startup"
            cv2.putText(win_out, error_msg, (35, 148), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (255, 255, 255), hm['f_thick_btm'], cv2.LINE_AA)
        elif not show_error_banner and (corrente_time - self.cfg.save_timestamp < 3.0):
            cv2.rectangle(win_out, (30, 110), (480, 170), (30, 30, 30), -1)
            cv2.putText(win_out, "Export complete: JSON + IMG" if self.cfg.SAVE_SEG_IMAGE else "Export complete: JSON ONLY", (50, 148), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (0, 200, 0), hm['f_thick_btm'], cv2.LINE_AA)

    def _draw_bottom_hint_bar(self, win_out, hm):
        """Draws the always-visible bottom-of-screen key hint bar."""
        istruzioni = "[?] Help Menu | [ENTER/Q] Next | [S] Save | [F] Fullscreen | [X] Clear All"
        cv2.putText(win_out, istruzioni, (15, 875), cv2.FONT_HERSHEY_SIMPLEX, hm['f_scale_btm'], (255, 255, 0), hm['f_thick_btm'], cv2.LINE_AA)

    def _draw_help_menu_status_lines(self, win_out, status, metrics_str, time_str):
        """Draws the help guide's system-status section."""
        cv2.putText(win_out, f"--- SYSTEM STATUS ---", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 210, 110), 2, cv2.LINE_AA)
        cv2.putText(win_out, status, (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(win_out, metrics_str, (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(win_out, time_str if not self.cfg.timer_is_paused else "STOPWATCH PAUSED", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)

    def _help_menu_commands(self):
        """Returns the full help-guide command list text, in display
        order."""
        return [
            "[?] : Close / Open this guide",
            "[BACKSPACE / DELETE] : Emergency system unlock",
            "[Q] or [ENTER] : Save file and move to the next one",
            "[S] : Save current state without changing image",
            "[X] : Clear all (full reset of the canvas and the JSON)",
            "[C] : Crack tool | [D] : Detachment tool",
            "[Y] : Close detachment polygon smartly",
            "[E] : Toggle Edit Mode (user node editing)",
            "[U] : Undo last action | [R] : Redo action",
            "[AUTO] Building Group assigned automatically by comparing each new photo with those already grouped",
            "[B] : Manually correct the Building Group (wrong/missed automatic assignment)",
            "[W] : Project fractures from a previous file (same building)",
            "[T] : Automatically snap an imported crack to the real edge (A*, one at a time)",
            "[V] : Bulk-retrace ALL imported cracks onto the real edge (with a plausibility check)",
            "[AUTO] After every save (S/Q/ENTER), automatic crack comparison with the group's other photos (if >=2 photos)",
            "[G] : Manually repeat the building group's crack comparison/discard (e.g. after correcting the group with B)",
            "[A] : Automatically fill a crack tract (2 clicks) and refine with [+]/[-] (both sides) or [/{ ]/} (one side at a time) for the training mask",
            "[F] : Fullscreen (ON/OFF)",
            "[P] : Pause the timer",
            "[M] : Toggle PNG visual mask export",
            "[H] : Enlarge/Shrink text interface",
            "[Space] : Show/Hide on-screen markers",
            "[N] : Show/Hide blue fill of segmented cracks",
            "[J] : Show/Hide FILE/Cracks Length/BUILDING status text"
        ]

    def _draw_help_menu_command_list(self, win_out, comandi):
        """Draws only the current scroll window of the command list.
        Returns (top, bottom, max_offset, offset) for the scrollbar."""
        content_top, content_bottom = 300, 830
        line_height = 32
        max_visible = max(1, (content_bottom - content_top) // line_height)
        self.cfg.help_scroll_visible_lines = max_visible
        max_offset = max(0, len(comandi) - max_visible)
        self.cfg.help_scroll_max_offset = max_offset
        self.cfg.help_scroll_offset = min(max(0, self.cfg.help_scroll_offset), max_offset)
        offset = self.cfg.help_scroll_offset
        visible_comandi = comandi[offset:offset + max_visible]

        y_cmd = 310
        for cmd in visible_comandi:
            colore = (210, 210, 210)
            if "BACKSPACE" in cmd or "Clear all" in cmd: colore = (90, 90, 235)
            elif "[?]" in cmd: colore = (110, 210, 110)
            cv2.putText(win_out, cmd, (80, y_cmd), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore, 1, cv2.LINE_AA)
            y_cmd += line_height
        return content_top, content_bottom, max_offset, offset

    def _draw_help_menu_scrollbar(self, win_out, content_top, content_bottom, max_offset, offset, n_commands):
        """Draws the draggable scrollbar track+thumb; also stores its
        bounds so mouse_callback can recognize a click on it."""
        if max_offset <= 0:
            self.cfg.help_scrollbar_rect = None
            return
        track_x0, track_x1 = 1120, 1135
        self.cfg.help_scrollbar_rect = (track_x0, content_top, track_x1, content_bottom)
        cv2.rectangle(win_out, (track_x0, content_top), (track_x1, content_bottom), (60, 55, 50), -1)
        thumb_h = max(20, int((content_bottom - content_top) * (self.cfg.help_scroll_visible_lines / n_commands)))
        thumb_travel = (content_bottom - content_top) - thumb_h
        thumb_y0 = content_top + (int(thumb_travel * (offset / max_offset)) if max_offset else 0)
        cv2.rectangle(win_out, (track_x0, thumb_y0), (track_x1, thumb_y0 + thumb_h), (160, 175, 190), -1)
        hint = "[Up]/[Down] or mouse wheel to scroll"
        cv2.putText(win_out, hint, (80, 862), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

    def _draw_help_menu_panel(self, win_out):
        """Draws the translucent "glass" background panel behind the
        help guide's text."""
        overlay_help = win_out.copy()
        cv2.rectangle(overlay_help, (50, 50), (1150, 850), (30, 28, 25), -1)
        cv2.addWeighted(overlay_help, 0.60, win_out, 0.40, 0, win_out)

    def _draw_help_menu(self, win_out, status, metrics_str, time_str):
        """Draws the whole [?] on-screen command guide."""
        if not self.cfg.show_help_menu:
            return
        self._draw_help_menu_panel(win_out)
        self._draw_help_menu_status_lines(win_out, status, metrics_str, time_str)
        cv2.putText(win_out, f"--- QUICK COMMANDS ---", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 175, 235), 2, cv2.LINE_AA)
        comandi = self._help_menu_commands()
        content_top, content_bottom, max_offset, offset = self._draw_help_menu_command_list(win_out, comandi)
        self._draw_help_menu_scrollbar(win_out, content_top, content_bottom, max_offset, offset, len(comandi))

    def _composite_and_display(self, win_out):
        """Letterboxes win_out onto the real window's exact size and
        pushes it to the display."""
        try:
            _, _, w_win, h_win = cv2.getWindowImageRect("Crack Detector Workspace")
            if w_win > 100 and h_win > 100:
                canvas_adattivo = np.zeros((h_win, w_win, 3), dtype=np.uint8)
                h_roi, w_roi = win_out.shape[:2]
                scala_proporzionale = min(w_win / float(w_roi), h_win / float(h_roi))
                w_nuovo = int(w_roi * scala_proporzionale)
                h_nuovo = int(h_roi * scala_proporzionale)

                img_riscalata = cv2.resize(win_out, (w_nuovo, h_nuovo), interpolation=cv2.INTER_LINEAR)

                x_offset = (w_win - w_nuovo) // 2
                y_offset = (h_win - h_nuovo) // 2
                canvas_adattivo[y_offset:y_offset+h_nuovo, x_offset:x_offset+w_nuovo] = img_riscalata

                self._display_frame("Crack Detector Workspace", canvas_adattivo)
            else:
                self._display_frame("Crack Detector Workspace", win_out)
        except Exception:
            self._display_frame("Crack Detector Workspace", win_out)

    def _compute_scene_overlays(self, scene, h_actual, w_active):
        """Draws crack/detachment overlays onto the scene, before the
        zoom crop, so they stay aligned at any zoom level."""
        self.cfg.blue_visual_mask = self._ensure_visual_mask_shape(self.cfg.blue_visual_mask, h_actual, w_active)
        self.cfg.green_visual_mask = self._ensure_visual_mask_shape(self.cfg.green_visual_mask, h_actual, w_active)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        self._paint_blue_crack_overlay(scene, k, h_actual, w_active)
        self._paint_green_detachment_overlay(scene, k)
        self._paint_width_edit_live_preview(scene, h_actual, w_active)

    def _build_window_frame(self):
        """Builds the scene, paints overlays, crops to zoom, and resizes
        to the window. Returns win_out for the HUD drawing that follows."""
        scene = self.cfg.img_background.copy()
        self._paint_temp_crack_preview(scene)
        h_actual, w_active = scene.shape[:2]
        self._compute_scene_overlays(scene, h_actual, w_active)
        roi = self._compute_zoom_roi(scene, h_actual, w_active)
        return self._resize_roi_to_window(roi)

    def _draw_hud_stack(self, win_out, current_idx, total_count, show_error_banner, error_time):
        """Draws every window-space HUD element, in their original
        stacking order. Returns (status, metrics_str, time_str)."""
        marker_dim = 22
        self._update_total_cracks_length()
        total_detachments_area_cm2 = sum(self.calcola_area_poligono_cm2(d['path'], self.cfg.PIXEL_TO_CM_SCALE) for d in self.cfg.saved_detachments if d.get('active', True))

        self._draw_tool_markers(win_out, marker_dim)

        hm = self._compute_hud_font_metrics()
        status = f"FILE: {current_idx}/{total_count} ({os.path.basename(self.cfg.CURRENT_IMAGE_PATH)}) | Tool: {self.cfg.current_tool.upper()} | Zoom: {self.cfg.zoom_factor:.1f}x"
        metrics_str = f"Cracks Length: {self.cfg.total_cracks_length_cm:.2f} cm | Detachments Area: {total_detachments_area_cm2:.2f} cm2"

        self._draw_info_overlay(win_out, status, metrics_str, hm)
        self._draw_calibration_indicator(win_out, hm)
        time_str = self._draw_session_timer(win_out, hm)
        self._draw_timeout_and_warp_banners(win_out, hm)
        self._draw_import_warning_banner(win_out, hm)
        self._draw_save_status_banner(win_out, show_error_banner, error_time, hm)
        self._draw_bottom_hint_bar(win_out, hm)
        self.draw_building_hud_overlay(win_out, self._resolve_building_hud_y(hm))
        self._draw_help_menu(win_out, status, metrics_str, time_str)
        return status, metrics_str, time_str

    def render_scene(self, current_idx, total_count, show_error_banner=False, error_time=0.0):
        """Builds and downsamples current crop frame pushing overlay
        indicators onto the view matrix."""
        try:
            win_out = self._build_window_frame()
            self._draw_hud_stack(win_out, current_idx, total_count, show_error_banner, error_time)
            self._composite_and_display(win_out)
        except Exception as e:
            print(f"[DISPLAY EXCEPTION] Error: {e}", file=sys.stderr)

    def _handle_manual_group_entry_keypress(self, key_clean):
        """[B] digit-entry mode: captures digits/ENTER/Backspace here,
        before their normal meaning below."""
        if key_clean in (13, 10):
            self._confirm_manual_building_group_entry()
        elif key_clean in (8, 127):
            if self.cfg.manual_group_entry_state["buffer"]:
                self.cfg.manual_group_entry_state["buffer"] = self.cfg.manual_group_entry_state["buffer"][:-1]
            else:
                self.cfg.manual_group_entry_state["active"] = False
                print("[MANUAL GROUP] Entry cancelled.")
        elif key_clean in (ord('b'), ord('B')):
            self.cfg.manual_group_entry_state["active"] = False
            self.cfg.manual_group_entry_state["buffer"] = ""
            print("[MANUAL GROUP] Entry cancelled.")
        elif 48 <= key_clean <= 57:
            if len(self.cfg.manual_group_entry_state["buffer"]) < 6:
                self.cfg.manual_group_entry_state["buffer"] += chr(key_clean)
        self.refresh_zoom_viewport()

    def _handle_escape_key(self):
        """[ESC]: exits the whole program cleanly."""
        print("[INFO] Cleanly terminated by user (ESC).")
        sys.exit(0)

    def _do_save_and_archive(self):
        """Shared by [Q]/[Enter] and [S]: exports the JSON/masks/CSV,
        archives the session, then runs the cross-photo crack check."""
        self.export_labelme_format()
        self.archive_current_session_to_processed()
        self.run_post_save_group_crack_check()

    def _handle_save_and_next_key(self):
        """[Q]/[Enter]: saves the current file and moves to the next
        image."""
        print("\n[Command Received] Closing and saving current file...")
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["idx"] = None
        self.cfg.translate_state["start"] = None
        self._do_save_and_archive()
        return True

    def _handle_save_key(self):
        """[S]: saves without moving to another image."""
        self._do_save_and_archive()
        return True

    def _handle_manual_align_key(self):
        """[L]: manually triggers the same homography alignment that
        normally runs automatically on load."""
        print("\n[MANUAL ALIGN] Starting homography-based alignment on request...")
        try:
            self.auto_import_best_previous_session()
            print("[MANUAL ALIGN SUCCESS] Fractures loaded and projected successfully!")
        except Exception as e:
            print(f"[MANUAL ALIGN ERROR] Could not load the alignment: {e}")

    def _redo_restore_crack_or_detachment(self, action_type, restored):
        """Redo case: "add this back" -- a crack/detachment that was
        removed (by Undo or by click-to-delete) is restored."""
        if action_type in ['crack', 'crack_delete', 'crack_new']:
            self.cfg.saved_cracks.append(restored)
            self.cfg.action_history.append('crack')
            print("[REDO] Crack restored.")
        elif action_type in ['detachment', 'detachment_delete']:
            self.cfg.saved_detachments.append(restored)
            self.cfg.action_history.append('detachment')
            print("[REDO] Detachment restored.")

    def _redo_redelete_crack_or_detachment(self, action_type, restored):
        """Redo case: "remove this again" -- reverses an Undo that had
        just brought a deleted crack/detachment back."""
        if action_type == 'crack_redelete':
            for i, c in enumerate(self.cfg.saved_cracks):
                if c is restored:
                    self.cfg.saved_cracks.pop(i)
                    break
            self.cfg.action_history.append(('crack_delete', restored))
            print("[REDO] Crack deleted again.")
        elif action_type == 'detachment_redelete':
            for i, d in enumerate(self.cfg.saved_detachments):
                if d is restored:
                    self.cfg.saved_detachments.pop(i)
                    break
            self.cfg.action_history.append(('detachment_delete', restored))
            print("[REDO] Detachment deleted again.")

    def _redo_clear_all(self, backup):
        """Redo case: re-applies a [X] Clear All that was just undone."""
        self.cfg.saved_cracks.clear()
        self.cfg.saved_detachments.clear()
        self.cfg.action_history.append(('clear_all', backup))
        self._reset_json_shapes_on_disk()
        print("[REDO] Canvas cleared again.")

    def _handle_redo_key(self):
        """[R]: redo-only (building-group assignment is now automatic,
        see auto_assign_building_group_by_similarity)."""
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["start"] = None
        self.cfg.translate_state["idx"] = None

        if not self.cfg.redo_history:
            print("[REDO] No action to restore in this image's history.")
            return
        action_type, restored = self.cfg.redo_history.pop()
        if action_type == 'clear_all':
            self._redo_clear_all(restored)
        else:
            self._redo_restore_crack_or_detachment(action_type, restored)
            self._redo_redelete_crack_or_detachment(action_type, restored)
        self.recalculate_masks()
        self.refresh_zoom_viewport()

    def _handle_manual_group_override_key(self):
        """[B]: lets the operator type any building-group number by
        hand, overriding the automatic on-load assignment."""
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["start"] = None
        self.cfg.translate_state["idx"] = None
        self.start_manual_building_group_entry()
        self.refresh_zoom_viewport()

    def _handle_w_import_key(self):
        """[W]: projects fractures from the best matching previous
        session onto the current photo."""
        print("\n[W-COMMAND] Scanning 'segmentated images' folder...")
        base_img, orig_json = self.find_last_processed_session_asset()
        if base_img and orig_json:
            success = self.warp_and_adapt_json_to_new_image(
                base_img, self.cfg.CURRENT_IMAGE_PATH, orig_json, self.cfg.JSON_OUTPUT_PATH
            )
            if success:
                self.load_labelme_format()
                cv2.setWindowProperty(self._window_name, cv2.WND_PROP_TOPMOST, 1)
        else:
            print("[WARNING] Cannot project: no previous asset found.")
            self._signal_import_warning("IMPORT NOT PERFORMED: no previous session found to project.")

    def _handle_navigate_without_saving(self, direction):
        """Codes 4/5 (Next/Previous Image buttons): moves the queue
        without exporting/archiving anything."""
        verb = "next" if direction == "next" else "previous"
        print(f"\n[NAVIGATE] Moving to the {verb} image WITHOUT saving.")
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["idx"] = None
        self.cfg.translate_state["start"] = None
        self.cfg.navigate_direction = direction
        return True

    def _handle_switch_mode_key(self):
        """Code 6 (Switch Mode button): exits the current image's loop
        without saving, flagging run()'s outer loop to rebuild image_queue
        for the other mode (1 <-> 2) instead of just advancing."""
        print("\n[SWITCH MODE] Switching data source without restarting the application...")
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["idx"] = None
        self.cfg.translate_state["start"] = None
        self.cfg.switch_mode_requested = True
        return True

    def process_keypress(self, key_raw):
        """Handles a single key press, real or synthetic (from a GUI button/menu, see _pending_keys). Returns True if the caller should advance to the next image, False otherwise.
        """
        key_clean = key_raw & 0xFF

        if self.cfg.manual_group_entry_state["active"]:
            self._handle_manual_group_entry_keypress(key_clean)
            return False

        if key_raw == 27:
            self._handle_escape_key()

        elif key_clean in [13, 10, ord('q'), ord('Q')]:
            return self._handle_save_and_next_key()

        elif key_clean in [ord('s'), ord('S')]:
            return self._handle_save_key()

        elif key_clean in [ord('l'), ord('L')]:
            self._handle_manual_align_key()

        elif key_clean in [ord('r'), ord('R')]:
            self._handle_redo_key()

        elif key_clean in [ord('b'), ord('B')]:
            self._handle_manual_group_override_key()

        elif key_clean in [ord('w'), ord('W')]:
            self._handle_w_import_key()

        elif key_clean == 4:
            return self._handle_navigate_without_saving("next")

        elif key_clean == 5:
            return self._handle_navigate_without_saving("previous")

        elif key_clean == 6:
            return self._handle_switch_mode_key()

        else:
            self.handle_keyboard(key_raw)

        return False

    def _create_display_window(self, window_name):
        """Creates/configures the main OpenCV display window. A GUI wrapper overrides this to keep it alive off-screen for geometry queries while displaying frames in its own embedded widget instead.
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(window_name, self.mouse_callback)

    def _display_frame(self, window_name, frame):
        """Shows one fully-composited frame. Default: cv2.imshow(). A GUI wrapper overrides this to draw into its own embedded widget instead.
        """
        cv2.imshow(window_name, frame)

    def _initialize_run_session(self):
        """Steps 0-2 of run(): GSD edge width, resumed building index,
        the image queue, and the display window. Returns total_files."""
        # A no-op while IMAGE_GSD_MM_PER_PX is unset; resolves before anything else runs.
        self.resolve_crack_edge_width_from_gsd()

        # Retrieves the last building index used, on restart
        self.cfg.CURRENT_BUILDING_INDEX = self.resume_last_building_index()
        if self.cfg.CURRENT_BUILDING_INDEX is not None:
            print(f"[RESUME] Restored last building group: BLDG{self.cfg.CURRENT_BUILDING_INDEX:03d}")

        # 1. Initializes the image queue
        self.build_chronological_queue()

        # 2. Configures the graphics window (fullscreen by default)
        window_name = "Crack Detector Workspace"
        self._create_display_window(window_name)
        self._window_name = window_name

        return len(self.cfg.image_queue)

    def _sync_queue_path_after_rename(self, queue_pos):
        """Keeps the queue's path in sync after initialize_image_session() may have auto-renamed the file on disk (building-group tagging)."""
        self.cfg.image_queue[queue_pos] = self.cfg.CURRENT_IMAGE_PATH

    def _start_image_session(self, queue_pos, index, total_files):
        """Loads one image into the working session and renders its first frame.
        """
        img_path = self.cfg.image_queue[queue_pos]
        self.initialize_image_session(img_path)
        self._sync_queue_path_after_rename(queue_pos)

        self.render_scene(index, total_files)
        self._wait_key(1)
        self._pump_extra_events()

    def _reset_per_image_timer_state(self):
        self.cfg.image_load_time = time.time()
        self.cfg.total_elapsed_paused_time = 0.0
        self.cfg.timer_is_paused = False

    def _drain_pending_keys(self):
        """Dispatches any synthetic key codes a GUI button queued through process_keypress(). Returns True if one triggered advancing images."""
        next_file_triggered = False
        while self._pending_keys and not next_file_triggered:
            next_file_triggered = self.process_keypress(self._pending_keys.pop(0))
        return next_file_triggered

    def _run_single_image_loop(self, index, total_files):
        """The interactive per-image loop: renders and waits for a keystroke, until one signals it's time to advance to another image."""
        image_save_counter = 0
        SHOW_MODE_ERROR = False
        next_file_triggered = False

        while not next_file_triggered:
            self.render_scene(index, total_files, show_error_banner=SHOW_MODE_ERROR, error_time=self.cfg.save_timestamp)

            # Pumps any external GUI panel's event loop and dispatches any
            # synthetic key codes it queued, through the same process_keypress().
            self._pump_extra_events()
            next_file_triggered = self._drain_pending_keys()
            if next_file_triggered:
                break

            key_raw = self._wait_key(30)
            if key_raw != -1:
                next_file_triggered = self.process_keypress(key_raw)

    def _advance_queue_position(self, queue_pos):
        """Chooses which direction to move for the next iteration: backward only for the Previous Image action, forward otherwise."""
        if self.cfg.navigate_direction == "previous":
            queue_pos = max(0, queue_pos - 1)
        else:
            queue_pos += 1
        self.cfg.navigate_direction = None
        return queue_pos

    def _advance_or_switch_mode(self, queue_pos, total_files):
        """After each image's loop: rebuilds image_queue for the other
        mode if Switch Mode was requested (code 6), otherwise advances
        queue_pos normally. Returns the (queue_pos, total_files) to
        continue run()'s outer loop with."""
        if self.cfg.switch_mode_requested:
            self.cfg.switch_mode_requested = False
            target_mode = "2" if self.cfg.modalita_scelta == "1" else "1"
            if self._rebuild_queue_for_mode(target_mode):
                queue_pos = 0
            return queue_pos, len(self.cfg.image_queue)
        return self._advance_queue_position(queue_pos), total_files

    def _write_final_csv_report(self):
        """Step 4 of run(): appends every buffered report_data_summary
        entry to the CSV, once, at the end of the whole session."""
        if not self.cfg.report_data_summary:
            return
        csv_output_path = os.path.join(self.cfg.SCRIPT_DIR, "segmentation_summary_report.csv")
        try:
            f_exists = os.path.exists(csv_output_path)
            with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=['filename', 'cracks_count', 'cracks_length_cm', 'detachments_count', 'detachments_area_cm2', 'scale_used', 'time_spent_minutes']
                )
                if not f_exists:
                    writer.writeheader()
                for entry in self.cfg.report_data_summary:
                    writer.writerow(entry)
            print(f"\n[AUTOMATED REPORT] Report updated! Path: {csv_output_path}")
        except Exception as csv_error:
            print(f"[REPORT EXCEPTION] Error writing CSV: {csv_error}")

    def _finalize_run(self):
        """The finally block: closes every window and gives it a
        moment to actually go away before exiting."""
        cv2.destroyAllWindows()
        for _ in range(5):
            self._wait_key(1)
        print("Workspace safely closed. Exiting.")

    def run(self):
        try:
            total_files = self._initialize_run_session()

            # A manually-advanced index (not a for-loop) so Previous Image can move queue_pos backward.
            queue_pos = 0
            while 0 <= queue_pos < len(self.cfg.image_queue):
                index = queue_pos + 1
                self._start_image_session(queue_pos, index, total_files)
                self._reset_per_image_timer_state()
                self._run_single_image_loop(index, total_files)
                queue_pos, total_files = self._advance_or_switch_mode(queue_pos, total_files)

            print("\n--- PIPELINE EXHAUSTED ---")

            # 4. CSV report generation
            self._write_final_csv_report()

        except Exception as e:
            print(f"[MAIN THREAD CRASH] Interruption: {e}", file=sys.stderr)

        finally:
            self._finalize_run()


if __name__ == "__main__":
    CrackSegmentation().run()
