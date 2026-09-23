"""config.py (object-oriented port): holds every configuration constant and runtime-state attribute the segmentation pipeline needs, as attributes of a single Config class passed around explicitly.
"""
import os
import sys
from pathlib import Path


def resolve_script_dir():
    """Returns the app's data folder (Images, segmentated images, already processed images, CSV report): next to the script when run from source, or ~/Documents/CrackSegmentation when packaged/frozen.
    """
    if getattr(sys, 'frozen', False):
        base = Path.home() / "Documents" / "CrackSegmentation"
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(__file__).resolve().parent


class Config:
    """Holds every configuration constant and mutable runtime-state
    attribute the segmentation pipeline needs."""

    def _init_paths(self):
        """Resolves operating directory paths based on the script location (or a per-user data folder when packaged -- see resolve_script_dir())."""
        self.SCRIPT_DIR = resolve_script_dir()
        self.IMAGE_FOLDER = self.SCRIPT_DIR  # Scans the active folder where the script resides
        self.folder_seg_img = 'segmentated images'
        self.OUTPUT_FOLDER = os.path.join(self.IMAGE_FOLDER, self.folder_seg_img)

    def _init_session_pointers(self):
        """Global runtime pointers modified dynamically at the beginning
        of each image loop."""
        self.CURRENT_IMAGE_PATH = ""
        self.JSON_OUTPUT_PATH = ""
        self.EXPORT_IMAGE_PATH = ""
        # Cache of the current image's raw bytes, reused for the base64 export payload.
        self.CURRENT_IMAGE_RAW_BYTES = None

    def _init_display_state(self):
        """The current image's arrays and the viewport's zoom/pan
        state."""
        self.is_fullscreen = True
        self.img_original = None
        self.img_background = None
        self.binary_mask = None
        self.skeleton_mask = None
        self.H_img, self.W_img = 0, 0

        # Workspace States and Viewport slicing parameters
        self.zoom_factor = 1.0
        self.zoom_center = [0, 0]
        self.zoom_box = [0, 0, 0, 0]

    def _init_tool_and_overlay_state(self):
        """The active tool, and every on-screen overlay's visibility/
        scroll state."""
        self.current_tool = 'crack'
        self.show_markers = True
        # Independent visibility toggle for the blue crack-overlay fill ([N] key).
        self.show_crack_overlay = True
        # FILE/Cracks-Length/BUILDING status text, off by default, toggled with [J].
        self.show_info_overlay = False
        # How many lines the on-screen help menu's command list is scrolled down by.
        self.help_scroll_offset = 0
        # On-screen bounds of the help guide's scrollbar; None if closed or not needed.
        self.help_scrollbar_rect = None
        # How many command-list lines are visible at once, kept in sync by render_scene().
        self.help_scroll_visible_lines = 1
        # How far help_scroll_offset can go, kept in sync by render_scene().
        self.help_scroll_max_offset = 0
        # True while the operator is actively dragging the help guide's scrollbar thumb.
        self.help_scrollbar_dragging = False
        # Which direction to move the queue once the current image's loop
        # exits without saving (Next/Previous Image); None advances forward.
        self.navigate_direction = None
        # True once the current image's loop exits because the operator
        # asked to switch Mode 1 <-> 2 (Switch Mode button) -- run()'s outer
        # loop rebuilds image_queue for the other mode instead of just
        # advancing to the next position when this is set.
        self.switch_mode_requested = False

    def _init_shapes_and_history(self):
        """The saved cracks/detachments themselves, plus per-session
        bookkeeping (building group, undo/redo, in-progress traces)."""
        self.saved_cracks = []
        self.saved_detachments = []
        # Keeps in memory the active building group index for the current session
        self.CURRENT_BUILDING_INDEX = None
        # Stores the mode chosen by the user at startup ("1" or "2")
        self.SESSION_MODE = "1"
        self.show_help_menu = False

        # Chronological queues isolating current drawing context configurations
        self.action_history = []
        self.redo_history = []

        self.temp_start = None
        self.temp_path = []
        self.temp_nodes = []
        self.save_timestamp = 0.0
        self.historical_processed_queue = []  # Keeps track of the actual historical order of images
        # Hover-state initialization for snapping onto existing cracks
        self.hover_extension_index = None
        self.hover_extension_mode = 'start'

    def _init_hud_and_edit_mode(self):
        # HUD text zoom state (False = Normal size, True = Enlarged)
        self.hud_large_size = False
        # edit mode
        self.edit_mode = False
        self.selected_edit_poly = None
        self.selected_edit_idx = None

    def _init_translate_state(self):
        """TRANSLATE MODE (KEY T): re-traces one imported crack via A* against the real edge (two clicks). See [V] for the bulk, no-click version, with its own plausibility guard against a bad homography estimate.
        """
        self.translate_state = {
            "active": False,
            "idx": None,
            "start": None,
        }
        # Tracks only the exact points the user clicked while drawing
        self.user_clicked_nodes = []

    def _init_timer_state(self):
        """--- STOPWATCH TIMER STATES ---"""
        self.timer_is_paused = False           # Tracks if the stopwatch is currently frozen
        self.total_elapsed_paused_time = 0.0   # Accumulates completed session seconds
        self.image_load_time = 0.0             # Timestamp marker when image opens
        self.pathfinding_timeout_triggered = False  # Activates the red HUD warning overlay
        self.warp_jitter_triggered = False     # Triggers the HUD warning for Warp jitter

    def _init_import_warning_state(self):
        """On-screen warning shown when W/L crack-projection fails to import (no previous session, no building code assigned, or a poor perspective match)."""
        self.import_warning_triggered = False
        self.import_warning_message = ""
        self.import_warning_timestamp = 0.0

    def _init_runtime_options(self):
        """User runtime options configuration flags."""
        self.SAVE_SEG_IMAGE = True
        self.CONTRAST_ENHANCEMENT = True
        self.MIN_DETACHMENT_AREA_CM2 = 2.0

    def _init_homography_thresholds(self):
        """Quality thresholds for the W/L SIFT+homography crack projection -- below these the import still happens, with an on-screen low-confidence warning."""
        self.HOMOGRAPHY_MIN_INLIERS = 15
        self.HOMOGRAPHY_MIN_INLIER_RATIO = 0.35

    def _init_building_group_thresholds(self):
        """Thresholds for automatic building-group assignment by SIFT+homography photo similarity -- stricter than HOMOGRAPHY_MIN_INLIERS, since a false positive merges two unrelated buildings.
        """
        self.BUILDING_GROUP_MATCH_MIN_GOOD_MATCHES = 12
        self.BUILDING_GROUP_MATCH_MIN_INLIERS = 20
        self.BUILDING_GROUP_MATCH_MIN_INLIER_RATIO = 0.40

        # Downscale bound (px) for SIFT during the automatic group-membership decision only; W/L's real crack projection stays full resolution.
        self.AUTO_GROUP_SIFT_MAX_DIM = 1200

    def _init_crack_compat_thresholds(self):
        """Thresholds for [G]: a crack is "incompatible" and discarded from the whole group when its sinuosity disagrees by more than this fraction across photos.
        """
        self.CRACK_COMPAT_SINUOSITY_THRESHOLD = 0.30
        self.CRACK_COMPAT_LENGTH_THRESHOLD = None

    def _init_crack_mask_settings(self):
        """Pixels to dilate the exported crack mask by, on each side of the traced centerline."""
        self.CRACK_MASK_DILATION_PX = 2

        # CRACK_AUTO_EDGE_MAX_OFFSET_PX caps automatic edge detection's per-side reach (px); CRACK_MASK_AUTO_EDGE_ENABLED disables it in favor of flat dilation.
        self.CRACK_AUTO_EDGE_MAX_OFFSET_PX = 2
        self.CRACK_MASK_AUTO_EDGE_ENABLED = True

    def _init_gsd_calibration(self):
        """Millimeters of real wall per pixel of the current photo batch; 0.0 means uncalibrated, leaving CRACK_AUTO_EDGE_MAX_OFFSET_PX/CRACK_MASK_DILATION_PX at their fallback values.
        """
        self.IMAGE_GSD_MM_PER_PX = 0.0

        # Target physical crack width (mm) the exported band should represent once GSD is known.
        self.CRACK_TARGET_PHYSICAL_WIDTH_MM = 3.0

        # Hard cap (px) on the per-side offset resolve_crack_edge_width_from_gsd() can compute.
        self.CRACK_AUTO_EDGE_GSD_SAFETY_CEILING_PX = 20

    def _init_width_edit_state(self):
        """[A] key state machine: active/pending_crack_idx/pending_i0 track the two-click tract selection; focused_crack_idx/focused_seg_idx track which tract [+]/[-] adjusts.
        """
        self.width_edit_state = {
            "active": False,
            "pending_crack_idx": None,
            "pending_i0": None,
            "focused_crack_idx": None,
            "focused_seg_idx": None,
        }

        # Safety cap (px per side) for the [+]/[-]/[ /{ /] /} width-edit keys.
        self.WIDTH_EDIT_MAX_PX = 100

    def _init_manual_group_entry_state(self):
        """[B] key state: active is whether on-screen digit entry is open; buffer holds the typed digits as a string."""
        self.manual_group_entry_state = {
            "active": False,
            "buffer": "",
        }

        # How far (px, perpendicular to the crack) _autofill_tract_mask() scans outward.
        self.WIDTH_EDIT_AUTOFILL_SEARCH_MARGIN_PX = 120

        # Kernel width (px) for the MORPH_CLOSE pass bridging small gaps between rays.
        self.WIDTH_EDIT_FILL_CLOSE_PX = 5

    def _init_calibration_and_misc(self):
        """Calibration variables and a few standalone session counters."""
        self.PIXEL_TO_CM_SCALE = 0.05
        self.calib_start = None
        self.calib_current_mouse = None
        self.calibration_mode = False
        self.report_data_summary = []
        self.total_cracks_length_cm = 0.0
        self.modalita_scelta = "1"

    def _init_visual_masks(self):
        """Initializes overlay mask placeholders, via a lazy numpy import so Config can be instantiated without numpy for e.g. argument-parsing purposes."""
        import numpy as np
        self.blue_visual_mask = np.zeros((100, 100), dtype=np.uint8)
        self.green_visual_mask = np.zeros((100, 100), dtype=np.uint8)

    def _init_queue_and_cache(self):
        """image_queue is the ordered list of photos to process; _SIFT_FEATURE_CACHE memoizes (keypoints, descriptors) per (path, mtime, size, max_dim) for building-group matching.
        """
        self.image_queue = []
        self._SIFT_FEATURE_CACHE = {}

    def __init__(self):
        self._init_paths()
        self._init_session_pointers()
        self._init_display_state()
        self._init_tool_and_overlay_state()
        self._init_shapes_and_history()
        self._init_hud_and_edit_mode()
        self._init_translate_state()
        self._init_timer_state()
        self._init_import_warning_state()
        self._init_runtime_options()
        self._init_homography_thresholds()
        self._init_building_group_thresholds()
        self._init_crack_compat_thresholds()
        self._init_crack_mask_settings()
        self._init_gsd_calibration()
        self._init_width_edit_state()
        self._init_manual_group_entry_state()
        self._init_calibration_and_misc()
        self._init_visual_masks()
        self._init_queue_and_cache()
