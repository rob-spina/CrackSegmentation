"""config.py (object-oriented port): holds every configuration constant and runtime-state attribute the segmentation pipeline needs, as attributes of a single Config class passed around explicitly.
"""
import os
import sys
import tempfile
from pathlib import Path


def resolve_script_dir():
    """Returns the app's data folder (Images, Binary files, JSON files, CSV report): next to the script when run from source, or a per-user data folder when packaged/frozen -- normally ~/Documents/CrackSegmentation, falling back to %LOCALAPPDATA%/~/CrackSegmentation or a temp folder if that isn't writable (e.g. a broken OneDrive redirect of Documents).
    """
    if getattr(sys, 'frozen', False):
        candidates = [Path.home() / "Documents" / "CrackSegmentation"]
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidates.append(Path(local_appdata) / "CrackSegmentation")
        candidates.append(Path.home() / "CrackSegmentation")

        last_error = None
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError as exc:
                last_error = exc
                print(f"[WARN] Could not use {candidate} as the data folder ({exc}); trying the next option...")

        # Last resort: a temp folder, so the app can still start instead of
        # crashing outright. Data here may not survive a reboot, but it
        # beats a hard crash on every launch.
        fallback = Path(tempfile.gettempdir()) / "CrackSegmentation"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] Falling back to a temporary folder for data: {fallback} (previous attempt failed: {last_error})")
        return fallback
    return Path(__file__).resolve().parent


# Data folders (v1.0.5): original photos, exported PNGs (masks + overlay) and LabelMe JSONs.
IMAGES_DIR_NAME = "Images"
BINARY_DIR_NAME = "Binary files"
JSON_DIR_NAME = "JSON files"
# Pre-1.0.5 folders, migrated automatically by migrate_legacy_folders().
LEGACY_ARCHIVE_DIR_NAME = "already processed images"
LEGACY_OUTPUT_DIR_NAME = "segmentated images"
_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _legacy_file_destination(fname):
    """Target folder name for a file found in a pre-1.0.5 folder, or None to leave it where it is."""
    lower = fname.lower()
    if lower.endswith('.json') or lower.endswith('.json.bak'):
        return JSON_DIR_NAME
    base, ext = os.path.splitext(lower)
    if ext in _IMAGE_EXTENSIONS and (base.endswith('-seg') or base.endswith('_mask')):
        return BINARY_DIR_NAME
    if ext in _IMAGE_EXTENSIONS:
        return IMAGES_DIR_NAME
    return None


# Which old folder holds the most recent copy of each kind of file: Mode 2 edited the JSONs
# in the archive, while since v1.0.3 masks and overlays were always rewritten in the output folder.
_LEGACY_PRIORITY = {
    JSON_DIR_NAME: (LEGACY_ARCHIVE_DIR_NAME, LEGACY_OUTPUT_DIR_NAME),
    IMAGES_DIR_NAME: (LEGACY_ARCHIVE_DIR_NAME, LEGACY_OUTPUT_DIR_NAME),
    BINARY_DIR_NAME: (LEGACY_OUTPUT_DIR_NAME, LEGACY_ARCHIVE_DIR_NAME),
}


def _move_if_free(src, dest_dir):
    """Moves src into dest_dir unless a file with that name is already there. True if moved."""
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.exists(dest):
        return False
    try:
        os.makedirs(dest_dir, exist_ok=True)
        os.replace(src, dest)
        return True
    except OSError as exc:
        print(f"[MIGRATION WARNING] Could not move {src}: {exc}")
        return False


def _remove_if_only_junk(folder):
    """Removes an old folder once nothing but Finder metadata (.DS_Store) is left in it."""
    try:
        leftovers = os.listdir(folder)
        if all(f == '.DS_Store' for f in leftovers):
            for f in leftovers:
                os.remove(os.path.join(folder, f))
            os.rmdir(folder)
    except OSError:
        pass


def migrate_legacy_folders(base_dir):
    """Moves files from the pre-1.0.5 folders into Images / Binary files / JSON files.
    Never overwrites a file: a name already present at the destination stays in the old folder.
    For duplicates the most recent copy wins (see _LEGACY_PRIORITY). Returns the number of files moved."""
    base_dir = str(base_dir)
    moved = 0
    for dest_name, legacy_order in _LEGACY_PRIORITY.items():
        for legacy_name in legacy_order:
            legacy_dir = os.path.join(base_dir, legacy_name)
            if not os.path.isdir(legacy_dir):
                continue
            for fname in sorted(os.listdir(legacy_dir)):
                src = os.path.join(legacy_dir, fname)
                if _legacy_file_destination(fname) == dest_name and os.path.isfile(src):
                    moved += _move_if_free(src, os.path.join(base_dir, dest_name))
    for legacy_name in (LEGACY_ARCHIVE_DIR_NAME, LEGACY_OUTPUT_DIR_NAME):
        _remove_if_only_junk(os.path.join(base_dir, legacy_name))
    if moved:
        print(f"[MIGRATION] {moved} file(s) moved from the old folders into "
              f"'{IMAGES_DIR_NAME}', '{BINARY_DIR_NAME}' and '{JSON_DIR_NAME}'.")
    return moved


class Config:
    """Holds every configuration constant and mutable runtime-state
    attribute the segmentation pipeline needs."""

    def _init_paths(self):
        """Resolves operating directory paths based on the script location (or a per-user data folder when packaged -- see resolve_script_dir())."""
        self.SCRIPT_DIR = resolve_script_dir()
        self.IMAGE_FOLDER = os.path.join(str(self.SCRIPT_DIR), IMAGES_DIR_NAME)
        # Masks and overlay PNGs go to 'Binary files'; JSONs (OUTPUT_FOLDER) to 'JSON files'.
        self.folder_seg_img = BINARY_DIR_NAME
        self.OUTPUT_FOLDER = os.path.join(str(self.SCRIPT_DIR), JSON_DIR_NAME)

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
        # True once a Mode 1 save removed the current image from image_queue;
        # the next image has already shifted into queue_pos, so don't advance.
        self.queue_item_consumed = False
        # Images saved (and removed from image_queue) since the queue was last
        # built -- keeps the HUD's "FILE: n/N" counting up in Mode 1.
        self.queue_items_done = 0

    def _init_shapes_and_history(self):
        """The saved cracks/detachments themselves, plus per-session
        bookkeeping (building group, undo/redo, in-progress traces)."""
        self.saved_cracks = []
        self.saved_detachments = []
        # [I] ignore regions: polygons over occluded areas (nets, scaffolding), excluded from training/evaluation.
        self.saved_ignores = []
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
        # CUT [1] / JOIN [2] crack tools: mode is None, 'cut' or 'join'; first is the
        # (crack_idx, point_idx) picked by the first of the two clicks.
        self.crack_cut_join_state = {"mode": None, "first": None}
        # Crack parts removed with CUT on the current photo: JOIN routes around them.
        self.cut_exclusions = []

    def _init_timer_state(self):
        """--- STOPWATCH TIMER STATES ---"""
        self.timer_is_paused = False           # Tracks if the stopwatch is currently frozen
        self.total_elapsed_paused_time = 0.0   # Accumulates completed session seconds
        self.image_load_time = 0.0             # Timestamp marker when image opens
        self.pathfinding_timeout_triggered = False  # Activates the red HUD warning overlay
        self.warp_jitter_triggered = False     # Triggers the HUD warning for Warp jitter
        self.warp_banner_message = "WARP SYNC: aligned"
        # Persistent banners (TIMEOUT, WARP SYNC) hide themselves after this many seconds; [.] hides them at once.
        self.BANNER_AUTOHIDE_SECONDS = 15.0
        self.banner_first_seen = {}            # banner flag name -> time it was first drawn

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
        """Quality thresholds for the W/L SIFT+homography crack projection -- below these the alignment is rejected and nothing is imported."""
        self.HOMOGRAPHY_MIN_INLIERS = 15
        self.HOMOGRAPHY_MIN_INLIER_RATIO = 0.50
        # Plausible range for the scale change |det| of the homography's linear part between two shots.
        self.HOMOGRAPHY_MIN_AREA_SCALE = 0.2
        self.HOMOGRAPHY_MAX_AREA_SCALE = 5.0
        # v1.0.7 perspective pre-warp ([W]/[L]): the target photo is re-seen in memory from the source viewpoint
        # through the first alignment and matched again, which corrects it (see perspective_match.py).
        self.WARP_PREWARP_REFINE = True
        # Area scales the pre-warp accepts (a close-up vs a wide view of the same facade), and the largest
        # correction it may apply (share of the photo diagonal); a bigger one means the first alignment was wrong.
        self.WARP_PREWARP_MIN_AREA_SCALE = 0.02
        self.WARP_PREWARP_MAX_AREA_SCALE = 50.0
        self.WARP_PREWARP_MAX_CORRECTION = 0.05
        # A first alignment agreeing on at least this share of the matches is already precise: no pre-warp (saves time).
        self.WARP_PREWARP_SKIP_ABOVE_RATIO = 0.8
        # A projected shape with fewer of its points inside the target photo is left out (not squashed onto its border).
        self.IMPORT_MIN_INSIDE_PHOTO = 0.5
        # SIFT features are taken only around the source cracks (the facade plane), grown by this fraction of the image size.
        self.WARP_SOURCE_MASK_MARGIN = 0.02

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

    def _init_crack_quality_settings(self):
        """Per-crack quality scores saved in the LabelMe JSON, and the rule that marks a crack as uncertain.

        reliability_pct (0-100): share of the traced line backed by dark texture in the adaptive-threshold mask (same score as the [J] panel).
        confidence (0-1): mean percentile rank of a dark-ridge filter (Sato) response along the line, against the surrounding wall.
        0.5 = no more line-like than the local background, 1.0 = clearly the most line-like structure around. A heuristic, not a calibrated probability.
        """
        # A crack is saved as uncertain when reliability_pct is below this (None disables the reliability rule).
        self.CRACK_UNCERTAIN_RELIABILITY_PCT = 50.0
        # ...or when confidence is below this (None disables the confidence rule; set it after checking your own data).
        self.CRACK_UNCERTAIN_CONFIDENCE = None
        # Label of uncertain cracks in the JSON (the photo name is appended, like "crack_<photo>"). Shape flags also get {"uncertain": true}.
        self.CRACK_UNCERTAIN_LABEL = "crack_incerta"
        # Also write <photo>-crack_uncertain_mask.png with only the uncertain cracks (usable as an ignore region in training).
        # The regular crack mask is unchanged and still contains every crack.
        self.CRACK_UNCERTAIN_EXPORT_MASK = True
        # Sato filter scales (px) for the confidence score; roughly half the expected crack widths.
        self.CRACK_CONFIDENCE_SIGMAS = (1.0, 2.0, 3.0)
        # Background window around each stretch of crack used for the confidence ranking (px).
        self.CRACK_CONFIDENCE_WINDOW_MARGIN_PX = 40
        # Crack stretch length (px) processed per window, to bound memory on long cracks.
        self.CRACK_CONFIDENCE_CHUNK_PX = 256

        # Multi-view check at save: each crack is looked for in the other photos of its building group (__BLDG tag).
        self.CRACK_MULTIVIEW_ENABLED = True
        # At most this many other photos of the group are checked (alphabetical order).
        self.CRACK_MULTIVIEW_MAX_VIEWS = 5
        # A photo confirms a crack when the crack's confidence there reaches this value (same 0-1 scale as confidence).
        self.CRACK_MULTIVIEW_CONFIRM_CONFIDENCE = 0.8
        # A crack is saved as uncertain when confirmed/checked photos is below this ratio (None disables the rule).
        self.CRACK_UNCERTAIN_MULTIVIEW_RATIO = None
        # Longer side (px) of the downscaled copies used to align the photos; larger = more precise but slower.
        self.CRACK_MULTIVIEW_SIFT_MAX_DIM = 2000
        # Largest residual shift (px) the local texture alignment may apply after the homography.
        self.CRACK_MULTIVIEW_MAX_SHIFT_PX = 12
        # Tolerance (px) around the projected line when reading the ridge response in the other photo.
        self.CRACK_MULTIVIEW_SEARCH_RADIUS_PX = 3

    def _init_ignore_region_settings(self):
        """[I] ignore regions: label in the JSON (photo name appended) and the <photo>-ignore_mask.png export."""
        self.IGNORE_LABEL = "ignore"
        # Write <photo>-ignore_mask.png (filled polygons, 255 = exclude from loss and metrics).
        self.IGNORE_EXPORT_MASK = True

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

    def _init_inspection_tools_state(self):
        """v1.0.6 inspection aids: HWAV (viewed-window tracking), the Zoom window tool and the Crack report.
        All three are button-only actions (sidebar), see process_keypress() codes 14-16."""
        # HWAV: low-resolution map of the photo areas already shown in a zoomed view (None until first used).
        self.viewed_mask = None
        self.viewed_mask_scale = 1.0
        # Longer side (px) of the viewed-area map; it only needs window-level precision.
        self.VIEWED_MASK_MAX_DIM = 1024
        self.show_viewed_windows = False
        # Zoom window: drag a rectangle to zoom straight onto it; drag_start/drag_end are window coords.
        self.zoom_window_state = {"active": False, "drag_start": None, "drag_end": None, "press_window_xy": None}
        # Smallest rectangle side (window px) accepted as a drag, and the deepest zoom it may reach.
        self.ZOOM_WINDOW_MIN_DRAG_PX = 8
        self.ZOOM_WINDOW_MAX_FACTOR = 40.0
        # Crack report: active = waiting for / showing a report; idx = saved_cracks index of the reported crack.
        self.crack_report_state = {"active": False, "idx": None}

    def _init_alternative_route_state(self):
        """v1.0.7 [Alternative route]: other edge routes between the ends of the crack traced last.
        Button-only action (sidebar, process_keypress() code 18); [$] makes the route on screen the crack's path."""
        # crack = the crack dict being re-routed; routes[0] = its current path, routes[1:] = the alternatives found.
        # index = route on screen; direction = +1 next (arrow right) / -1 back (arrow left); exhausted = no more routes.
        self.alt_route_state = {"active": False, "crack": None, "routes": [], "index": 0, "direction": 1, "exhausted": False}
        # The crack traced last in this photo (the one the button re-routes); None until a crack is traced.
        self.last_traced_crack = None
        # A route longer than this x the current path (+10 px) is not offered.
        self.ALT_ROUTE_MAX_DETOUR = 2.5
        # Shares of a route's pixels that must lie on the detected edges, and away from every route already shown.
        self.ALT_ROUTE_MIN_ON_EDGE = 0.85
        self.ALT_ROUTE_MIN_NEW = 0.25
        # Most alternative routes offered for one crack.
        self.ALT_ROUTE_MAX_ROUTES = 8

    def _init_crack_filter_state(self):
        """v1.0.7 [Show cracks by number]: shows only the cracks whose numbers were typed (display only).
        Button-only action (sidebar, process_keypress() code 19)."""
        # None = every crack shown; else {"numbers": [...], "hidden_keys": path keys of the hidden cracks}.
        self.crack_filter = None
        # Display-only blue mask of the cracks shown, rebuilt with the other masks (None with no filter).
        self.filtered_blue_mask = None

    def _init_building_portion_state(self):
        """v1.0.7 [Building portion]: a window over the part of the building shared with the other photos of
        the group; while set, [W]/[L] align on it and import only inside it (sidebar, process_keypress() code 20)."""
        # rect = (x0, y0, x1, y1) image px or None; drag = None / "new" / "move" while the mouse is pressed.
        # compatible = orange polygons of the area shared with the rest of the group (None = not searched yet).
        self.portion_state = {"active": False, "rect": None, "drag": None, "anchor": None,
                              "rect_at_press": None, "press_window_xy": None, "compatible": None}
        # Longer side (px) the photos are reduced to when looking for the shared (orange) area.
        self.COMPATIBLE_AREA_MAX_DIM = 1600
        # Share of a projected shape's points that must fall inside the window for the shape to be imported.
        self.IMPORT_PORTION_MIN_INSIDE = 0.5

    def _init_trace_waypoint_state(self):
        """v1.0.7: Shift+click while tracing a crack adds an intermediate point the route must pass through."""
        # Intermediate points of the trace in progress, and the route traced so far through them.
        self.trace_waypoints = []
        self.trace_partial_path = []
        # Most intermediate points per crack; None = no limit (Shift+click as many as needed).
        self.TRACE_MAX_WAYPOINTS = None
        # Guided route through those points (guided_trace.py): search padding and click snap radius (px).
        self.TRACE_GUIDE_PAD_PX = 60
        self.TRACE_GUIDE_SNAP_PX = 4
        # Crack widths (ridge filter sigmas, px) of the thin-dark-line detector, and how much cheaper such lines are.
        self.TRACE_GUIDE_RIDGE_SIGMAS = (1.0, 1.5, 2.0)
        self.TRACE_GUIDE_RIDGE_WEIGHT = 20.0
        # How far a leg may stray from the line joining its two points: max(MIN_PX, FRAC x leg length).
        self.TRACE_GUIDE_CORRIDOR_MIN_PX = 20
        self.TRACE_GUIDE_CORRIDOR_FRAC = 0.08
        # Extra cost factor for running back along the previous leg (what made the route turn back at a point).
        self.TRACE_GUIDE_BACKTRACK_FACTOR = 20.0

    def _init_validation_settings(self):
        """v1.0.6 [Validation]: which segmented photos are moved to 'Suitable for training'.
        A photo qualifies when it has at least one crack with quality scores and meets every enabled rule below
        (None disables a rule). Near-duplicate photos are then reduced to the best-scoring one."""
        self.VALIDATION_DIR_NAME = "Suitable for training"
        # Mean reliability_pct of the photo's cracks must reach this.
        self.VALIDATION_MIN_MEAN_RELIABILITY_PCT = 60.0
        # Mean confidence (0-1) of the photo's cracks must reach this.
        self.VALIDATION_MIN_MEAN_CONFIDENCE = 0.60
        # Share of uncertain cracks ([?]) in the photo must not exceed this.
        self.VALIDATION_MAX_UNCERTAIN_RATIO = 0.25
        # Two photos are "too similar" when their 256-bit difference hashes differ in at most this many bits.
        self.VALIDATION_DUPLICATE_MAX_HASH_DISTANCE = 20
        # Set once [Validation] moved photos out of 'Images': run()'s outer loop rebuilds the queue.
        self.reload_queue_requested = False
        # True while the window stays open with no photo left (queue finished): only Validation, Switch Mode and Quit act.
        self.queue_idle = False

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
        self._init_crack_quality_settings()
        self._init_ignore_region_settings()
        self._init_gsd_calibration()
        self._init_width_edit_state()
        self._init_manual_group_entry_state()
        self._init_calibration_and_misc()
        self._init_visual_masks()
        self._init_inspection_tools_state()
        self._init_alternative_route_state()
        self._init_crack_filter_state()
        self._init_building_portion_state()
        self._init_trace_waypoint_state()
        self._init_validation_settings()
        self._init_queue_and_cache()
