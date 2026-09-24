"""test_gui_shell_parity.py

Headless verification that the changes made to smart_segmentation.py to
support the optional graphical control panel (crack_segmentation_gui.py)
do not alter behavior when the class is used exactly as before (no GUI
panel attached, `python3 smart_segmentation.py`).

Covers:
  A. process_keypress() dispatch, sampled across every branch of the
     extracted block (manual group entry priority, ESC, Enter/Q, S, L, R,
     B, W, and the handle_keyboard() fallthrough).
  B. The three _prompt_*() hooks: default (terminal input()) behavior is
     byte-identical to the original bare input() calls, including the
     ValueError -> None path.
  C. _pending_keys / _pump_extra_events: stay inert (empty list, no-op)
     for plain CrackSegmentation -- proving headless behavior is provably
     unchanged, not just "probably fine".
  D. A synthetic-key-queue test showing a GUI button click (appended to
     _pending_keys) drives process_keypress() through the exact same path,
     and the exact same resulting state, as the matching real keystroke.
  E. Full run() end-to-end smoke test with only cv2's GUI/HighGUI calls
     mocked (namedWindow/setWindowProperty/setMouseCallback/imshow/
     waitKey/destroyAllWindows) -- everything else (image I/O, JSON
     export, masks, CSV report) runs for real against a tiny synthetic
     image.

Uses the standard-library unittest (no third-party test runner needed).
Run with:  python3 -m unittest test_gui_shell_parity -v
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import cv2

from config import Config
import smart_segmentation
from smart_segmentation import CrackSegmentation

# archive_current_session_to_processed() (and a few sibling lookup methods)
# hardcode their "already processed images" folder next to the .py file
# itself (os.path.dirname(os.path.abspath(__file__))) rather than going
# through self.cfg.SCRIPT_DIR like build_chronological_queue() does. In
# normal use this is invisible (cfg.SCRIPT_DIR defaults to that very same
# directory), but an isolated test that points cfg.SCRIPT_DIR at a tmpdir
# needs to know where the archived file will REALLY land.
_REAL_MODULE_DIR = os.path.dirname(os.path.abspath(smart_segmentation.__file__))
_REAL_ARCHIVE_DIR = os.path.join(_REAL_MODULE_DIR, "already processed images")
_REAL_CSV_PATH = os.path.join(_REAL_MODULE_DIR, "segmentation_summary_report.csv")


def _cleanup_real_module_dir_artifacts():
    """export_labelme_format()'s incremental CSV writer and
    archive_current_session_to_processed()'s move both use the same
    hardcoded real-module-directory path (see note above) instead of
    cfg.SCRIPT_DIR, so an isolated tmpdir-based test still leaves a couple
    of files next to the .py file. Not a side effect of anything this test
    suite changed -- just tidied up here so repeated test runs stay clean."""
    shutil.rmtree(_REAL_ARCHIVE_DIR, ignore_errors=True)
    try:
        os.remove(_REAL_CSV_PATH)
    except OSError:
        pass


def make_app():
    return CrackSegmentation(Config())


class _MockedHighGui:
    """Context manager patching every cv2 HighGUI (real-window-dependent)
    function used anywhere in smart_segmentation.py, so tests can run with
    no display attached at all. cv2.waitKey is delay-aware: only calls made
    with delay==30 (the real per-image interactive loop's own
    `cv2.waitKey(30)`) pop from `key_sequence`; every other call (the
    cosmetic `cv2.waitKey(1)` right after a new image loads, and any
    `cv2.waitKey(0)` inside a modal warning popup) always returns -1,
    exactly like "no key pressed" -- so a queued key can only ever be
    consumed by the one call site it's meant for.
    """

    def __init__(self, key_sequence=()):
        self._queue = list(key_sequence)
        self._patches = []

    def _fake_wait_key(self, delay=0):
        if delay == 30 and self._queue:
            return self._queue.pop(0)
        return -1

    def __enter__(self):
        targets = {
            "cv2.namedWindow": mock.DEFAULT,
            "cv2.setWindowProperty": mock.DEFAULT,
            "cv2.setMouseCallback": mock.DEFAULT,
            "cv2.imshow": mock.DEFAULT,
            "cv2.destroyAllWindows": mock.DEFAULT,
            "cv2.destroyWindow": mock.DEFAULT,
            "cv2.moveWindow": mock.DEFAULT,
            "cv2.resizeWindow": mock.DEFAULT,
        }
        for name in targets:
            p = mock.patch(name)
            p.start()
            self._patches.append(p)
        p_wait = mock.patch("cv2.waitKey", side_effect=self._fake_wait_key)
        p_wait.start()
        self._patches.append(p_wait)
        p_rect = mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1920, 1080))
        p_rect.start()
        self._patches.append(p_rect)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def make_synthetic_image_folder(tmpdir, n_images=1, size=(220, 300)):
    images_dir = os.path.join(tmpdir, "Images")
    os.makedirs(images_dir, exist_ok=True)
    paths = []
    for i in range(n_images):
        img = np.full((size[0], size[1], 3), 200, dtype=np.uint8)
        cv2.line(img, (20, 20), (size[1] - 20, size[0] - 20), (30, 30, 30), 2)
        path = os.path.join(images_dir, f"img_{i:02d}.png")
        cv2.imwrite(path, img)
        paths.append(path)
    return images_dir, paths


class TestSiftFeatureCache(unittest.TestCase):
    """_get_cached_sift_features() distinguishes an unreadable image
    (never cached, so a later retry can pick up a fixed/replaced file)
    from a readable image with zero detected features (cached as
    (kp, None), same as any other successful read) -- verified here
    since that distinction was easy to lose while splitting the
    function into smaller pieces."""

    def test_unreadable_image_is_not_cached(self):
        app = make_app()
        app.cfg.AUTO_GROUP_SIFT_MAX_DIM = 800
        missing_path = "/tmp/definitely_does_not_exist_12345.jpg"
        kp, des = app._get_cached_sift_features(missing_path)
        self.assertIsNone(kp)
        self.assertIsNone(des)
        cache_keys = [k for k in app.cfg._SIFT_FEATURE_CACHE if k[0].endswith("12345.jpg")]
        self.assertEqual(cache_keys, [], "an unreadable image must not be cached")

    def test_readable_blank_image_is_cached_even_with_no_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = make_app()
            app.cfg.AUTO_GROUP_SIFT_MAX_DIM = 800
            img_path = os.path.join(tmpdir, "blank.png")
            blank = np.full((50, 50, 3), 128, dtype=np.uint8)  # flat color: no SIFT keypoints
            cv2.imwrite(img_path, blank)
            app._get_cached_sift_features(img_path)
            cache_keys = [k for k in app.cfg._SIFT_FEATURE_CACHE if k[0] == os.path.abspath(img_path)]
            self.assertEqual(len(cache_keys), 1, "a readable image must be cached even with zero features")


class TestSwitchModeAtRuntime(unittest.TestCase):
    """USER REQUEST: switch between Mode 1 (new images) and Mode 2 (reload
    already segmented) without restarting the app, via a new Switch Mode
    button/menu item (code 6, see process_keypress())."""

    def _make_mode2_source(self, tmpdir, base_name="already_seg"):
        """A minimal already-segmented image + matching JSON pair, the
        shape _load_mode2_queue() looks for."""
        archive_dir = os.path.join(tmpdir, "already processed images")
        os.makedirs(archive_dir, exist_ok=True)
        img = np.full((220, 300, 3), 180, dtype=np.uint8)
        img_path = os.path.join(archive_dir, f"{base_name}.png")
        cv2.imwrite(img_path, img)
        with open(os.path.join(archive_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump({"shapes": [], "imagePath": f"{base_name}.png"}, f)
        return archive_dir, img_path

    def test_load_mode2_queue_finds_images_with_an_uppercase_extension(self):
        # USER REPORT: on Linux (a case-sensitive filesystem), a JSON's
        # paired image with an uppercase extension (.PNG, .JPG -- common
        # straight off a phone/camera) was never found, even though it
        # plainly existed on disk -- the old code checked
        # os.path.exists() against a lowercase-only candidate path,
        # which only ever worked "by accident" on macOS/Windows (both
        # case-insensitive filesystems), never on Linux.
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_mode2_case_")
        try:
            archive_dir = os.path.join(tmpdir, "already processed images")
            os.makedirs(archive_dir, exist_ok=True)
            img = np.full((220, 300, 3), 180, dtype=np.uint8)
            img_path = os.path.join(archive_dir, "photo1.PNG")  # deliberately uppercase
            ok, buf = cv2.imencode(".png", img)
            with open(img_path, "wb") as f:
                f.write(buf.tobytes())
            with open(os.path.join(archive_dir, "photo1.json"), "w", encoding="utf-8") as f:
                json.dump({"shapes": [], "imagePath": "photo1.PNG"}, f)

            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            app._load_mode2_queue(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))

            self.assertEqual(app.cfg.image_queue, [img_path])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rebuild_queue_for_mode_switches_from_1_to_2(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_switchmode_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=1)
            _, mode2_img_path = self._make_mode2_source(tmpdir)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)
            app.cfg.modalita_scelta = "1"
            app.cfg.image_queue = [os.path.join(tmpdir, "Images", "img_00.png")]

            ok = app._rebuild_queue_for_mode("2")

            self.assertTrue(ok)
            self.assertEqual(app.cfg.modalita_scelta, "2")
            self.assertEqual(app.cfg.image_queue, [mode2_img_path])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rebuild_queue_for_mode_reverts_when_both_the_target_and_fallback_are_empty(self):
        # _load_mode2_queue() already falls back to scanning mode 1's folder
        # on its own when mode 2 is empty (matching the original startup
        # behavior) -- so to see a genuine revert-to-previous-queue here,
        # BOTH folders must be empty, not just the target mode's.
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_switchmode_empty_")
        try:
            os.makedirs(os.path.join(tmpdir, "Images"), exist_ok=True)  # empty on purpose
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)
            app.cfg.modalita_scelta = "1"
            original_queue = ["/some/previous/image.png"]
            app.cfg.image_queue = list(original_queue)

            ok = app._rebuild_queue_for_mode("2")

            self.assertFalse(ok, "switching when both folders are empty must fail, not silently succeed")
            self.assertEqual(app.cfg.modalita_scelta, "1", "must revert to the previous mode")
            self.assertEqual(app.cfg.image_queue, original_queue, "must revert to the previous queue")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


    def test_switch_mode_key_end_to_end_via_run(self):
        """A real run(): starts in mode 1 on img_00, code 6 switches to
        mode 2 mid-session, landing on the already-segmented file -- all
        without app.run() ever returning or restarting."""
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_switchmode_e2e_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=1)
            self._make_mode2_source(tmpdir, base_name="already_seg")
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            seen_modes = []
            original_start = app._start_image_session

            def _record_and_start(queue_pos, index, total_files):
                seen_modes.append((app.cfg.modalita_scelta, os.path.basename(app.cfg.image_queue[queue_pos])))
                return original_start(queue_pos, index, total_files)

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 mock.patch.object(app, "_start_image_session", side_effect=_record_and_start), \
                 _MockedHighGui(key_sequence=[6, ord('s')]):
                app.run()


            self.assertEqual(len(seen_modes), 2, "expected exactly two image sessions: mode 1, then mode 2")
            self.assertEqual(seen_modes[0], ("1", "img_00.png"))
            self.assertEqual(seen_modes[1][0], "2")
            self.assertIn("already_seg", seen_modes[1][1])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCrackReliabilityMetric(unittest.TestCase):
    """USER REQUEST: per-crack reliability metric, computed from how much
    of the traced path is corroborated by real crack-like texture in
    binary_mask (see _auto_detect_crack_edge_offsets), shown in the [J]
    info panel."""

    def test_returns_none_without_a_binary_mask(self):
        app = make_app()
        app.cfg.binary_mask = None
        self.assertIsNone(app._compute_crack_reliability([(5, 5), (10, 5)]))

    def test_returns_none_for_an_empty_path(self):
        app = make_app()
        app.cfg.binary_mask = np.zeros((50, 50), dtype=np.uint8)
        self.assertIsNone(app._compute_crack_reliability([]))
        self.assertIsNone(app._compute_crack_reliability(None))

    def test_fully_corroborated_path_scores_near_100_percent(self):
        app = make_app()
        app.cfg.binary_mask = np.full((50, 50), 255, dtype=np.uint8)
        app.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX = 3
        path = [(10, 25), (15, 25), (20, 25), (25, 25), (30, 25)]
        pct = app._compute_crack_reliability(path)
        self.assertGreater(pct, 90)

    def test_uncorroborated_path_scores_0_percent(self):
        app = make_app()
        app.cfg.binary_mask = np.zeros((50, 50), dtype=np.uint8)
        app.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX = 3
        path = [(10, 25), (15, 25), (20, 25), (25, 25), (30, 25)]
        self.assertEqual(app._compute_crack_reliability(path), 0.0)

    def test_reliability_lines_skip_inactive_cracks_and_group_by_five(self):
        app = make_app()
        app.cfg.binary_mask = np.full((50, 50), 255, dtype=np.uint8)
        app.cfg.CRACK_AUTO_EDGE_MAX_OFFSET_PX = 3
        straight_path = [(10, 25), (15, 25), (20, 25), (25, 25)]
        app.cfg.saved_cracks = [
            {'path': straight_path, 'active': True},
            {'path': straight_path, 'active': False},  # skipped
            {'path': straight_path, 'active': True},
            {'path': straight_path, 'active': True},
            {'path': straight_path, 'active': True},
            {'path': straight_path, 'active': True},
            {'path': straight_path, 'active': True},
        ]
        lines = app._build_crack_reliability_lines()
        self.assertEqual(len(lines), 2, "6 active cracks at 5 per line must produce 2 lines")
        self.assertEqual(lines[0].count("Crack"), 5)
        self.assertEqual(lines[1].count("Crack"), 1)
        # Inactive crack (index 2 in saved_cracks) must not appear at all.
        self.assertNotIn("Crack 2:", lines[0] + lines[1])

    def test_reliability_lines_empty_when_there_are_no_cracks(self):
        app = make_app()
        app.cfg.binary_mask = np.full((50, 50), 255, dtype=np.uint8)
        app.cfg.saved_cracks = []
        self.assertEqual(app._build_crack_reliability_lines(), [])

    def test_building_status_line_is_positioned_before_the_reliability_section(self):
        # USER REQUEST: "Building" must appear ABOVE "Crack Reliability" in
        # the [J] panel, not overlapping it -- both now derive their y from
        # the same hm dict, one line apart, so this ordering is guaranteed
        # whenever a building line is shown (CURRENT_IMAGE_PATH is set).
        app = make_app()
        app.cfg.CURRENT_IMAGE_PATH = "/tmp/photo__BLDG001.jpg"
        hm = app._compute_hud_font_metrics()
        building_y = app._resolve_building_hud_y(hm)
        reliability_label_y = hm['y_pos2'] + hm['line_h'] * 2  # one building line, then the label
        self.assertLess(building_y, reliability_label_y)
        self.assertEqual(building_y, hm['y_pos2'] + hm['line_h'])

    def test_panel_reserves_no_building_line_when_no_image_is_loaded(self):
        app = make_app()
        app.cfg.CURRENT_IMAGE_PATH = None
        app.cfg.show_info_overlay = True
        app.cfg.binary_mask = np.full((50, 50), 255, dtype=np.uint8)
        app.cfg.saved_cracks = [{'path': [(10, 25), (15, 25), (20, 25)], 'active': True}]
        win_out = np.zeros((300, 800, 3), dtype=np.uint8)
        hm = app._compute_hud_font_metrics()
        # Must not raise even though draw_building_hud_overlay() itself will
        # no-op (no CURRENT_IMAGE_PATH) right after this.
        app._draw_info_overlay(win_out, "status", "metrics", hm)


class TestClearAllIsUndoableAndRedoable(unittest.TestCase):
    """USER REPORT: "Clear X non funziona con undo e redo" -- [X] must
    itself be a proper undoable/redoable action: [U] right after
    restores everything it cleared, [R] re-applies the clear."""

    def _make_app_with_content(self):
        app = make_app()
        app.cfg.JSON_OUTPUT_PATH = "/tmp/nonexistent_test_clear_all.json"
        app.cfg.saved_cracks = [{'path': [(0, 0), (1, 1)], 'start': (0, 0), 'end': (1, 1), 'active': True, 'session_id': 'x'}]
        app.cfg.saved_detachments = [{'path': [(2, 2), (3, 3), (4, 4)], 'active': True, 'session_id': 'x'}]
        return app

    def test_clear_all_records_one_undoable_history_entry(self):
        app = self._make_app_with_content()
        app.cfg.redo_history = [('crack_new', {'path': [(9, 9)]})]

        app._clear_all()

        self.assertEqual(app.cfg.saved_cracks, [])
        self.assertEqual(app.cfg.saved_detachments, [])
        self.assertEqual(len(app.cfg.action_history), 1)
        self.assertEqual(app.cfg.action_history[0][0], 'clear_all')
        self.assertEqual(app.cfg.redo_history, [], "starting a new action must drop any pending redo")

    def test_undo_after_clear_all_restores_everything(self):
        app = self._make_app_with_content()
        original_cracks = list(app.cfg.saved_cracks)
        original_detachments = list(app.cfg.saved_detachments)

        app._clear_all()
        app._undo_last_action()

        self.assertEqual(app.cfg.saved_cracks, original_cracks)
        self.assertEqual(app.cfg.saved_detachments, original_detachments)

    def test_redo_after_undoing_a_clear_all_re_clears(self):
        app = self._make_app_with_content()

        app._clear_all()
        app._undo_last_action()
        app._handle_redo_key()

        self.assertEqual(app.cfg.saved_cracks, [])
        self.assertEqual(app.cfg.saved_detachments, [])

    def test_full_round_trip_stays_consistent(self):
        app = self._make_app_with_content()
        original_cracks = list(app.cfg.saved_cracks)

        app._clear_all()
        app._undo_last_action()
        app._handle_redo_key()
        app._undo_last_action()

        self.assertEqual(app.cfg.saved_cracks, original_cracks, "a second undo must restore again")

    def test_undo_after_clear_then_draw_undoes_the_new_drawing_not_the_clear(self):
        # With clear_all properly recorded in history, a Clear followed
        # by a new crack followed by a single Undo must undo the newest
        # action (the drawing), not silently consume the clear entry.
        app = self._make_app_with_content()
        app._clear_all()

        new_crack = {'path': [(5, 5), (6, 6)], 'start': (5, 5), 'end': (6, 6), 'active': True, 'session_id': 'y'}
        app.cfg.saved_cracks.append(new_crack)
        app.cfg.action_history.append('crack')

        app._undo_last_action()

        self.assertEqual(app.cfg.saved_cracks, [], "the single Undo press must remove the newly drawn crack")
        self.assertEqual(len(app.cfg.redo_history), 1)


class TestTotalCracksLengthIsNotACumulativeCounter(unittest.TestCase):
    """USER REPORT: "la lunghezza del crack ha un valore spropositato e
    ha un comportamento come un cronometro con numeri che aumentano
    velocemente" -- _update_total_cracks_length() used `+=` onto
    self.cfg.total_cracks_length_cm, which PERSISTS across frames; since
    render_scene() (and this helper with it) runs on every single frame,
    this made the displayed length grow without bound instead of showing
    the current, correct total."""

    def test_repeated_calls_do_not_keep_growing_the_total(self):
        app = make_app()
        app.cfg.saved_cracks = [{'path': [(0, 0)] * 10, 'active': True}]
        app.cfg.PIXEL_TO_CM_SCALE = 0.05

        app._update_total_cracks_length()
        first = app.cfg.total_cracks_length_cm
        app._update_total_cracks_length()
        second = app.cfg.total_cracks_length_cm
        app._update_total_cracks_length()
        third = app.cfg.total_cracks_length_cm

        self.assertEqual(first, 0.5)
        self.assertEqual(second, first, "the total must not grow across repeated calls")
        self.assertEqual(third, first, "the total must not grow across repeated calls")

    def test_reflects_the_current_saved_cracks_not_a_stale_value(self):
        app = make_app()
        app.cfg.PIXEL_TO_CM_SCALE = 0.05
        app.cfg.saved_cracks = [{'path': [(0, 0)] * 10, 'active': True}]
        app._update_total_cracks_length()
        self.assertEqual(app.cfg.total_cracks_length_cm, 0.5)

        # Shrinking the cracks must shrink the reported total too --
        # impossible with a `+=` accumulator, which can only ever grow.
        app.cfg.saved_cracks = [{'path': [(0, 0)] * 4, 'active': True}]
        app._update_total_cracks_length()
        self.assertEqual(app.cfg.total_cracks_length_cm, 0.2)


class TestApplyDirectionalMargin(unittest.TestCase):
    """No existing test covered _apply_directional_margin() directly,
    despite its own docstring documenting two real historical bugs --
    added while splitting it into smaller helpers, to close that gap."""

    def _make_straight_horizontal_setup(self):
        app = make_app()
        app.cfg.H_img, app.cfg.W_img = 100, 100
        path = [(10, 50), (11, 50), (12, 50), (13, 50), (14, 50)]
        # binary_mask: the crack's real extent in the photo, 5px above
        # and below the centerline (rows 45..55), everywhere along the path.
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[45:56, 8:17] = 255
        app.cfg.binary_mask = binary_mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        return app, mask, path

    def test_growing_one_side_leaves_the_other_side_untouched(self):
        app, mask, path = self._make_straight_horizontal_setup()
        app._apply_directional_margin(mask, path, 0, 4, side_sign=1, delta=2, height=100, width=100)
        # side_sign=+1 on a rightward-traveling horizontal path perpendicular-
        # extends downward (+y) -- some pixel below the centerline at x=12
        # (a middle point, away from boundary tangent effects) must now be filled.
        self.assertTrue(any(mask[51:53, 12]), "the grown side should have some fill")
        # The opposite side (above the centerline) must be completely untouched.
        self.assertFalse(any(mask[45:50, 12]), "the untouched side must stay empty")

    def test_growth_is_capped_at_the_real_crack_contour(self):
        app, mask, path = self._make_straight_horizontal_setup()
        # binary_mask only extends 5px on this side -- asking for far more
        # growth must still stop at that real extent, not fill indefinitely.
        app._apply_directional_margin(mask, path, 0, 4, side_sign=1, delta=50, height=100, width=100)
        self.assertTrue(mask[55, 12] or mask[54, 12], "should fill up to roughly the real contour")
        self.assertFalse(any(mask[57:65, 12]), "must not grow past the crack's real extent in binary_mask")

    def test_shrink_retracts_from_the_current_outer_edge(self):
        app, mask, path = self._make_straight_horizontal_setup()
        app._apply_directional_margin(mask, path, 0, 4, side_sign=1, delta=3, height=100, width=100)
        filled_before = int(np.count_nonzero(mask[:, 12]))
        app._apply_directional_margin(mask, path, 0, 4, side_sign=1, delta=-1, height=100, width=100)
        filled_after = int(np.count_nonzero(mask[:, 12]))
        self.assertLess(filled_after, filled_before, "shrinking should remove at least the outermost pixel")

    def test_zero_delta_or_empty_path_is_a_no_op(self):
        app, mask, path = self._make_straight_horizontal_setup()
        result = app._apply_directional_margin(mask, path, 0, 4, side_sign=1, delta=0, height=100, width=100)
        self.assertFalse(np.any(result))
        result2 = app._apply_directional_margin(mask, [], 0, -1, side_sign=1, delta=5, height=100, width=100)
        self.assertFalse(np.any(result2))


class TestProcessKeypressDispatch(unittest.TestCase):

    def test_manual_group_entry_absolute_priority_and_return_false(self):
        app = make_app()
        app.cfg.manual_group_entry_state["active"] = True
        app.cfg.manual_group_entry_state["buffer"] = ""
        with mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            result = app.process_keypress(ord('5'))
        self.assertFalse(result)
        self.assertEqual(app.cfg.manual_group_entry_state["buffer"], "5")
        mrz.assert_called_once()

    def test_manual_group_entry_intercepts_even_s(self):
        app = make_app()
        app.cfg.manual_group_entry_state["active"] = True
        with mock.patch.object(app, "export_labelme_format") as mexp, \
             mock.patch.object(app, "refresh_zoom_viewport"):
            result = app.process_keypress(ord('s'))
        self.assertFalse(result)
        mexp.assert_not_called()

    def test_esc_exits_cleanly(self):
        app = make_app()
        with self.assertRaises(SystemExit) as ctx:
            app.process_keypress(27)
        self.assertEqual(ctx.exception.code, 0)

    def test_enter_and_q_save_and_advance(self):
        for code in (13, 10, ord('q'), ord('Q')):
            app = make_app()
            with mock.patch.object(app, "export_labelme_format") as mexp, \
                 mock.patch.object(app, "archive_current_session_to_processed") as march, \
                 mock.patch.object(app, "run_post_save_group_crack_check") as mchk:
                result = app.process_keypress(code)
            self.assertTrue(result, f"key {code!r} should trigger next-file")
            mexp.assert_called_once()
            march.assert_called_once()
            mchk.assert_called_once()

    def test_s_saves_and_advances(self):
        app = make_app()
        with mock.patch.object(app, "export_labelme_format") as mexp, \
             mock.patch.object(app, "archive_current_session_to_processed") as march, \
             mock.patch.object(app, "run_post_save_group_crack_check") as mchk:
            result = app.process_keypress(ord('s'))
        self.assertTrue(result)
        mexp.assert_called_once()
        march.assert_called_once()
        mchk.assert_called_once()

    def test_l_triggers_manual_align_and_does_not_advance(self):
        app = make_app()
        with mock.patch.object(app, "auto_import_best_previous_session") as mimp:
            result = app.process_keypress(ord('l'))
        self.assertFalse(result)
        mimp.assert_called_once()

    def test_r_redo_empty_history_does_not_advance(self):
        app = make_app()
        app.cfg.redo_history = []
        self.assertFalse(app.process_keypress(ord('r')))

    def test_r_redo_restores_crack(self):
        app = make_app()
        app.cfg.redo_history = [('crack', {'path': [(0, 0), (1, 1)], 'session_id': 'x'})]
        with mock.patch.object(app, "recalculate_masks") as mrec, \
             mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            result = app.process_keypress(ord('R'))
        self.assertFalse(result)
        self.assertEqual(len(app.cfg.saved_cracks), 1)
        self.assertEqual(app.cfg.action_history, ['crack'])
        mrec.assert_called_once()
        mrz.assert_called_once()

    def test_r_redo_restores_crack_tagged_crack_new(self):
        # BUGFIX regression test -- 'crack_new' (not 'crack') is what the
        # [U]/Undo handler in handle_keyboard() actually pushes onto
        # redo_history when undoing a just-drawn crack (see
        # test_full_undo_then_redo_cycle_restores_a_deleted_crack below
        # for the real, end-to-end reproduction of the bug report). This
        # is the exact tag that was silently un-recognized before the fix.
        app = make_app()
        app.cfg.redo_history = [('crack_new', {'path': [(0, 0), (1, 1)], 'session_id': 'x'})]
        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('r'))
        self.assertEqual(len(app.cfg.saved_cracks), 1,
                          "a crack undone with [U] must come back with [R] -- this was the reported bug")

    def test_full_undo_then_redo_cycle_restores_a_deleted_crack(self):
        # USER REPORT, reproduced end-to-end through the REAL Undo (U,
        # handle_keyboard) and Redo (R, process_keypress) code paths --
        # not a hand-seeded redo_history, which is exactly what let this
        # bug slip past the earlier, narrower test above.
        app = make_app()
        # Mirrors exactly what completing a new crack does (see the
        # a_star_pathfinding call site in mouse_callback): appends to
        # saved_cracks and pushes the plain string 'crack' onto
        # action_history.
        app.cfg.saved_cracks = [{'path': [(0, 0), (1, 1)], 'session_id': str(time.time())}]
        app.cfg.action_history = ['crack']
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = None

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))  # Undo -- real handler, not a stand-in
        self.assertEqual(app.cfg.saved_cracks, [], "Undo should have removed the crack")
        self.assertEqual(len(app.cfg.redo_history), 1)

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('r'))  # Redo -- real handler
        self.assertEqual(len(app.cfg.saved_cracks), 1,
                          "Redo must restore the exact crack Undo just removed")

    def test_t_automatic_snap_completion_pushes_crack_extension_not_bare_crack(self):
        # BUGFIX found while investigating a "undo/redo on fractures
        # doesn't work" report: the [T] automatic-snap tool's second
        # click modifies an EXISTING crack's path/start/end IN PLACE, but
        # used to push the bare string 'crack' onto action_history --
        # which Undo's handler reads as "a brand new crack was appended",
        # so pressing Undo after a snap deleted the ENTIRE crack instead
        # of just reverting the snap. It must push a ('crack_extension',
        # backup) tuple instead, the same mechanism this file's OTHER
        # branch of the same T-tool already correctly uses.
        app = make_app()
        original_start, original_end, original_path = (0, 0), (10, 10), [(0, 0), (5, 5), (10, 10)]
        app.cfg.saved_cracks = [{'start': original_start, 'end': original_end, 'path': list(original_path),
                                  'session_id': str(time.time())}]
        app.cfg.translate_state["active"] = True
        app.cfg.translate_state["start"] = (50, 50)  # first click already done, on the imported crack
        app.cfg.translate_state["idx"] = 0
        app.cfg.action_history = []

        new_path = [(1, 1), (6, 6), (11, 11)]
        with mock.patch.object(app, "a_star_pathfinding", return_value=new_path), \
             mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"), \
             mock.patch.object(app, "transform_window_to_real_coords", return_value=(60, 60)):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 60, 60, 0, None)  # second click, on the real crack

        self.assertEqual(app.cfg.saved_cracks[0]['path'], new_path, "the snap itself should still have applied")
        self.assertEqual(len(app.cfg.action_history), 1)
        last = app.cfg.action_history[-1]
        self.assertIsInstance(last, tuple, "must be a ('crack_extension', backup) tuple, not the bare string 'crack'")
        self.assertEqual(last[0], 'crack_extension')
        self.assertEqual(last[1]['start'], original_start)
        self.assertEqual(last[1]['end'], original_end)
        self.assertEqual(last[1]['path'], original_path)

    def test_undo_after_automatic_snap_reverts_the_shape_not_the_whole_crack(self):
        # The other half of the same fix: Undo must restore the ORIGINAL
        # path/start/end on that specific crack, not delete it outright.
        app = make_app()
        original_start, original_end, original_path = (0, 0), (10, 10), [(0, 0), (5, 5), (10, 10)]
        app.cfg.saved_cracks = [{'start': [1, 1], 'end': [11, 11], 'path': [(1, 1), (6, 6), (11, 11)],
                                  'session_id': str(time.time())}]
        app.cfg.action_history = [('crack_extension', {
            'index': 0, 'start': original_start, 'end': original_end, 'path': original_path,
        })]
        app.cfg.current_tool = None
        app.cfg.temp_start = None
        app.cfg.temp_nodes = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))

        self.assertEqual(len(app.cfg.saved_cracks), 1, "the crack must still exist -- only its shape should revert")
        self.assertEqual(app.cfg.saved_cracks[0]['start'], original_start)
        self.assertEqual(app.cfg.saved_cracks[0]['end'], original_end)
        self.assertEqual(app.cfg.saved_cracks[0]['path'], original_path)

    def test_v_bulk_retrace_records_a_backup_for_undo(self):
        # BUGFIX (same investigation): [V] used to modify possibly many
        # cracks' shapes with NO backup recorded at all, so Undo right
        # after a bulk retrace couldn't revert it.
        app = make_app()
        orig_path_0 = [(0, 0), (5, 5), (10, 10)]
        orig_path_1 = [(20, 20), (25, 25), (30, 30)]
        app.cfg.saved_cracks = [
            {'start': [0, 0], 'end': [10, 10], 'path': list(orig_path_0), 'orig_path': orig_path_0, 'active': True},
            {'start': [20, 20], 'end': [30, 30], 'path': list(orig_path_1), 'orig_path': orig_path_1, 'active': True},
        ]
        app.cfg.action_history = []
        new_path_0 = [(1, 1), (6, 6), (11, 11)]
        new_path_1 = [(21, 21), (26, 26), (31, 31)]

        with mock.patch.object(app, "a_star_pathfinding", side_effect=[new_path_0, new_path_1]), \
             mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('v'))

        self.assertEqual(app.cfg.saved_cracks[0]['path'], new_path_0, "the retrace itself should still have applied")
        self.assertEqual(app.cfg.saved_cracks[1]['path'], new_path_1)
        self.assertEqual(len(app.cfg.action_history), 1)
        last = app.cfg.action_history[-1]
        self.assertIsInstance(last, tuple)
        self.assertEqual(last[0], 'crack_bulk_retrace')
        self.assertEqual(len(last[1]), 2, "one backup per crack the retrace actually modified")
        self.assertEqual(last[1][0]['path'], orig_path_0)
        self.assertEqual(last[1][1]['path'], orig_path_1)

    def test_undo_after_v_bulk_retrace_reverts_every_crack_it_touched(self):
        app = make_app()
        app.cfg.saved_cracks = [
            {'start': [1, 1], 'end': [11, 11], 'path': [(1, 1), (6, 6), (11, 11)]},
            {'start': [21, 21], 'end': [31, 31], 'path': [(21, 21), (26, 26), (31, 31)]},
        ]
        app.cfg.action_history = [('crack_bulk_retrace', [
            {'index': 0, 'start': (0, 0), 'end': (10, 10), 'path': [(0, 0), (5, 5), (10, 10)]},
            {'index': 1, 'start': (20, 20), 'end': (30, 30), 'path': [(20, 20), (25, 25), (30, 30)]},
        ])]
        app.cfg.current_tool = None
        app.cfg.temp_start = None
        app.cfg.temp_nodes = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))

        self.assertEqual(len(app.cfg.saved_cracks), 2, "both cracks must still exist -- only their shapes revert")
        self.assertEqual(app.cfg.saved_cracks[0]['path'], [(0, 0), (5, 5), (10, 10)])
        self.assertEqual(app.cfg.saved_cracks[1]['path'], [(20, 20), (25, 25), (30, 30)])

    def test_click_near_crack_marker_deletes_it_and_records_crack_delete(self):
        # Confirms the exact scenario the fix below targets: clicking near
        # a crack's start/end marker (current_tool == 'crack', no trace in
        # progress) deletes that crack immediately and records it as
        # ('crack_delete', removed_crack) in action_history.
        app = make_app()
        crack = {'start': (100, 100), 'end': (200, 200), 'path': [(100, 100), (150, 150), (200, 200)]}
        app.cfg.saved_cracks = [crack]
        app.cfg.action_history = []
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = None

        with mock.patch.object(app, "transform_window_to_real_coords", return_value=(101, 101)), \
             mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 101, 101, 0, None)  # near the 'start' marker

        self.assertEqual(app.cfg.saved_cracks, [], "the crack should be deleted by the click")
        self.assertEqual(len(app.cfg.action_history), 1)
        self.assertEqual(app.cfg.action_history[0], ('crack_delete', crack))

    def test_undo_after_click_to_delete_restores_the_crack(self):
        # BUGFIX (likely the most common way to hit the reported "undo on
        # fractures doesn't work"): clicking near a crack's marker deletes
        # it and records ('crack_delete', removed_crack) -- but Undo had
        # NO case at all for this tag, so the tuple was just discarded and
        # the crack could never come back with [U].
        app = make_app()
        deleted_crack = {'start': (100, 100), 'end': (200, 200), 'path': [(100, 100), (150, 150), (200, 200)]}
        app.cfg.saved_cracks = []
        app.cfg.action_history = [('crack_delete', deleted_crack)]
        app.cfg.current_tool = None
        app.cfg.temp_start = None
        app.cfg.temp_nodes = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))

        self.assertEqual(len(app.cfg.saved_cracks), 1, "the deleted crack must come back")
        self.assertEqual(app.cfg.saved_cracks[0], deleted_crack)
        self.assertEqual(app.cfg.redo_history, [('crack_redelete', deleted_crack)],
                          "must also queue up a re-delete for a following Redo")

    def test_redo_after_that_undo_deletes_the_crack_again(self):
        # USER REPORT (follow-up): restoring the crack on Undo was only
        # half the fix -- Redo right afterward did nothing at all, since
        # every existing Redo branch only ever means "add this back".
        # Full cycle: click-to-delete -> Undo (restore) -> Redo (must
        # delete it again, and leave the cycle repeatable).
        app = make_app()
        deleted_crack = {'start': (100, 100), 'end': (200, 200), 'path': [(100, 100), (150, 150), (200, 200)]}
        app.cfg.saved_cracks = [deleted_crack]
        app.cfg.redo_history = [('crack_redelete', deleted_crack)]
        app.cfg.action_history = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('r'))

        self.assertEqual(app.cfg.saved_cracks, [], "Redo must delete the crack again")
        self.assertEqual(app.cfg.action_history, [('crack_delete', deleted_crack)],
                          "must be undo-able again afterward, exactly like the original delete")

    def test_full_delete_undo_redo_cycle_via_real_mouse_and_keyboard(self):
        # The whole round trip, through the REAL code paths throughout:
        # click near a marker deletes it (mouse_callback) -> Undo restores
        # it (handle_keyboard) -> Redo deletes it again (process_keypress).
        app = make_app()
        crack = {'start': (100, 100), 'end': (200, 200), 'path': [(100, 100), (150, 150), (200, 200)]}
        app.cfg.saved_cracks = [crack]
        app.cfg.action_history = []
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = None

        with mock.patch.object(app, "transform_window_to_real_coords", return_value=(101, 101)), \
             mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 101, 101, 0, None)
        self.assertEqual(app.cfg.saved_cracks, [])

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))
        self.assertEqual(len(app.cfg.saved_cracks), 1, "Undo should bring it back")

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('r'))
        self.assertEqual(app.cfg.saved_cracks, [], "Redo should delete it again")

    def test_undo_after_click_to_delete_restores_the_detachment(self):
        # Mirrors the crack fix above for detachments -- clicking near a
        # detachment's first node deletes it the same way (see
        # mouse_callback's detachment tool), and Undo had no case for
        # 'detachment_delete' either.
        app = make_app()
        deleted_detachment = {'path': [(50, 50), (60, 60), (70, 50)]}
        app.cfg.saved_detachments = []
        app.cfg.action_history = [('detachment_delete', deleted_detachment)]
        app.cfg.current_tool = None
        app.cfg.temp_start = None
        app.cfg.temp_nodes = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.handle_keyboard(ord('u'))

        self.assertEqual(len(app.cfg.saved_detachments), 1, "the deleted detachment must come back")
        self.assertEqual(app.cfg.saved_detachments[0], deleted_detachment)
        self.assertEqual(app.cfg.redo_history, [('detachment_redelete', deleted_detachment)])

    def test_redo_after_detachment_undo_deletes_it_again(self):
        app = make_app()
        deleted_detachment = {'path': [(50, 50), (60, 60), (70, 50)]}
        app.cfg.saved_detachments = [deleted_detachment]
        app.cfg.redo_history = [('detachment_redelete', deleted_detachment)]
        app.cfg.action_history = []

        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('r'))

        self.assertEqual(app.cfg.saved_detachments, [])
        self.assertEqual(app.cfg.action_history, [('detachment_delete', deleted_detachment)])

    def test_b_opens_manual_group_entry(self):
        app = make_app()
        with mock.patch.object(app, "start_manual_building_group_entry") as mstart, \
             mock.patch.object(app, "refresh_zoom_viewport"):
            result = app.process_keypress(ord('b'))
        self.assertFalse(result)
        mstart.assert_called_once()

    def test_w_no_previous_session_signals_warning(self):
        app = make_app()
        with mock.patch.object(app, "find_last_processed_session_asset", return_value=(None, None)), \
             mock.patch.object(app, "_signal_import_warning") as mwarn:
            result = app.process_keypress(ord('w'))
        self.assertFalse(result)
        mwarn.assert_called_once()

    def test_w_uses_self_window_name_not_a_local(self):
        app = make_app()
        app._window_name = "Crack Detector Workspace"
        with mock.patch.object(app, "find_last_processed_session_asset",
                                return_value=("base.jpg", "orig.json")), \
             mock.patch.object(app, "warp_and_adapt_json_to_new_image", return_value=True), \
             mock.patch.object(app, "load_labelme_format"), \
             mock.patch("cv2.setWindowProperty") as mprop:
            app.process_keypress(ord('w'))
        mprop.assert_called_once_with("Crack Detector Workspace", cv2.WND_PROP_TOPMOST, 1)

    def test_next_image_without_saving_advances_but_does_not_export(self):
        app = make_app()
        with mock.patch.object(app, "export_labelme_format") as mexp, \
             mock.patch.object(app, "archive_current_session_to_processed") as march, \
             mock.patch.object(app, "run_post_save_group_crack_check") as mchk:
            result = app.process_keypress(4)
        self.assertTrue(result, "should still advance to the next image")
        self.assertEqual(app.cfg.navigate_direction, "next")
        mexp.assert_not_called()
        march.assert_not_called()
        mchk.assert_not_called()

    def test_previous_image_sets_navigate_direction_and_does_not_export(self):
        app = make_app()
        with mock.patch.object(app, "export_labelme_format") as mexp, \
             mock.patch.object(app, "archive_current_session_to_processed") as march:
            result = app.process_keypress(5)
        self.assertTrue(result)
        self.assertEqual(app.cfg.navigate_direction, "previous")
        mexp.assert_not_called()
        march.assert_not_called()

    def test_navigate_without_saving_resets_translate_state(self):
        app = make_app()
        app.cfg.translate_state["active"] = True
        app.cfg.translate_state["idx"] = 3
        app.cfg.translate_state["start"] = (1, 2)
        app.process_keypress(4)
        self.assertFalse(app.cfg.translate_state["active"])
        self.assertIsNone(app.cfg.translate_state["idx"])
        self.assertIsNone(app.cfg.translate_state["start"])

    def test_unhandled_key_falls_through_to_handle_keyboard(self):
        app = make_app()
        with mock.patch.object(app, "handle_keyboard") as mhk:
            result = app.process_keypress(ord('x'))
        self.assertFalse(result)
        mhk.assert_called_once_with(ord('x'))

    def test_x_clears_all_via_real_handle_keyboard(self):
        app = make_app()
        app.cfg.JSON_OUTPUT_PATH = ""
        app.cfg.saved_cracks = [{'path': [(0, 0)]}]
        app.cfg.saved_detachments = [{'path': [(0, 0)]}]
        with mock.patch.object(app, "recalculate_masks"), \
             mock.patch.object(app, "refresh_zoom_viewport"):
            result = app.process_keypress(ord('x'))
        self.assertFalse(result)
        self.assertEqual(app.cfg.saved_cracks, [])
        self.assertEqual(app.cfg.saved_detachments, [])

    def test_n_toggles_blue_crack_overlay(self):
        app = make_app()
        self.assertTrue(app.cfg.show_crack_overlay,
                         "should default to visible, unchanged from every previous delivery")
        with mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            result = app.process_keypress(ord('n'))
        self.assertFalse(result)
        self.assertFalse(app.cfg.show_crack_overlay)
        mrz.assert_called_once()

        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('N'))
        self.assertTrue(app.cfg.show_crack_overlay, "uppercase N should toggle it back on")

    def test_blue_overlay_paint_statements_are_structurally_gated_by_the_flag(self):
        # Mechanical/structural check (AST-based, like the process_keypress
        # extraction check) that _paint_blue_crack_overlay() bails out via
        # an early `if not self.cfg.show_crack_overlay: return` before doing
        # any of its painting -- render_scene() itself was split into
        # focused helpers; this logic now lives in _paint_blue_crack_overlay.
        import ast
        import smart_segmentation
        with open(smart_segmentation.__file__, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)

        target_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_paint_blue_crack_overlay"
        )

        # First real statement (after the docstring, if any) must be the
        # early-return guard on show_crack_overlay.
        body = [n for n in target_fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        guard = body[0]
        self.assertIsInstance(guard, ast.If, "expected an early-return guard as the first statement")
        self.assertIn("show_crack_overlay", ast.unparse(guard.test))
        self.assertTrue(any(isinstance(s, ast.Return) for s in guard.body), "guard must return early when the flag is off")

        rest_src = ast.unparse(ast.Module(body=body[1:], type_ignores=[]))
        self.assertIn("mb == 255", rest_src)
        self.assertIn("(255, 0, 0)", rest_src)
        # The width-segment fills (_paint_width_segment_fills) are called
        # after the guard -- confirms they're gated too.
        self.assertIn("_paint_width_segment_fills", rest_src)

        # The green (detachment) dilation lives in a sibling function,
        # never called from here.
        self.assertNotIn("green_visual_mask", rest_src)

    def test_j_toggles_info_overlay_default_off(self):
        # USER REPORT: the FILE/Cracks Length/BUILDING status text used to
        # be permanently burned onto every frame with no way to hide it.
        # Off by default now; [J] toggles it.
        app = make_app()
        self.assertFalse(app.cfg.show_info_overlay,
                          "must default to HIDDEN -- it used to be permanently shown, that's the bug being fixed")
        with mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            result = app.process_keypress(ord('j'))
        self.assertFalse(result)
        self.assertTrue(app.cfg.show_info_overlay)
        mrz.assert_called_once()

        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.process_keypress(ord('J'))
        self.assertFalse(app.cfg.show_info_overlay, "uppercase J should toggle it back off")

    def test_info_overlay_paint_statements_are_structurally_gated_in_render_scene(self):
        # This gating logic now lives in _draw_info_overlay (render_scene()
        # itself was split into focused helpers, see _draw_hud_stack) --
        # bails out via an early return before doing any painting.
        import ast
        import smart_segmentation
        with open(smart_segmentation.__file__, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)

        target_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_draw_info_overlay"
        )

        body = [n for n in target_fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        guard = body[0]
        self.assertIsInstance(guard, ast.If, "expected an early-return guard as the first statement")
        self.assertIn("show_info_overlay", ast.unparse(guard.test))
        self.assertTrue(any(isinstance(s, ast.Return) for s in guard.body), "guard must return early when the flag is off")

        rest_src = ast.unparse(ast.Module(body=body[1:], type_ignores=[]))
        self.assertIn("status", rest_src)
        self.assertIn("metrics_str", rest_src)
        self.assertIn("addWeighted", rest_src, "expected a translucent panel behind the text")

    def test_building_hud_overlay_has_an_early_return_gated_on_info_toggle(self):
        import ast
        import smart_segmentation
        with open(smart_segmentation.__file__, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)

        method = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "draw_building_hud_overlay"
        )
        found = any(
            isinstance(child, ast.If) and "show_info_overlay" in ast.unparse(child.test)
            for child in ast.walk(method)
        )
        self.assertTrue(found, "draw_building_hud_overlay() should early-return when show_info_overlay is False")

    def test_building_hud_still_draws_when_info_overlay_is_on(self):
        app = make_app()
        app.cfg.show_info_overlay = True
        app.cfg.CURRENT_IMAGE_PATH = "/tmp/img_00__BLDG001.png"
        app.cfg.saved_cracks = []
        app.cfg.saved_detachments = []
        canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
        app.draw_building_hud_overlay(canvas)
        self.assertTrue(canvas.any(), "expected some pixels to be drawn when the info overlay is enabled")

    def test_building_hud_draws_nothing_when_info_overlay_is_off(self):
        app = make_app()
        app.cfg.show_info_overlay = False
        app.cfg.CURRENT_IMAGE_PATH = "/tmp/img_00__BLDG001.png"
        canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
        app.draw_building_hud_overlay(canvas)
        self.assertFalse(canvas.any(), "nothing should be drawn while the info overlay is toggled off")


class TestMouseCallbackHelpGuideSuppression(unittest.TestCase):
    """USER REPORT (first round): clicking the help guide's drawn
    scrollbar was instead placing a crack marker on the photo
    underneath -- fixed by giving the open guide absolute priority.
    USER REPORT (second round): clicking it only ever jumped by a fixed
    page, and actually DRAGGING it (the natural way to use a scrollbar)
    did nothing. Track used throughout: (1120, 300, 1135, 830), so
    track_h=530 -- clean round numbers are used for max_offset (20) to
    keep the proportional-position math easy to verify by hand."""

    def test_click_while_help_open_does_not_reach_the_crack_tool(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = None  # nothing drawn this frame -- no scrollbar to hit
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = None
        before = list(app.cfg.saved_cracks)
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 500, 500, 0, None)
        self.assertEqual(app.cfg.temp_start, None, "a click while help is open must not start a new crack")
        self.assertEqual(app.cfg.saved_cracks, before)

    def test_click_outside_scrollbar_bounds_is_ignored(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_offset = 5
        app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 500, 500, 0, None)  # well outside the track
        self.assertEqual(app.cfg.help_scroll_offset, 5, "offset must not change for a click outside the scrollbar")
        self.assertFalse(app.cfg.help_scrollbar_dragging, "a click outside the track must not start a drag")

    def test_click_at_top_of_track_jumps_to_offset_zero(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scroll_offset = 10
        with mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 1127, 300, 0, None)
        self.assertEqual(app.cfg.help_scroll_offset, 0)
        mrz.assert_called_once()

    def test_click_at_bottom_of_track_jumps_to_max_offset(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scroll_offset = 0
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 1127, 830, 0, None)
        self.assertEqual(app.cfg.help_scroll_offset, 20)

    def test_click_at_track_midpoint_jumps_to_half_offset(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)  # midpoint y = 565
        app.cfg.help_scroll_max_offset = 20
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 1127, 565, 0, None)
        self.assertEqual(app.cfg.help_scroll_offset, 10)

    def test_lbuttondown_in_track_starts_dragging(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 1127, 400, 0, None)
        self.assertTrue(app.cfg.help_scrollbar_dragging)

    def test_mousemove_while_dragging_keeps_following_the_mouse(self):
        # USER REPORT: this is the exact interaction that used to do
        # nothing -- pressing down, then moving the mouse without
        # releasing, expecting the list to keep scrolling along with it.
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scrollbar_dragging = True
        with mock.patch.object(app, "refresh_zoom_viewport") as mrz:
            app.mouse_callback(cv2.EVENT_MOUSEMOVE, 1127, 565, 0, None)
        self.assertEqual(app.cfg.help_scroll_offset, 10)
        mrz.assert_called_once()

    def test_mousemove_without_an_active_drag_does_not_scroll(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scroll_offset = 7
        app.cfg.help_scrollbar_dragging = False  # never pressed down inside the track
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, 1127, 565, 0, None)
        self.assertEqual(app.cfg.help_scroll_offset, 7, "moving the mouse without dragging must not scroll")

    def test_lbuttonup_ends_the_drag(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scrollbar_dragging = True
        app.mouse_callback(cv2.EVENT_LBUTTONUP, 1127, 565, 0, None)
        self.assertFalse(app.cfg.help_scrollbar_dragging)

    def test_drag_past_the_track_bottom_clamps_to_max_offset(self):
        # A real scrollbar keeps following the mouse even past its own
        # edges instead of stopping -- the end result should just clamp.
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scrollbar_dragging = True
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_MOUSEMOVE, 1127, 2000, 0, None)  # way below the track
        self.assertEqual(app.cfg.help_scroll_offset, 20)

    def test_drag_past_the_track_top_clamps_to_zero(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = (1120, 300, 1135, 830)
        app.cfg.help_scroll_max_offset = 20
        app.cfg.help_scrollbar_dragging = True
        with mock.patch.object(app, "refresh_zoom_viewport"):
            app.mouse_callback(cv2.EVENT_MOUSEMOVE, 1127, -500, 0, None)  # way above the track
        self.assertEqual(app.cfg.help_scroll_offset, 0)

    def test_mouse_move_and_release_are_also_suppressed_while_help_is_open(self):
        app = make_app()
        app.cfg.show_help_menu = True
        app.cfg.help_scrollbar_rect = None
        app.cfg.current_tool = 'crack'
        app.cfg.temp_start = (10, 10)
        app.mouse_callback(cv2.EVENT_MOUSEMOVE, 500, 500, 0, None)
        app.mouse_callback(cv2.EVENT_LBUTTONUP, 500, 500, 0, None)
        self.assertEqual(app.cfg.temp_start, (10, 10), "help-open suppression must cover every mouse event, not just clicks")


class TestPromptHooksDefaultBehavior(unittest.TestCase):

    def test_prompt_mode_choice_returns_stripped_input(self):
        app = make_app()
        with mock.patch("builtins.input", return_value="  2  "):
            self.assertEqual(app._prompt_mode_choice(), "2")

    def test_prompt_mode_choice_defaults_to_1_on_exception(self):
        app = make_app()
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(app._prompt_mode_choice(), "1")

    def test_prompt_pixel_scale_valid(self):
        app = make_app()
        with mock.patch("builtins.input", return_value="0.042"):
            self.assertAlmostEqual(app._prompt_pixel_scale(0.05), 0.042)

    def test_prompt_pixel_scale_invalid_returns_none(self):
        app = make_app()
        with mock.patch("builtins.input", return_value="not-a-number"):
            self.assertIsNone(app._prompt_pixel_scale(0.05))

    def test_prompt_calibration_distance_valid(self):
        app = make_app()
        with mock.patch("builtins.input", return_value="12.5"):
            self.assertAlmostEqual(app._prompt_calibration_distance(), 12.5)

    def test_prompt_calibration_distance_invalid_returns_none(self):
        app = make_app()
        with mock.patch("builtins.input", return_value="oops"):
            self.assertIsNone(app._prompt_calibration_distance())

    def test_k_key_uses_prompt_hook_and_only_applies_positive_values(self):
        app = make_app()
        app.cfg.PIXEL_TO_CM_SCALE = 0.05
        with mock.patch.object(app, "_prompt_pixel_scale", return_value=0.09) as mprompt:
            app.handle_keyboard(ord('k'))
        mprompt.assert_called_once_with(0.05)
        self.assertAlmostEqual(app.cfg.PIXEL_TO_CM_SCALE, 0.09)

        app.cfg.PIXEL_TO_CM_SCALE = 0.05
        with mock.patch.object(app, "_prompt_pixel_scale", return_value=-1.0):
            app.handle_keyboard(ord('k'))
        self.assertAlmostEqual(app.cfg.PIXEL_TO_CM_SCALE, 0.05)

        app.cfg.PIXEL_TO_CM_SCALE = 0.05
        with mock.patch.object(app, "_prompt_pixel_scale", return_value=None):
            app.handle_keyboard(ord('k'))
        self.assertAlmostEqual(app.cfg.PIXEL_TO_CM_SCALE, 0.05)


class TestHeadlessInertness(unittest.TestCase):

    def test_wait_key_default_calls_real_cv2_waitkey(self):
        app = make_app()
        with mock.patch("cv2.waitKey", return_value=42) as mwait:
            result = app._wait_key(30)
        mwait.assert_called_once_with(30)
        self.assertEqual(result, 42)

    def test_acknowledge_save_warning_default_shows_popup_and_waits(self):
        app = make_app()
        with mock.patch("cv2.namedWindow"), mock.patch("cv2.imshow"), \
             mock.patch("cv2.moveWindow"), mock.patch("cv2.setWindowProperty"), \
             mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1200, 900)), \
             mock.patch("cv2.waitKey") as mwait, mock.patch("cv2.destroyWindow") as mdestroy:
            app._acknowledge_save_warning()
        mwait.assert_called_once_with(0)
        mdestroy.assert_called_once_with("Save Warning")

    def test_confirm_crack_filter_removal_default_returns_true_on_y(self):
        app = make_app()
        with mock.patch("cv2.namedWindow"), mock.patch("cv2.imshow"), \
             mock.patch("cv2.moveWindow"), mock.patch("cv2.setWindowProperty"), \
             mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1200, 900)), \
             mock.patch("cv2.waitKey", return_value=ord('y')), mock.patch("cv2.destroyWindow"):
            result = app._confirm_crack_filter_removal("BLDG001", ["a.json", "b.json"], [0])
        self.assertTrue(result)

    def test_confirm_crack_filter_removal_default_returns_false_on_other_key(self):
        app = make_app()
        with mock.patch("cv2.namedWindow"), mock.patch("cv2.imshow"), \
             mock.patch("cv2.moveWindow"), mock.patch("cv2.setWindowProperty"), \
             mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1200, 900)), \
             mock.patch("cv2.waitKey", return_value=ord('n')), mock.patch("cv2.destroyWindow"):
            result = app._confirm_crack_filter_removal("BLDG001", ["a.json", "b.json"], [0])
        self.assertFalse(result)

    def test_report_fatal_error_is_a_noop_by_default(self):
        app = make_app()
        result = app._report_fatal_error("some message")
        self.assertIsNone(result)  # no exception, no window, just does nothing

    def test_create_display_window_default_matches_original_calls(self):
        app = make_app()
        with mock.patch("cv2.namedWindow") as mnw, \
             mock.patch("cv2.setWindowProperty") as mswp, \
             mock.patch("cv2.setMouseCallback") as msmc:
            app._create_display_window("Crack Detector Workspace")
        mnw.assert_called_once_with("Crack Detector Workspace", cv2.WINDOW_NORMAL)
        self.assertEqual(mswp.call_count, 2)  # fullscreen + topmost
        msmc.assert_called_once_with("Crack Detector Workspace", app.mouse_callback)

    def test_display_frame_default_is_plain_imshow(self):
        app = make_app()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with mock.patch("cv2.imshow") as mshow:
            app._display_frame("Crack Detector Workspace", frame)
        mshow.assert_called_once_with("Crack Detector Workspace", frame)

    def test_render_scene_display_routes_through_display_frame_hook(self):
        # Any GUI override of _display_frame() must be reached by
        # render_scene()'s actual display step -- proven by overriding
        # it and confirming cv2.imshow itself is never called directly.
        tmpdir = tempfile.mkdtemp(prefix="crackseg_displayhook_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=1)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            captured = []
            with mock.patch.object(app, "_display_frame", side_effect=lambda w, f: captured.append((w, f))), \
                 mock.patch("cv2.imshow") as mshow, \
                 mock.patch("cv2.getWindowImageRect", return_value=(0, 0, 1200, 900)), \
                 mock.patch("cv2.namedWindow"), mock.patch("cv2.setWindowProperty"), \
                 mock.patch("cv2.setMouseCallback"):
                app.initialize_image_session(os.path.join(tmpdir, "Images", "img_00.png"))
                app._window_name = "Crack Detector Workspace"
                app.render_scene(1, 1)

            mshow.assert_not_called()
            self.assertTrue(captured, "render_scene() should have displayed at least one frame via the hook")
            self.assertEqual(captured[-1][0], "Crack Detector Workspace")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_pending_keys_starts_empty(self):
        app = make_app()
        self.assertEqual(app._pending_keys, [])

    def test_pump_extra_events_is_a_noop(self):
        app = make_app()
        result = app._pump_extra_events()
        self.assertIsNone(result)
        self.assertEqual(app._pending_keys, [])


class TestButtonPathMatchesKeyPath(unittest.TestCase):

    def test_queued_synthetic_key_and_real_key_produce_identical_state(self):
        app_key = make_app()
        app_key.cfg.JSON_OUTPUT_PATH = ""
        app_key.cfg.saved_cracks = [{'path': [(0, 0)]}]
        with mock.patch.object(app_key, "recalculate_masks"), \
             mock.patch.object(app_key, "refresh_zoom_viewport"):
            app_key.process_keypress(ord('x'))

        app_btn = make_app()
        app_btn.cfg.JSON_OUTPUT_PATH = ""
        app_btn.cfg.saved_cracks = [{'path': [(0, 0)]}]
        app_btn._pending_keys.append(ord('x'))
        with mock.patch.object(app_btn, "recalculate_masks"), \
             mock.patch.object(app_btn, "refresh_zoom_viewport"):
            while app_btn._pending_keys:
                app_btn.process_keypress(app_btn._pending_keys.pop(0))

        self.assertEqual(app_key.cfg.saved_cracks, [])
        self.assertEqual(app_btn.cfg.saved_cracks, [])


class TestFatalErrorOnEmptyImageFolder(unittest.TestCase):
    """build_chronological_queue()'s "no valid images found" exit --
    verifies the new _report_fatal_error() hook is called with the exact
    printed message BEFORE sys.exit(1), and that a plain (non-GUI)
    CrackSegmentation still exits exactly as before (hook is a no-op)."""

    def test_empty_images_folder_calls_hook_then_exits(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_empty_")
        try:
            os.makedirs(os.path.join(tmpdir, "Images"), exist_ok=True)  # empty, no images
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            calls = []
            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "_report_fatal_error", side_effect=lambda m: calls.append(m)):
                with self.assertRaises(SystemExit) as ctx:
                    app.build_chronological_queue()

            self.assertEqual(ctx.exception.code, 1)
            self.assertEqual(len(calls), 1)
            self.assertIn(os.path.join(tmpdir, "Images"), calls[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_hook_is_noop_and_still_exits(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_empty2_")
        try:
            os.makedirs(os.path.join(tmpdir, "Images"), exist_ok=True)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)  # no override -- exactly like every previous delivery

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"):
                with self.assertRaises(SystemExit) as ctx:
                    app.build_chronological_queue()
            self.assertEqual(ctx.exception.code, 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestRunEndToEndHeadless(unittest.TestCase):
    """These tests exercise the REAL run() method (image queue iteration,
    process_keypress dispatch, CSV report writing) end to end. render_scene()
    itself is mocked out: it queries the real OS window's pixel geometry via
    cv2.getWindowImageRect() in several places (see e.g. line ~4281), which
    has no meaningful headless/no-display substitute and is unrelated to the
    GUI-shell changes under test here -- the project's own prior test suites
    followed the same approach (method-level headless tests with the display
    itself mocked out, per the delivery notes)."""

    def test_run_processes_one_image_via_s_key(self):
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=1)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 _MockedHighGui(key_sequence=[ord('s')]):
                app.run()

            json_files = [f for f in os.listdir(_REAL_ARCHIVE_DIR) if f.lower().endswith(".json")] \
                if os.path.isdir(_REAL_ARCHIVE_DIR) else []
            self.assertTrue(json_files,
                             "S should have exported a LabelMe JSON before archiving")
            with open(os.path.join(_REAL_ARCHIVE_DIR, json_files[0]), encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("shapes", data)

            self.assertTrue(os.path.exists(_REAL_CSV_PATH) or
                             os.path.exists(os.path.join(tmpdir, "segmentation_summary_report.csv")),
                             "run() should still write the CSV report at the end")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()

    def test_next_image_button_skips_without_saving_then_second_image_saves(self):
        """USER REQUEST: browse forward without exporting/archiving anything.
        Two images queued; code 4 (Next Image, no save) on the first,
        then a real [S] on the second -- only the second should ever be
        archived."""
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_nav_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=2)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 _MockedHighGui(key_sequence=[4, ord('s')]):
                app.run()

            json_files = sorted(f for f in os.listdir(_REAL_ARCHIVE_DIR) if f.lower().endswith(".json")) \
                if os.path.isdir(_REAL_ARCHIVE_DIR) else []
            self.assertEqual(len(json_files), 1, f"expected exactly one archived JSON (the 2nd image only), got {json_files}")
            self.assertIn("img_01", json_files[0], "the archived file should be the SECOND image, not the skipped first one")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()

    def test_previous_image_button_moves_back_one_position(self):
        """code 5 (Previous Image) on the second image should return to the
        first, then a real [S] there archives THAT one (img_00), proving
        queue_pos actually moved backward rather than just staying put."""
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_prev_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=2)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir
            app = CrackSegmentation(cfg)

            # img_00 (skip forward, no save) -> img_01 (go back, no save) -> img_00 (save)
            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 _MockedHighGui(key_sequence=[4, 5, ord('s')]):
                app.run()

            json_files = [f for f in os.listdir(_REAL_ARCHIVE_DIR) if f.lower().endswith(".json")] \
                if os.path.isdir(_REAL_ARCHIVE_DIR) else []
            self.assertEqual(len(json_files), 1)
            self.assertIn("img_00", json_files[0], "Previous Image should have gone back to the first image")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()

    def test_run_processes_one_image_via_gui_button_queue(self):
        """Same as above, but the 'save' comes from a GUI button (appended to
        _pending_keys by a _pump_extra_events override) instead of a real
        cv2.waitKey() keystroke."""
        tmpdir = tempfile.mkdtemp(prefix="crackseg_test_gui_")
        try:
            make_synthetic_image_folder(tmpdir, n_images=1)
            cfg = Config()
            cfg.SCRIPT_DIR = tmpdir

            class FakePanelApp(CrackSegmentation):
                def __init__(self, cfg):
                    super().__init__(cfg)
                    self._armed = True

                def _pump_extra_events(self):
                    if self._armed:
                        self._armed = False
                        self._pending_keys.append(ord('s'))
                    return None

            app = FakePanelApp(cfg)

            with mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
                 mock.patch.object(app, "render_scene"), \
                 _MockedHighGui():
                app.run()

            json_files = [f for f in os.listdir(_REAL_ARCHIVE_DIR) if f.lower().endswith(".json")] \
                if os.path.isdir(_REAL_ARCHIVE_DIR) else []
            self.assertTrue(json_files,
                             "the queued button click should have exported/saved, exactly like a real S keypress")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _cleanup_real_module_dir_artifacts()


if __name__ == "__main__":
    unittest.main(verbosity=2)
