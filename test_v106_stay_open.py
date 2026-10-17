"""test_v106_stay_open.py

USER REPORT (v1.0.6): with every photo already processed, reopening the app closed it at once
("pipeline exhausted"), so Validation could not be run. The GUI now stays open on an idle screen:
Validation, Switch Mode and Quit keep working; only Quit / Esc / closing the window exits.

Run with:  python3 -m unittest test_v106_stay_open -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from config import Config
from smart_segmentation import CrackSegmentation
from test_gui_shell_parity import _MockedHighGui, _cleanup_real_module_dir_artifacts
from test_gui_panel_wiring import make_app as make_gui_app


def _all_processed_folder(tmpdir):
    """'Images' holds one photo that already has its JSON: Mode 1 has nothing left to do."""
    images = os.path.join(tmpdir, "Images")
    jsons = os.path.join(tmpdir, "JSON files")
    os.makedirs(images)
    os.makedirs(jsons)
    cv2.imwrite(os.path.join(images, "wall__BLDG001.png"), np.full((120, 160, 3), 200, dtype=np.uint8))
    with open(os.path.join(jsons, "wall__BLDG001.json"), "w", encoding="utf-8") as f:
        json.dump({"shapes": [], "imagePath": "wall__BLDG001.png", "imageHeight": 120, "imageWidth": 160}, f)


class TestStayOpenWhenNothingIsLeft(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="crackseg_stayopen_")
        _all_processed_folder(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _cleanup_real_module_dir_artifacts()

    def _app(self):
        cfg = Config()
        cfg.SCRIPT_DIR = self.tmpdir
        return CrackSegmentation(cfg)

    def test_empty_mode1_start_no_longer_exits(self):
        app = self._app()
        app.cfg.image_queue = []
        with mock.patch.object(app, "_keep_open_when_queue_empty", return_value=True):
            app._finalize_queue_or_exit()   # used to sys.exit(1)

    def test_plain_opencv_window_keeps_the_old_behaviour(self):
        app = self._app()
        app.cfg.image_queue = []
        with mock.patch.object(app, "_report_fatal_error"), self.assertRaises(SystemExit):
            app._finalize_queue_or_exit()

    def test_idle_then_switch_mode_opens_the_segmented_photos(self):
        app = self._app()
        seen = []
        original_start = app._start_image_session

        def _record(queue_pos, index, total_files):
            seen.append((app.cfg.modalita_scelta, os.path.basename(app.cfg.image_queue[queue_pos])))
            return original_start(queue_pos, index, total_files)

        with mock.patch("smart_segmentation.resolve_script_dir", return_value=self.tmpdir), \
             mock.patch.object(app, "_keep_open_when_queue_empty", return_value=True), \
             mock.patch.object(app, "_prompt_mode_choice", return_value="1"), \
             mock.patch.object(app, "_prompt_queue_exhausted", return_value=False), \
             mock.patch.object(app, "render_scene"), \
             mock.patch.object(app, "_start_image_session", side_effect=_record), \
             _MockedHighGui(key_sequence=[6, 27]), \
             self.assertRaises(SystemExit) as ended:
            app.run()   # idle -> Switch Mode -> Mode 2 photo -> Esc
        self.assertEqual(ended.exception.code, 0, "only Quit/Esc ends the session")
        self.assertEqual(seen, [("2", "wall__BLDG001.png")])

    def test_validation_works_while_idle(self):
        app = self._app()
        with mock.patch.object(app, "_keep_open_when_queue_empty", return_value=True), \
             mock.patch.object(app, "run_training_validation", return_value=False) as validate, \
             mock.patch.object(app, "_render_idle_screen"), \
             _MockedHighGui(key_sequence=[17, 27]), \
             self.assertRaises(SystemExit):
            app.cfg.modalita_scelta = "1"
            app._window_name = "Crack Detector Workspace"
            app._wait_for_new_queue()
        validate.assert_called_once()

    def test_validation_is_available_while_idle_even_with_skipped_photos(self):
        app = self._app()
        app.cfg.image_queue = ["/x/a.png", "/x/b.png"]
        app.cfg.CURRENT_IMAGE_PATH = "/x/a.png"
        # v1.0.8 rule: Validation is available once every photo in 'Images' has its JSON, so one photo
        # still without JSON is needed for it to be unavailable while the queue is running.
        cv2.imwrite(os.path.join(self.tmpdir, "Images", "skipped__BLDG001.png"),
                    np.full((120, 160, 3), 200, dtype=np.uint8))
        self.assertFalse(app._validation_available())
        app.cfg.queue_idle = True
        self.assertTrue(app._validation_available())

    def test_clicks_are_ignored_while_idle(self):
        app = self._app()
        app.cfg.queue_idle = True
        with mock.patch.object(app, "_handle_crack_tool_mouse") as crack_tool:
            app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
        crack_tool.assert_not_called()


class TestGuiStaysOpen(unittest.TestCase):

    def test_gui_keeps_the_window_open_and_says_so(self):
        app = make_gui_app(self)
        self.assertTrue(app._keep_open_when_queue_empty())
        self.assertIn("stay here", app._queue_end_message("1", "2", 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
