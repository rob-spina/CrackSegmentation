"""test_v201_delete_image.py

Headless tests for the 2.0.1 [Delete image] button (delete_image.py, code 149):
  - after confirmation the photo, its JSON and its 'Binary files' exports are moved to 'Deleted images' (never
    erased), the photo leaves the queue and the next one opens; its [To be reviewed] mark goes;
  - cancelled: nothing moves; no photo on screen: nothing happens; a name already in 'Deleted images' is kept.

Run with:  python3 -m unittest test_v201_delete_image -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import crack_segmentation_gui as gm
from delete_image import DELETE_IMAGE_CODE, DELETED_DIR_NAME
from test_gui_shell_parity import make_app


class TestDeleteImage(unittest.TestCase):

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="crackseg_delete_")
        self.addCleanup(shutil.rmtree, self.data, True)
        for d in ("Images", "JSON files", "Binary files"):
            os.makedirs(os.path.join(self.data, d))
        self.photos = []
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            path = os.path.join(self.data, "Images", name)
            cv2.imwrite(path, np.full((20, 30, 3), 100, np.uint8))
            self.photos.append(path)
        with open(os.path.join(self.data, "JSON files", "b.json"), "w") as f:
            json.dump({"shapes": []}, f)
        for sfx in ("-seg.png", "-crack_mask.png"):
            open(os.path.join(self.data, "Binary files", "b" + sfx), "w").close()
        open(os.path.join(self.data, "Binary files", "bb-crack_mask.png"), "w").close()   # another photo
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR = self.data
        cfg.queue_idle = False
        cfg.image_queue = list(self.photos)
        cfg.CURRENT_IMAGE_PATH = self.photos[1]
        cfg.JSON_OUTPUT_PATH = os.path.join(self.data, "JSON files", "b.json")

    def _press(self, answer):
        with mock.patch.object(self.app, "_confirm_delete_image", return_value=answer) as ask, \
                mock.patch.object(self.app, "refresh_zoom_viewport"):
            result = self.app.process_keypress(DELETE_IMAGE_CODE)
        return result, ask

    def _ls(self, *parts):
        path = os.path.join(self.data, *parts)
        return sorted(os.listdir(path)) if os.path.isdir(path) else []

    def test_moves_photo_json_and_masks_and_opens_the_next(self):
        index = {"b": {"note": "", "since": ""}}
        self.app._save_review_index(index)
        result, ask = self._press(True)
        self.assertTrue(result, "the image loop moves on")
        self.assertIn("b.jpg", ask.call_args[0][0])
        self.assertIn("its JSON", ask.call_args[0][0])
        self.assertEqual(self._ls("Images"), ["a.jpg", "c.jpg"])
        self.assertEqual(self._ls("JSON files"), [])
        self.assertEqual(self._ls("Binary files"), ["bb-crack_mask.png"])
        self.assertEqual(self._ls(DELETED_DIR_NAME, "Images"), ["b.jpg"])
        self.assertEqual(self._ls(DELETED_DIR_NAME, "JSON files"), ["b.json"])
        self.assertEqual(self._ls(DELETED_DIR_NAME, "Binary files"), ["b-crack_mask.png", "b-seg.png"])
        self.assertEqual(self.app.cfg.image_queue, [self.photos[0], self.photos[2]])
        self.assertTrue(self.app.cfg.queue_item_consumed)
        self.assertEqual(self.app._load_review_index(), {})

    def test_cancel_keeps_everything(self):
        result, _ = self._press(False)
        self.assertFalse(result)
        self.assertEqual(self._ls("Images"), ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(self._ls(DELETED_DIR_NAME), [])

    def test_photo_without_json(self):
        self.app.cfg.CURRENT_IMAGE_PATH = self.photos[0]
        self.app.cfg.JSON_OUTPUT_PATH = os.path.join(self.data, "JSON files", "a.json")
        result, ask = self._press(True)
        self.assertTrue(result)
        self.assertNotIn("its JSON", ask.call_args[0][0])
        self.assertEqual(self._ls(DELETED_DIR_NAME, "Images"), ["a.jpg"])

    def test_same_name_deleted_twice_is_kept(self):
        os.makedirs(os.path.join(self.data, DELETED_DIR_NAME, "Images"))
        open(os.path.join(self.data, DELETED_DIR_NAME, "Images", "b.jpg"), "w").close()
        self._press(True)
        self.assertEqual(len(self._ls(DELETED_DIR_NAME, "Images")), 2)

    def test_idle_does_nothing(self):
        self.app.cfg.queue_idle = True
        with mock.patch.object(self.app, "_confirm_delete_image") as ask:
            self.assertFalse(self.app.run_delete_image())
        ask.assert_not_called()

    def test_gui_button_menu_and_lock(self):
        codes = [entry[3] for _, entries in gm._LEFT_SIDEBAR_SECTIONS for entry in entries]
        self.assertIn(DELETE_IMAGE_CODE, codes)
        self.assertIn(DELETE_IMAGE_CODE, [code for _, code in gm._FILE_COMMANDS])
        cfg = self.app.cfg
        self.assertTrue(gm._SIDEBAR_ENABLED_STATES[DELETE_IMAGE_CODE](cfg))
        cfg.queue_idle = True
        self.assertFalse(gm._SIDEBAR_ENABLED_STATES[DELETE_IMAGE_CODE](cfg))


if __name__ == "__main__":
    unittest.main()
