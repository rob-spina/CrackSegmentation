"""test_v201_review.py

Headless tests for the 2.0.1 [To be reviewed] button (review_flags.py, code 140):
  - the button marks / unmarks the photo on screen in '<data>/to_be_reviewed.json' (with an optional note) and in
    the photo's JSON "flags"; Cancel on the note does not mark it; the photo keeps its name;
  - a mark survives the building tag (__BLDG001) and capital extensions; saving keeps the flags in the JSON;
  - Mode 2 (start-up and Switch Mode) asks which photos to load only when both groups have photos, and filters
    the queue; the choice is reapplied on silent reloads; photos marked only in their JSON are imported.

Run with:  python3 -m unittest test_v201_review -v
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
import review_flags as rf
from review_flags import REVIEW_CODE, review_key
from test_gui_shell_parity import make_app


def _press(app, code):
    with mock.patch.object(app, "refresh_zoom_viewport"):
        return app.process_keypress(code)


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images = os.path.join(self.tmp, "Images")
        self.jsons = os.path.join(self.tmp, "JSON files")
        os.makedirs(self.images)
        os.makedirs(self.jsons)
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR = self.tmp
        cfg.OUTPUT_FOLDER = self.jsons
        cfg.IMAGE_FOLDER = self.images

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _photo(self, name, with_json=True, flags=None):
        path = os.path.join(self.images, name)
        cv2.imwrite(path, np.full((20, 30, 3), 128, np.uint8))
        if with_json:
            base = os.path.splitext(name)[0]
            with open(os.path.join(self.jsons, base + ".json"), "w", encoding="utf-8") as f:
                json.dump({"version": "5.0.1", "flags": flags or {}, "shapes": [], "imagePath": name,
                           "imageData": "AAAA" * 100}, f, indent=2)
        return path

    def _show(self, path):
        cfg = self.app.cfg
        cfg.CURRENT_IMAGE_PATH = path
        cfg.JSON_OUTPUT_PATH = os.path.join(self.jsons, os.path.splitext(os.path.basename(path))[0] + ".json")

    def _json_flags(self, path):
        with open(os.path.join(self.jsons, os.path.splitext(os.path.basename(path))[0] + ".json"),
                  encoding="utf-8") as f:
            return json.load(f)["flags"]

    def _index(self):
        with open(os.path.join(self.tmp, rf.REVIEW_INDEX_NAME), encoding="utf-8") as f:
            return json.load(f)


class ReviewKeyTest(unittest.TestCase):

    def test_building_tag_extension_and_seg_are_ignored(self):
        self.assertEqual(review_key("/x/DJI_0001.JPG"), "dji_0001")
        self.assertEqual(review_key("/x/DJI_0001__BLDG003.jpg"), "dji_0001")
        self.assertEqual(review_key("DJI_0001__BLDG003-seg.json"), "dji_0001")
        self.assertEqual(review_key("DJI_0001.json"), "dji_0001")


class ToggleTest(_Base):

    def test_mark_with_note_then_unmark(self):
        photo = self._photo("wall.jpg")
        self._show(photo)
        with mock.patch.object(self.app, "_prompt_review_note", return_value="crack 3 doubtful"):
            _press(self.app, REVIEW_CODE)
        self.assertTrue(self.app.is_photo_to_be_reviewed())
        self.assertEqual(self._index()["wall"]["note"], "crack 3 doubtful")
        self.assertEqual(self._json_flags(photo), {"to_be_reviewed": True, "review_note": "crack 3 doubtful"})
        self.assertTrue(os.path.exists(photo))            # the photo keeps its name
        self.assertEqual(self.app.review_flags_for_json()["review_note"], "crack 3 doubtful")

        _press(self.app, REVIEW_CODE)                     # pressed again: unmarked, no prompt
        self.assertFalse(self.app.is_photo_to_be_reviewed())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, rf.REVIEW_INDEX_NAME)))
        self.assertEqual(self._json_flags(photo), {})

    def test_cancel_does_not_mark(self):
        self._show(self._photo("wall.jpg"))
        with mock.patch.object(self.app, "_prompt_review_note", return_value=None):
            _press(self.app, REVIEW_CODE)
        self.assertFalse(self.app.is_photo_to_be_reviewed())

    def test_other_flags_are_kept(self):
        photo = self._photo("wall.jpg", flags={"other": 1})
        self._show(photo)
        with mock.patch.object(self.app, "_prompt_review_note", return_value=""):
            _press(self.app, REVIEW_CODE)
        self.assertEqual(self._json_flags(photo), {"other": 1, "to_be_reviewed": True})

    def test_mark_survives_building_rename(self):
        self._show(self._photo("wall.jpg"))
        with mock.patch.object(self.app, "_prompt_review_note", return_value=""):
            _press(self.app, REVIEW_CODE)
        self.assertTrue(self.app.is_photo_to_be_reviewed(os.path.join(self.images, "wall__BLDG002.jpg")))

    def test_photo_without_json_is_marked_and_saved_flags_follow(self):
        self._show(self._photo("new.png", with_json=False))
        with mock.patch.object(self.app, "_prompt_review_note", return_value="ask Fabio"):
            _press(self.app, REVIEW_CODE)
        self.assertEqual(self.app._build_labelme_data_dict("")["flags"],
                         {"to_be_reviewed": True, "review_note": "ask Fabio"})

    def test_unmarked_photo_saves_empty_flags(self):
        self._show(self._photo("wall.jpg"))
        self.assertEqual(self.app._build_labelme_data_dict("")["flags"], {})

    def test_hud_label(self):
        self._show(self._photo("wall.jpg"))
        texts = []
        self.app._put_hud_text = lambda win, text, org, color, *a, **k: texts.append(text)
        self.app._draw_review_label(None)
        self.assertEqual(texts, [])
        with mock.patch.object(self.app, "_prompt_review_note", return_value="width"):
            _press(self.app, REVIEW_CODE)
        self.app._draw_review_label(None)
        self.assertEqual(texts, ["TO BE REVIEWED: width"])


class Mode2FilterTest(_Base):

    def setUp(self):
        super().setUp()
        self.a = self._photo("a.jpg")
        self.b = self._photo("b.jpg")
        self.c = self._photo("c.jpg")
        self._photo("new.jpg", with_json=False)
        self._show(self.b)
        with mock.patch.object(self.app, "_prompt_review_note", return_value=""):
            _press(self.app, REVIEW_CODE)

    def _rebuild(self, choice, ask=True):
        with mock.patch.object(self.app, "_prompt_review_filter", return_value=choice) as prompt:
            self.assertTrue(self.app._rebuild_queue_for_mode("2", ask_review_filter=ask))
        return [os.path.basename(p) for p in self.app.cfg.image_queue], prompt

    def test_only_to_be_reviewed(self):
        queue, prompt = self._rebuild(rf.REVIEW_FILTER_ONLY)
        self.assertEqual(queue, ["b.jpg"])
        prompt.assert_called_once_with(1, 2)

    def test_only_others(self):
        queue, _ = self._rebuild(rf.REVIEW_FILTER_OTHERS)
        self.assertEqual(queue, ["a.jpg", "c.jpg"])

    def test_all(self):
        queue, _ = self._rebuild(rf.REVIEW_FILTER_ALL)
        self.assertEqual(queue, ["a.jpg", "b.jpg", "c.jpg"])

    def test_choice_reapplied_without_asking(self):
        self._rebuild(rf.REVIEW_FILTER_ONLY)
        queue, prompt = self._rebuild(rf.REVIEW_FILTER_ALL, ask=False)   # e.g. reload after Validation
        self.assertEqual(queue, ["b.jpg"])
        prompt.assert_not_called()

    def test_no_question_when_no_photo_is_marked(self):
        self._show(self.b)
        _press(self.app, REVIEW_CODE)                     # unmark the only one
        queue, prompt = self._rebuild(rf.REVIEW_FILTER_ONLY)
        prompt.assert_not_called()
        self.assertEqual(queue, ["a.jpg", "b.jpg", "c.jpg"])

    def test_no_question_in_mode_1(self):
        with mock.patch.object(self.app, "_prompt_review_filter") as prompt:
            self.app._rebuild_queue_for_mode("1", ask_review_filter=True)
        prompt.assert_not_called()
        self.assertEqual([os.path.basename(p) for p in self.app.cfg.image_queue], ["new.jpg"])

    def test_empty_filter_result_falls_back_to_all(self):
        self._rebuild(rf.REVIEW_FILTER_ONLY)
        self._show(self.b)
        _press(self.app, REVIEW_CODE)                     # the last one to review is done
        queue, _ = self._rebuild(None, ask=False)
        self.assertEqual(queue, ["a.jpg", "b.jpg", "c.jpg"])

    def test_switch_mode_asks(self):
        self.app.cfg.modalita_scelta = "1"
        self.app.cfg.switch_mode_requested = True
        with mock.patch.object(self.app, "_prompt_review_filter", return_value=rf.REVIEW_FILTER_ONLY) as prompt:
            self.app._advance_or_switch_mode(0, 1)
        prompt.assert_called_once()
        self.assertEqual([os.path.basename(p) for p in self.app.cfg.image_queue], ["b.jpg"])

    def test_startup_mode_2_asks(self):
        with mock.patch.object(self.app, "_prompt_mode_choice", return_value="2"), \
                mock.patch.object(self.app, "_print_pipeline_banner"), \
                mock.patch("smart_segmentation.migrate_legacy_folders"), \
                mock.patch.object(self.app, "_prompt_review_filter", return_value=rf.REVIEW_FILTER_OTHERS):
            self.app.build_chronological_queue()
        self.assertEqual([os.path.basename(p) for p in self.app.cfg.image_queue], ["a.jpg", "c.jpg"])

    def test_marks_in_json_flags_are_imported(self):
        self._photo("d.jpg", flags={"to_be_reviewed": True, "review_note": "from \"Leonardo\""})
        queue, prompt = self._rebuild(rf.REVIEW_FILTER_ONLY)
        self.assertEqual(queue, ["b.jpg", "d.jpg"])
        self.assertEqual(self._index()["d"]["note"], 'from "Leonardo"')
        prompt.assert_called_once_with(2, 2)


class GuiWiringTest(unittest.TestCase):

    def test_sidebar_and_menu(self):
        codes = [entry[3] for _, entries in gm._LEFT_SIDEBAR_SECTIONS for entry in entries]
        self.assertIn(REVIEW_CODE, codes)
        self.assertIn(REVIEW_CODE, [code for _, code in gm._TOOL_COMMANDS])
        self.assertIn(REVIEW_CODE, gm._SIDEBAR_TOGGLE_STATES)

    def test_review_does_not_end_sam_click_to_accept(self):
        import sam_accept
        self.assertIn(REVIEW_CODE, sam_accept.SAM_ACCEPT_KEEP_KEYS)


if __name__ == "__main__":
    unittest.main()
