"""test_v230_building_filter.py

Headless tests for the 2.3.0 [Filter by building] button (building_filter.py, code 163):
  - the building groups of 'Images' (tag __BLDGnnn) with their labelled / to-label photos;
  - the three choices: labelled (Mode 2), to label (Mode 1), all (each photo in the mode it needs; a photo labelled
    in Mode 1 stays in the queue);
  - Switch Mode / a rebuilt queue turns the filter off; the button is in the left sidebar and in the Tools menu.

Run with:  python3 -m unittest test_v230_building_filter -v
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import building_filter as bf
import crack_segmentation_gui as gm
from test_gui_shell_parity import make_app


class _Data(unittest.TestCase):

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="crackseg_bf_")
        self.addCleanup(shutil.rmtree, self.data, True)
        for d in ("Images", "JSON files", "Binary files"):
            os.makedirs(os.path.join(self.data, d))
        self.app = make_app()
        self.app.cfg.SCRIPT_DIR = self.data
        self.app.cfg.IMAGE_FOLDER = os.path.join(self.data, "Images")
        self.app.cfg.OUTPUT_FOLDER = os.path.join(self.data, "JSON files")
        self.app.cfg.queue_idle = True
        self._photo("a__BLDG024.jpg", labelled=True)
        self._photo("b__BLDG024.jpg")
        self._photo("c__BLDG024.jpg", labelled=True)
        self._photo("d__BLDG3.jpg")
        self._photo("e__BLDG100.jpg", labelled=True)
        self._photo("untagged.jpg")
        self._photo("a__BLDG024-seg.jpg")

    def _photo(self, name, labelled=False):
        path = os.path.join(self.data, "Images", name)
        cv2.imwrite(path, np.full((20, 30, 3), 120, np.uint8))
        if labelled:
            with open(os.path.join(self.data, "JSON files", os.path.splitext(name)[0] + ".json"), "w") as f:
                json.dump({"shapes": []}, f)
        return path

    def _press(self, choice):
        with mock.patch.object(self.app, "_prompt_building_filter", return_value=choice) as ask, \
                mock.patch.object(self.app, "_notify_building_filter") as told, redirect_stdout(io.StringIO()):
            result = self.app._handle_idle_key(bf.BUILDING_FILTER_CODE)
        return result, ask, told

    def names(self):
        return [os.path.basename(p) for p in self.app.cfg.image_queue]


class TestGroups(_Data):

    def test_tag(self):
        self.assertEqual(bf.building_tag("x/abc__bldg024.JPG"), "BLDG024")
        self.assertIsNone(bf.building_tag("abc.jpg"))

    def test_groups_counts_and_order(self):
        g = bf.building_groups_in(self.data)
        self.assertEqual(list(g), ["BLDG3", "BLDG024", "BLDG100"])
        self.assertEqual([os.path.basename(p) for p in g["BLDG024"]["labelled"]], ["a__BLDG024.jpg", "c__BLDG024.jpg"])
        self.assertEqual([os.path.basename(p) for p in g["BLDG024"]["to_label"]], ["b__BLDG024.jpg"])
        self.assertNotIn(None, g)

    def test_filtered_photos(self):
        g = bf.building_groups_in(self.data)
        self.assertEqual(len(bf.filtered_photos(g, "BLDG024", bf.BUILDING_FILTER_LABELLED)), 2)
        self.assertEqual(len(bf.filtered_photos(g, "BLDG024", bf.BUILDING_FILTER_TO_LABEL)), 1)
        self.assertEqual(len(bf.filtered_photos(g, "BLDG024", bf.BUILDING_FILTER_ALL)), 3)
        self.assertEqual(bf.filtered_photos(g, "BLDG999", bf.BUILDING_FILTER_ALL), [])


class TestButton(_Data):

    def test_labelled_opens_in_mode_2(self):
        result, ask, _ = self._press(("BLDG024", bf.BUILDING_FILTER_LABELLED))
        self.assertTrue(result)
        self.assertEqual(list(ask.call_args[0][0]), ["BLDG3", "BLDG024", "BLDG100"])
        self.assertEqual(self.names(), ["a__BLDG024.jpg", "c__BLDG024.jpg"])
        self.assertEqual(self.app.cfg.modalita_scelta, "2")
        self.assertFalse(self.app.building_filter_mixed())

    def test_to_label_opens_in_mode_1(self):
        self.assertTrue(self._press(("BLDG024", bf.BUILDING_FILTER_TO_LABEL))[0])
        self.assertEqual(self.names(), ["b__BLDG024.jpg"])
        self.assertEqual(self.app.cfg.modalita_scelta, "1")

    def test_all_switches_mode_photo_by_photo(self):
        self.assertTrue(self._press(("BLDG024", bf.BUILDING_FILTER_ALL))[0])
        self.assertEqual(self.names(), ["a__BLDG024.jpg", "b__BLDG024.jpg", "c__BLDG024.jpg"])
        self.assertTrue(self.app.building_filter_mixed())
        modes = []
        for p in self.app.cfg.image_queue:
            self.app._apply_building_filter_mode(p)
            modes.append(self.app.cfg.modalita_scelta)
        self.assertEqual(modes, ["2", "1", "2"])

    def test_cancel_and_empty_choice(self):
        result, _, told = self._press(None)
        self.assertFalse(result)
        result, _, told = self._press(("BLDG3", bf.BUILDING_FILTER_LABELLED))
        self.assertFalse(result)
        self.assertIn("BLDG3", told.call_args[0][0])

    def test_no_tagged_photo(self):
        for f in os.listdir(os.path.join(self.data, "Images")):
            os.remove(os.path.join(self.data, "Images", f))
        result, ask, told = self._press(("BLDG024", bf.BUILDING_FILTER_ALL))
        self.assertFalse(result)
        ask.assert_not_called()
        self.assertIn("building tag", told.call_args[0][0])

    def test_active_queue_jumps_to_first(self):
        self.app.cfg.queue_idle = False
        self.app.cfg.CURRENT_IMAGE_PATH = ""
        with mock.patch.object(self.app, "_prompt_building_filter", return_value=("BLDG100", "all")), \
                redirect_stdout(io.StringIO()):
            self.assertTrue(self.app.process_keypress(bf.BUILDING_FILTER_CODE))
        self.assertEqual(self.app.cfg.navigate_target, 0)
        self.assertEqual(self.names(), ["e__BLDG100.jpg"])


class TestQueueBehaviour(_Data):

    def _save_in_mode_1(self, photo):
        cfg = self.app.cfg
        cfg.modalita_scelta = "1"
        cfg.CURRENT_IMAGE_PATH = photo
        cfg.JSON_OUTPUT_PATH = os.path.join(self.data, "JSON files", "b__BLDG024.json")
        with open(cfg.JSON_OUTPUT_PATH, "w") as f:
            json.dump({"shapes": []}, f)
        with redirect_stdout(io.StringIO()):
            self.app._archive_saved_files(os.path.join(self.data, "JSON files"))

    def test_mixed_queue_keeps_the_photo_just_labelled(self):
        self._press(("BLDG024", bf.BUILDING_FILTER_ALL))
        photo = self.app.cfg.image_queue[1]
        self._save_in_mode_1(photo)
        self.assertIn(photo, self.app.cfg.image_queue)
        self.assertFalse(self.app.cfg.queue_item_consumed)
        self.app._apply_building_filter_mode(photo)
        self.assertEqual(self.app.cfg.modalita_scelta, "2")

    def test_to_label_queue_drops_it_as_mode_1_does(self):
        self._press(("BLDG024", bf.BUILDING_FILTER_TO_LABEL))
        photo = self.app.cfg.image_queue[0]
        self._save_in_mode_1(photo)
        self.assertNotIn(photo, self.app.cfg.image_queue)

    def test_rebuilt_queue_turns_the_filter_off(self):
        self._press(("BLDG024", bf.BUILDING_FILTER_ALL))
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.app._rebuild_queue_for_mode("1"))
        self.assertEqual(self.app.cfg.building_filter, {})
        self.assertIn("untagged.jpg", self.names())

    def test_failed_rebuild_keeps_the_filter(self):
        self._press(("BLDG024", bf.BUILDING_FILTER_ALL))
        kept = list(self.app.cfg.image_queue)
        for f in os.listdir(os.path.join(self.data, "Images")):
            os.remove(os.path.join(self.data, "Images", f))
        with redirect_stdout(io.StringIO()):
            self.assertFalse(self.app._rebuild_queue_for_mode("1"))
        self.assertEqual(self.app.cfg.image_queue, kept)
        self.assertEqual(self.app.cfg.building_filter["tag"], "BLDG024")
        self.assertEqual(len(self.app.cfg.image_queue), 3)


class TestGui(_Data):

    def test_button_and_menu_entry(self):
        codes = [b[3] for _, buttons in gm._LEFT_SIDEBAR_SECTIONS for b in buttons]
        self.assertIn(bf.BUILDING_FILTER_CODE, codes)
        self.assertIn(bf.BUILDING_FILTER_CODE, [c for _, c in gm._TOOL_COMMANDS])

    def test_dialog_choice(self):
        groups = bf.building_groups_in(self.data)
        d = gm._BuildingFilterDialog(fake_tkinter.Tk(), groups)
        self.assertEqual(d.which.get(), bf.BUILDING_FILTER_ALL)
        self.assertIn("3. All (1)", d.radios[bf.BUILDING_FILTER_ALL].kw["text"])
        d.combo.pick(1)
        self.assertIn("1. Photos already labelled (2)", d.radios[bf.BUILDING_FILTER_LABELLED].kw["text"])
        d.which.set(bf.BUILDING_FILTER_TO_LABEL)
        d._ok()
        self.assertEqual(d.result, ("BLDG024", bf.BUILDING_FILTER_TO_LABEL))


if __name__ == "__main__":
    unittest.main()


class TestUnsavedChangesAfterSave(unittest.TestCase):
    """v2.3.1: a photo saved several times must not be reported as modified (the undo history survives a save)."""

    def _app(self):
        import types
        from dataset_split import DatasetSplitMixin
        app = DatasetSplitMixin.__new__(DatasetSplitMixin)
        app.cfg = types.SimpleNamespace(action_history=[], saved_cracks=[], saved_detachments=[], saved_ignores=[],
                                        CURRENT_IMAGE_PATH="/x/a__BLDG001.jpg")
        return app

    def test_clean_after_save_dirty_after_edit(self):
        app = self._app()
        self.assertFalse(app.has_unsaved_changes(), "nothing done yet")
        app.cfg.action_history.append("crack_start")
        app.cfg.saved_cracks.append({"path": [(1, 2), (3, 4)]})
        self.assertTrue(app.has_unsaved_changes(), "never saved")
        app.mark_annotations_saved()
        self.assertFalse(app.has_unsaved_changes(), "just saved, history still full")
        app.cfg.saved_cracks.append({"path": [(5, 6), (7, 8)]})
        app.cfg.action_history.append("crack_start")
        self.assertTrue(app.has_unsaved_changes(), "edited after the save")

    def test_other_photo_with_same_shapes_is_not_clean(self):
        app = self._app()
        app.cfg.action_history.append("crack_start")
        app.mark_annotations_saved()
        app.cfg.CURRENT_IMAGE_PATH = "/x/b__BLDG001.jpg"
        self.assertTrue(app.has_unsaved_changes())

    def test_saved_photo_is_not_asked_again(self):
        app = self._app()
        app.cfg.action_history.append("crack_start")
        app.mark_annotations_saved()
        asked = []
        self.assertTrue(app._settle_unsaved_changes("filter by building", ask=lambda n: asked.append(n)))
        self.assertEqual(asked, [])
