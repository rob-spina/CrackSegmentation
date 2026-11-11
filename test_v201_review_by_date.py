"""test_v201_review_by_date.py

Headless tests for the 2.0.1 [To be reviewed by date] button (review_flags.py, code 160): every segmented photo whose
JSON was last saved on or before a date is marked TO BE REVIEWED (index + JSON flags, saving date kept); photos
already marked keep their note; photos in the sets are marked too and counted apart; Remove takes away only the
marks given by date; Mode 2 then loads only these photos.

Run with:  python3 -m unittest test_v201_review_by_date -v
"""
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import crack_segmentation_gui as gm
import review_flags as rf
from review_flags import REVIEW_BY_DATE_CODE, review_key
from test_gui_shell_parity import make_app

DAY = 86400


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = make_app()
        cfg = self.app.cfg
        cfg.SCRIPT_DIR = self.tmp
        cfg.IMAGE_FOLDER = os.path.join(self.tmp, "Images")
        cfg.OUTPUT_FOLDER = os.path.join(self.tmp, "JSON files")
        self.today = datetime.date.today()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def photo(self, name, days_ago=0, set_name=None, json_in_set=None, with_photo=True, flags=None):
        """A photo (in 'Images' or a set's 'Images') and its JSON saved `days_ago` days ago."""
        base = os.path.join(self.tmp, set_name) if set_name else self.tmp
        if with_photo:
            os.makedirs(os.path.join(base, "Images"), exist_ok=True)
            cv2.imwrite(os.path.join(base, "Images", name), np.full((8, 8, 3), 128, np.uint8))
        json_base = base if (json_in_set if json_in_set is not None else set_name == "Training set") else self.tmp
        folder = os.path.join(json_base, "JSON files")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, os.path.splitext(name)[0] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": "5.0.1", "flags": flags or {}, "shapes": [], "imagePath": name}, f)
        stamp = time.time() - days_ago * DAY
        os.utime(path, (stamp, stamp))
        return path

    def run_button(self, date=None, note="re-segment", remove=False, confirm=True):
        answer = {"date": date or self.today, "note": note, "remove": remove}
        app = self.app
        with mock.patch.object(app, "_prompt_review_by_date", return_value=answer), \
                mock.patch.object(app, "_confirm_review_by_date", return_value=confirm) as asked, \
                mock.patch.object(app, "_notify_review_by_date") as told, \
                mock.patch.object(app, "refresh_zoom_viewport"):
            result = app.process_keypress(REVIEW_BY_DATE_CODE)
        self.asked = asked.call_args[0][0] if asked.called else ""
        self.told = told.call_args[0][0] if told.called else ""
        return result

    def index(self):
        try:
            with open(os.path.join(self.tmp, rf.REVIEW_INDEX_NAME), encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return {}

    @staticmethod
    def flags(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("flags", {})


class TestParseDate(unittest.TestCase):

    def test_formats(self):
        parse = rf.ReviewFlagsMixin.parse_review_date
        self.assertEqual(parse("2026-11-08"), datetime.date(2026, 11, 8))
        self.assertEqual(parse("08/11/2026"), datetime.date(2026, 11, 8))
        self.assertEqual(parse("8-11-2026"), datetime.date(2026, 11, 8))
        self.assertEqual(parse("08.11.2026"), datetime.date(2026, 11, 8))
        self.assertEqual(parse(""), datetime.date.today())
        self.assertIsNone(parse("ieri"))


class TestMarkByDate(_Base):

    def test_today_marks_everything_saved(self):
        a = self.photo("a.jpg", days_ago=10)
        b = self.photo("b.JPG", days_ago=0)
        self.run_button()
        idx = self.index()
        self.assertEqual(set(idx), {"a", "b"})
        self.assertEqual(idx["a"]["until"], self.today.isoformat())
        self.assertEqual(idx["a"]["note"], "re-segment")
        self.assertEqual(self.flags(a), {"to_be_reviewed": True, "review_note": "re-segment"})
        self.assertTrue(self.flags(b)["to_be_reviewed"])
        self.assertIn("2 photo(s)", self.told)

    def test_only_up_to_the_date(self):
        self.photo("old.jpg", days_ago=5)
        self.photo("new.jpg", days_ago=0)
        self.run_button(date=self.today - datetime.timedelta(days=2))
        self.assertEqual(set(self.index()), {"old"})

    def test_the_whole_day_counts(self):
        path = self.photo("late.jpg")
        end = datetime.datetime.combine(self.today, datetime.time(23, 59)).timestamp()
        os.utime(path, (end, end))
        self.run_button()
        self.assertIn("late", self.index())

    def test_saving_date_is_kept(self):
        path = self.photo("a.jpg", days_ago=7)
        before = os.path.getmtime(path)
        self.run_button()
        self.assertAlmostEqual(os.path.getmtime(path), before, delta=1)

    def test_already_marked_keep_their_note(self):
        self.photo("a.jpg", days_ago=3)
        self.photo("b.jpg", days_ago=3)
        with open(os.path.join(self.tmp, rf.REVIEW_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump({"a": {"note": "my doubt", "since": "2026-10-01"}}, f)
        self.run_button()
        idx = self.index()
        self.assertEqual(idx["a"], {"note": "my doubt", "since": "2026-10-01"})
        self.assertIn("until", idx["b"])
        self.assertIn("1 will be marked", self.asked)
        self.assertIn("already marked", self.asked)

    def test_building_tag_and_seg_json(self):
        self.photo("wall__BLDG003.jpg", days_ago=1)
        self.run_button()
        self.assertIn("wall", self.index())
        self.assertTrue(self.app.is_photo_to_be_reviewed(os.path.join(self.tmp, "Images", "wall__BLDG003.jpg")))

    def test_sets_are_marked_and_counted(self):
        self.photo("img.jpg")
        t = self.photo("tr.jpg", set_name="Training set")           # JSON moved with the photo
        self.photo("va.jpg", set_name="Validation set")             # JSON stays in the data folder
        self.photo("lost.jpg", with_photo=False)
        self.run_button()
        self.assertEqual(set(self.index()), {"img", "tr", "va", "lost"})
        self.assertTrue(self.flags(t)["to_be_reviewed"])
        self.assertIn("1 in 'Images'", self.asked)
        self.assertIn("1 in 'Training set'", self.asked)
        self.assertIn("1 in 'Validation set'", self.asked)
        self.assertIn("Reset pipeline training", self.asked)
        self.assertIn("1 JSON(s) without their photo", self.asked)

    def test_cancel_changes_nothing(self):
        path = self.photo("a.jpg")
        self.assertFalse(self.run_button(confirm=False) is True and self.index())
        self.assertEqual(self.index(), {})
        self.assertEqual(self.flags(path), {})

    def test_nothing_to_mark(self):
        self.photo("new.jpg")
        self.run_button(date=self.today - datetime.timedelta(days=3))
        self.assertEqual(self.asked, "")
        self.assertIn("no photo saved up to", self.told)

    def test_future_date_is_today(self):
        self.photo("a.jpg")
        self.run_button(date=self.today + datetime.timedelta(days=30))
        self.assertEqual(self.index()["a"]["until"], self.today.isoformat())


class TestRemove(_Base):

    def test_removes_only_marks_by_date(self):
        a = self.photo("a.jpg", days_ago=2)
        b = self.photo("b.jpg", days_ago=2, flags={"to_be_reviewed": True, "review_note": "manual"})
        with open(os.path.join(self.tmp, rf.REVIEW_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump({"b": {"note": "manual", "since": "2026-10-01"}}, f)
        self.run_button()
        self.assertEqual(set(self.index()), {"a", "b"})
        self.run_button(remove=True)
        self.assertEqual(set(self.index()), {"b"})
        self.assertEqual(self.flags(a), {})
        self.assertEqual(self.flags(b)["review_note"], "manual")
        self.assertIn("1 of the 2", self.asked)

    def test_removed_marks_are_not_reimported(self):
        self.photo("a.jpg", days_ago=1)
        self.run_button()
        self.run_button(remove=True)
        self.app._import_review_flags_from_jsons()
        self.assertEqual(self.index(), {})


class TestMode2AfterMarking(_Base):

    def test_mode2_loads_only_them(self):
        self.photo("a.jpg", days_ago=4)
        self.photo("b.jpg", days_ago=0)
        self.run_button(date=self.today - datetime.timedelta(days=1))
        cfg = self.app.cfg
        cfg.modalita_scelta = "2"
        cfg.image_queue = [os.path.join(self.tmp, "Images", n) for n in ("a.jpg", "b.jpg")]
        with mock.patch.object(self.app, "_prompt_review_filter", return_value=rf.REVIEW_FILTER_ONLY):
            self.app._apply_review_filter(ask=True)
        self.assertEqual([os.path.basename(p) for p in cfg.image_queue], ["a.jpg"])


class TestWiring(_Base):

    def test_idle_screen(self):
        self.photo("a.jpg")
        self.app.cfg.queue_idle = True
        with mock.patch.object(self.app, "mark_review_by_date") as run:
            self.app._handle_idle_key(REVIEW_BY_DATE_CODE)
        run.assert_called_once()

    def test_sidebar_and_menu(self):
        codes = [b[3] for _, buttons in gm._LEFT_SIDEBAR_SECTIONS for b in buttons] \
            if hasattr(gm, "_LEFT_SIDEBAR_SECTIONS") else None
        with open(gm.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"To be reviewed by date", "", REVIEW_BY_DATE_CODE', src)
        self.assertTrue(any(code == REVIEW_BY_DATE_CODE for _, code in gm._TOOL_COMMANDS))
        if codes is not None:
            self.assertIn(REVIEW_BY_DATE_CODE, codes)

    def test_dialog(self):
        root = fake_tkinter.Tk() if hasattr(fake_tkinter, "Tk") else gm.tk.Tk()
        dialog = gm._ReviewByDateDialog(root)
        self.assertEqual(dialog.date_var.get(), datetime.date.today().isoformat())
        dialog.date_var.set("ieri")
        dialog._ok()
        self.assertIsNone(dialog.result)
        dialog.date_var.set("01/11/2026")
        dialog.remove_var.set(True)
        dialog._ok()
        self.assertEqual(dialog.result["date"], datetime.date(2026, 11, 1))
        self.assertTrue(dialog.result["remove"])


if __name__ == "__main__":
    unittest.main()
