"""test_v240_issue_log.py

Headless tests for the 2.4.0 issue log (issue_log.py, codes 164 / 165):
  - segmentation_issues.csv with the columns N, image, category, problem, mode, destination;
  - the destination set proposed from the photo's rows or from the set holding its building;
  - one row per photo and category, n kept when the photo is saved again, rows replaced by the new choice;
  - nothing is asked while the log is off; Skip leaves the list as it is;
  - the button is in the left sidebar (shown pressed while on), the table in the "Issues" menu.

Run with:  python3 -m unittest test_v240_issue_log -v
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import issue_log as il
import crack_segmentation_gui as gm
from test_gui_shell_parity import make_app


class TestCsv(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="crackseg_il_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = il.issue_log_path(self.dir)

    def test_columns_and_numbers(self):
        il.set_photo_issues(self.path, "a__BLDG001.jpg", "2", [("A", "fughe"), ("C", "macerie")], "Test set")
        il.set_photo_issues(self.path, "b__BLDG002.jpg", "1", [("B", "cornice")])
        self.assertEqual([r["destination"] for r in il.read_issues(self.path)], ["Test set", "Test set", "none"])
        with open(self.path, encoding="utf-8-sig") as fh:
            self.assertEqual(fh.readline().strip(), "N,image,category,problem,mode,destination")
        rows = il.read_issues(self.path)
        self.assertEqual([(r["N"], r["image"], r["category"], r["mode"]) for r in rows],
                         [("1", "a__BLDG001.jpg", "A", "2"), ("2", "a__BLDG001.jpg", "C", "2"),
                          ("3", "b__BLDG002.jpg", "B", "1")])

    def test_saving_again_replaces_the_photo_rows_and_keeps_n(self):
        il.set_photo_issues(self.path, "a.jpg", "2", [("A", "fughe"), ("C", "macerie")])
        il.set_photo_issues(self.path, "b.jpg", "2", [("B", "cornice")])
        il.set_photo_issues(self.path, "a.jpg", "1", [("C", "macerie -- in basso"), ("D", "cavillatura")])
        rows = {(r["image"], r["category"]): r for r in il.read_issues(self.path)}
        self.assertNotIn(("a.jpg", "A"), rows)
        self.assertEqual(rows[("a.jpg", "C")]["N"], "2")
        self.assertEqual(rows[("a.jpg", "C")]["problem"], "macerie -- in basso")
        self.assertEqual(rows[("a.jpg", "D")]["N"], "4")
        self.assertEqual(rows[("a.jpg", "D")]["mode"], "1")
        il.set_photo_issues(self.path, "a.jpg", "2", [])
        self.assertEqual([r["image"] for r in il.read_issues(self.path)], ["b.jpg"])

    def test_old_italian_columns_are_read(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("n,immagine,categoria,problema,modalità,destinazione\n"
                     "7,a.jpg,A,fughe di muratura tracciate come fessure -- in alto,2,nessuna\n")
        rows = il.read_issues(self.path)
        self.assertEqual((rows[0]["N"], rows[0]["image"], rows[0]["destination"]), ("7", "a.jpg", "none"))
        self.assertEqual(rows[0]["problem"], "masonry joints traced as cracks -- in alto")
        il.set_photo_issues(self.path, "b.jpg", "1", [("B", "x")])
        with open(self.path, encoding="utf-8-sig") as fh:
            self.assertEqual(fh.readline().strip(), "N,image,category,problem,mode,destination")
        self.assertEqual([r["N"] for r in il.read_issues(self.path)], ["7", "8"])

    def test_problem_text_and_delete(self):
        self.assertEqual(il.problem_text("A"), "masonry joints traced as cracks")
        self.assertEqual(il.problem_text("B", "zona in alto"),
                         "edges, corners, cornices, roads or streaks traced -- zona in alto")
        self.assertEqual(il.problem_text("X", "", "ombra tracciata"), "ombra tracciata")
        il.set_photo_issues(self.path, "a.jpg", "2", [("A", "x"), ("B", "y")])
        self.assertEqual(il.delete_issue_rows(self.path, ["1"]), 1)
        self.assertEqual([r["N"] for r in il.read_issues(self.path)], ["2"])


class TestApp(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="crackseg_il_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.app = make_app()
        self.app.cfg.SCRIPT_DIR = self.dir
        self.app.cfg.CURRENT_IMAGE_PATH = os.path.join(self.dir, "Images", "a__BLDG005.jpg")
        self.app.cfg.modalita_scelta = "2"

    def test_destination_suggestion(self):
        names = self.app.cfg.DATASET_SPLIT_DIRS
        self.assertEqual(self.app.suggested_destination("a__BLDG005.jpg", []), (None, ""))
        os.makedirs(os.path.join(self.dir, "Test set", "Images"))
        open(os.path.join(self.dir, "Test set", "Images", "z__BLDG005.jpg"), "w").close()
        open(os.path.join(self.dir, "Test set", "Images", "y__BLDG0050.jpg"), "w").close()
        key, hint = self.app.suggested_destination("a__BLDG005.jpg", [])
        self.assertEqual(key, "test")
        self.assertIn("BLDG005 already has photos in Test set", hint)
        self.assertEqual(self.app.suggested_destination("a__BLDG005.jpg", [{"destination": names["training"]}]),
                         ("training", ""))
        self.assertEqual(self.app.suggested_destination("b__BLDG050.jpg", []), (None, ""))

    def test_off_asks_nothing(self):
        with mock.patch.object(self.app, "_ask_segmentation_issues") as ask:
            self.assertIsNone(self.app.log_segmentation_issues())
        ask.assert_not_called()
        self.assertFalse(os.path.exists(self.app.issue_log_file()))

    def test_toggle_code_and_log_after_save(self):
        with redirect_stdout(io.StringIO()):
            self.app.process_keypress(il.ISSUE_LOG_CODE)
        self.assertTrue(self.app.issue_log_on())
        with mock.patch.object(self.app, "_ask_segmentation_issues", return_value=([("A", "fughe")], "training")) \
                as ask, \
                redirect_stdout(io.StringIO()):
            self.app.log_segmentation_issues()
        ask.assert_called_once_with("a__BLDG005.jpg", [])
        rows = il.read_issues(self.app.issue_log_file())
        self.assertEqual((rows[0]["image"], rows[0]["category"], rows[0]["mode"], rows[0]["destination"]),
                         ("a__BLDG005.jpg", "A", "2", "Training set"))
        with mock.patch.object(self.app, "_ask_segmentation_issues", return_value=None), \
                redirect_stdout(io.StringIO()):
            self.app.log_segmentation_issues()
        self.assertEqual(len(il.read_issues(self.app.issue_log_file())), 1, "Skip leaves the list as it is")
        with redirect_stdout(io.StringIO()):
            self.app.process_keypress(il.ISSUE_LOG_CODE)
        self.assertFalse(self.app.issue_log_on())

    def test_save_calls_the_log(self):
        warn = mock.patch.object(self.app, "_acknowledge_save_warning")
        warn.start()
        self.addCleanup(warn.stop)
        with mock.patch.object(self.app, "_prepare_export_folders", side_effect=OSError("disk")), \
                mock.patch.object(self.app, "log_segmentation_issues") as log, redirect_stdout(io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            self.app.export_labelme_format()
        log.assert_not_called()   # a failed save logs nothing
        names = ["_prepare_export_folders", "_get_current_image_base64", "_build_labelme_data_dict",
                 "_run_multiview_check_safely", "_finalize_labelme_shapes", "_write_labelme_json",
                 "_remove_stale_photo_outputs", "_export_segmented_preview_image", "_export_binary_masks",
                 "_append_csv_report_row", "_play_save_beep", "_archive_saved_files", "mark_annotations_saved"]
        patches = [mock.patch.object(self.app, n, return_value={}) for n in names]
        patches.append(mock.patch.object(self.app, "_compute_export_summary_stats", return_value=(0, 0, 0, 0)))
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with mock.patch.object(self.app, "log_segmentation_issues") as log, redirect_stdout(io.StringIO()):
            self.app.export_labelme_format()
        log.assert_called_once()


class TestGui(unittest.TestCase):

    def test_button_menu_and_toggle_state(self):
        codes = [b[3] for _, buttons in gm._LEFT_SIDEBAR_SECTIONS for b in buttons]
        self.assertIn(il.ISSUE_LOG_CODE, codes)
        menu_codes = [c for _, commands in gm._ALL_MENU_GROUPS for _, c in commands]
        self.assertIn(il.ISSUE_LOG_VIEW_CODE, menu_codes)
        self.assertIn(il.ISSUE_LOG_CODE, menu_codes)
        toolbar = [c for _, c in gm.GuiCrackSegmentation._toolbar_shortcuts(None)]
        self.assertIn(il.ISSUE_LOG_VIEW_CODE, toolbar, "Issues button next to Manual")
        cfg = make_app().cfg
        self.assertFalse(gm._SIDEBAR_TOGGLE_STATES[il.ISSUE_LOG_CODE](cfg))
        cfg.issue_log_on = True
        self.assertTrue(gm._SIDEBAR_TOGGLE_STATES[il.ISSUE_LOG_CODE](cfg))

    def test_dialog_ticks_the_current_categories(self):
        rows = [{"N": "1", "image": "a.jpg", "category": "C", "problem": "macerie", "mode": "2",
                 "destination": "Test set"}]
        d = gm._IssueDialog(fake_tkinter.Tk(), "a.jpg", rows, None, "test", "")
        self.assertEqual(d.destination.get(), "test")
        self.assertTrue(d.vars["C"].get())
        self.assertFalse(d.vars["A"].get())
        d.vars["A"].set(True)
        d.note.insert(0, "in basso")
        d._ok()
        self.assertEqual(d.result, ([("A", "masonry joints traced as cracks -- in basso"),
                                     ("C", "rubble or collapse traced: area to ignore -- in basso")], "test"))
        d2 = gm._IssueDialog(fake_tkinter.Tk(), "a.jpg", [])
        self.assertEqual(d2.destination.get(), "none")
        self.assertIsNone(d2.destination_key())
        d2._skip()
        self.assertIsNone(d2.result)

    def test_panel_lists_and_deletes(self):
        tmp = tempfile.mkdtemp(prefix="crackseg_il_")
        self.addCleanup(shutil.rmtree, tmp, True)
        app = make_app()
        app.cfg.SCRIPT_DIR = tmp
        il.set_photo_issues(app.issue_log_file(), "a.jpg", "2", [("A", "x"), ("B", "y")])
        app._toggle_issue_log_view = lambda: None
        panel = gm.IssueLogPanel(fake_tkinter.Tk(), app)
        panel.refresh()
        items = panel.tree.get_children()
        self.assertEqual(len(items), 2)
        self.assertEqual(panel.tree.item(items[0])["values"], ("1", "a.jpg", "A", "x", "2", "none"))
        panel.tree.selection_set(items[0])
        with mock.patch.object(gm.messagebox, "askyesno", return_value=True):
            self.assertEqual(panel.delete_selected(), 1)
        self.assertEqual(len(panel.tree.get_children()), 1)


class TestPanelsBetweenSidebars(unittest.TestCase):
    """v2.4.0: the manual and the issue table take the place of the photo; the sidebars stay."""

    def setUp(self):
        import test_gui_panel_wiring as wiring
        self.app = wiring.make_app(self)

    def test_manual_and_issues_live_in_the_middle_row(self):
        self.assertIn(self.app._pdf_panel.frame, self.app._middle_row.children)
        self.assertIn(self.app._issue_panel.frame, self.app._middle_row.children)

    def test_switching_panels_brings_the_photo_back(self):
        with mock.patch.object(self.app._pdf_panel, "show"), mock.patch.object(self.app._issue_panel, "refresh"):
            self.app._toggle_help_menu()
            self.assertTrue(self.app._pdf_panel_visible)
            self.app._toggle_issue_log_view()
            self.assertFalse(self.app._pdf_panel_visible)
            self.assertTrue(self.app._issue_panel_visible)
            self.app._toggle_issue_log_view()
        self.assertFalse(self.app._issue_panel_visible)


class TestInfoPanel(unittest.TestCase):
    """v2.4.0: [J] / Info shows the photo data on a white panel between the sidebars, not a glass on the photo."""

    def setUp(self):
        import test_gui_panel_wiring as wiring
        self.app = wiring.make_app(self)

    def test_split_fields(self):
        self.assertEqual(gm.InfoPanel.split_fields("FILE: 3/10 (a.jpg) | Tool: CRACK", "BUILDING: BLDG005 | ELEMENTS: 4"),
                         [("File", "3/10 (a.jpg)"), ("Tool", "CRACK"), ("Building", "BLDG005"), ("Elements", "4")])

    def test_toggle_shows_the_panel_and_leaves_the_glass_flag(self):
        self.assertIn(self.app._info_panel.frame, self.app._middle_row.children)
        self.app._toggle_info_overlay()
        self.assertTrue(self.app._info_panel_visible)
        self.assertFalse(self.app.cfg.show_info_overlay, "the glass flag stays for the crack report")
        with mock.patch.object(self.app._pdf_panel, "show"):
            self.app._toggle_help_menu()
        self.assertFalse(self.app._info_panel_visible, "the manual replaces the info panel")
        self.app._toggle_help_menu()
        self.app._toggle_info_overlay()
        self.app._toggle_info_overlay()
        self.assertFalse(self.app._info_panel_visible)

    def test_data_of_the_photo(self):
        self.app.cfg.CURRENT_IMAGE_PATH = "/x/a__BLDG005.jpg"
        self.app.cfg.queue_idle = False
        self.app.cfg.saved_cracks = [{"path": [(0, 0), (5, 5)], "active": True}]
        self.app._info_texts = ("FILE: 1/2 (a__BLDG005.jpg) | Tool: CRACK | Zoom: 1.0x", "Cracks Length: 1.00 cm")
        with mock.patch.object(self.app, "_crack_quality", return_value=(87.4, 0.61, False)), \
                mock.patch.object(self.app, "_crack_multiview", return_value=None):
            fields, cracks = self.app.info_panel_data()
        names = dict(fields)
        self.assertEqual(names["File"], "1/2 (a__BLDG005.jpg)")
        self.assertEqual(names["Building"], "BLDG005")
        self.assertEqual(cracks, (("1", "87", "0.61", "", ""),))
        self.app.cfg.CURRENT_IMAGE_PATH = None
        self.assertEqual(self.app.info_panel_data()[1], ())


if __name__ == "__main__":
    unittest.main()
