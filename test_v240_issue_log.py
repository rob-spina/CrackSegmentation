"""test_v240_issue_log.py

Headless tests for the 2.4.0 issue log (issue_log.py, codes 164 / 165):
  - segmentation_issues.csv with the columns n, immagine, categoria, problema, modalità, destinazione;
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
        self.assertEqual([r["destinazione"] for r in il.read_issues(self.path)], ["Test set", "Test set", "nessuna"])
        with open(self.path, encoding="utf-8-sig") as fh:
            self.assertEqual(fh.readline().strip(), "n,immagine,categoria,problema,modalità,destinazione")
        rows = il.read_issues(self.path)
        self.assertEqual([(r["n"], r["immagine"], r["categoria"], r["modalità"]) for r in rows],
                         [("1", "a__BLDG001.jpg", "A", "2"), ("2", "a__BLDG001.jpg", "C", "2"),
                          ("3", "b__BLDG002.jpg", "B", "1")])

    def test_saving_again_replaces_the_photo_rows_and_keeps_n(self):
        il.set_photo_issues(self.path, "a.jpg", "2", [("A", "fughe"), ("C", "macerie")])
        il.set_photo_issues(self.path, "b.jpg", "2", [("B", "cornice")])
        il.set_photo_issues(self.path, "a.jpg", "1", [("C", "macerie -- in basso"), ("D", "cavillatura")])
        rows = {(r["immagine"], r["categoria"]): r for r in il.read_issues(self.path)}
        self.assertNotIn(("a.jpg", "A"), rows)
        self.assertEqual(rows[("a.jpg", "C")]["n"], "2")
        self.assertEqual(rows[("a.jpg", "C")]["problema"], "macerie -- in basso")
        self.assertEqual(rows[("a.jpg", "D")]["n"], "4")
        self.assertEqual(rows[("a.jpg", "D")]["modalità"], "1")
        il.set_photo_issues(self.path, "a.jpg", "2", [])
        self.assertEqual([r["immagine"] for r in il.read_issues(self.path)], ["b.jpg"])

    def test_problem_text_and_delete(self):
        self.assertEqual(il.problem_text("A"), "fughe di muratura tracciate come fessure")
        self.assertEqual(il.problem_text("B", "zona in alto"),
                         "bordi, spigoli, cornici, strade o striature tracciati -- zona in alto")
        self.assertEqual(il.problem_text("X", "", "ombra tracciata"), "ombra tracciata")
        il.set_photo_issues(self.path, "a.jpg", "2", [("A", "x"), ("B", "y")])
        self.assertEqual(il.delete_issue_rows(self.path, ["1"]), 1)
        self.assertEqual([r["n"] for r in il.read_issues(self.path)], ["2"])


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
        self.assertEqual(self.app.suggested_destination("a__BLDG005.jpg", [{"destinazione": names["training"]}]),
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
        self.assertEqual((rows[0]["immagine"], rows[0]["categoria"], rows[0]["modalità"], rows[0]["destinazione"]),
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
        rows = [{"n": "1", "immagine": "a.jpg", "categoria": "C", "problema": "macerie", "modalità": "2",
                 "destinazione": "Test set"}]
        d = gm._IssueDialog(fake_tkinter.Tk(), "a.jpg", rows, None, "test", "")
        self.assertEqual(d.destination.get(), "test")
        self.assertTrue(d.vars["C"].get())
        self.assertFalse(d.vars["A"].get())
        d.vars["A"].set(True)
        d.note.insert(0, "in basso")
        d._ok()
        self.assertEqual(d.result, ([("A", "fughe di muratura tracciate come fessure -- in basso"),
                                     ("C", "macerie o crollo tracciati: zona da ignorare -- in basso")], "test"))
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
        self.assertEqual(panel.tree.item(items[0])["values"], ("1", "a.jpg", "A", "x", "2", "nessuna"))
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


if __name__ == "__main__":
    unittest.main()
