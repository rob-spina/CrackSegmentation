"""test_v140_manual.py

USER REPORT (1.4.0): in the installed app the [?] manual showed only a grey panel, and pressing [?] again did not
bring the photo back. A failure while rendering the pages (after the PDF opened) escaped PdfManualPanel.show(),
so _toggle_help_menu() never recorded the panel as visible and every press tried to show it again.
Now the error is written in the panel, the next [?] returns to the photo, and padded pixmap rows (stride) are read.

Run with:  python3 -m unittest test_v140_manual -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported
import fake_fitz

import crack_segmentation_gui as gm
from test_gui_panel_wiring import make_app


class _BrokenPage(fake_fitz.FakePage):
    def get_pixmap(self, dpi=110):
        raise RuntimeError("pixmap failed")


class _BrokenFitz(fake_fitz.FakeFitzModule):
    def open(self, path):
        doc = super().open(path)
        doc.pages = [_BrokenPage()]
        return doc


class _PaddedPixmap(fake_fitz.FakePixmap):
    """Rows padded to a stride wider than width * 3, as some PyMuPDF builds return."""

    def __init__(self, width=5, height=4, pad=3):
        super().__init__(width, height)
        self.stride = width * 3 + pad
        row = bytes([180, 180, 180]) * width + bytes(pad)
        self.samples = row * height


class _PaddedPage(fake_fitz.FakePage):
    def get_pixmap(self, dpi=110):
        return _PaddedPixmap()


class _PaddedFitz(fake_fitz.FakeFitzModule):
    def open(self, path):
        doc = super().open(path)
        doc.pages = [_PaddedPage()]
        return doc


def _texts(panel):
    return [item.get("text", "") for item in panel.canvas._items.values() if "text" in item]


class TestManualRenderFailure(unittest.TestCase):

    def test_pymupdf_failing_on_import_is_reported(self):
        panel = gm.PdfManualPanel(fake_tkinter.Tk(), "/fake/manual.pdf")
        with mock.patch.object(panel, "_import_pymupdf",
                               side_effect=FileNotFoundError("pymupdf/layout/resources/onnx/layout.yaml")):
            panel.show()   # must not raise
        texts = _texts(panel)
        self.assertTrue(any("PyMuPDF" in t and "layout.yaml" in t for t in texts), texts)

    def test_spec_bundles_the_pymupdf_data_files(self):
        spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packaging", "crack_segmentation_gui.spec")
        with open(spec, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn('collect_data_files("pymupdf")', text)
        self.assertIn("+ PYMUPDF_DATAS", text)


    def _patch(self, module):
        patcher = mock.patch.dict(sys.modules, {"fitz": module, "pymupdf": module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_render_error_is_shown_in_the_panel(self):
        self._patch(_BrokenFitz(n_pages=1))
        panel = gm.PdfManualPanel(fake_tkinter.Tk(), "/fake/manual.pdf")
        panel.show()   # must not raise
        self.assertTrue(any("pixmap failed" in t for t in _texts(panel)))

    def test_failed_render_is_not_retried_on_every_press(self):
        module = _BrokenFitz(n_pages=1)
        self._patch(module)
        panel = gm.PdfManualPanel(fake_tkinter.Tk(), "/fake/manual.pdf")
        panel.show()
        panel.hide()
        panel.show()
        self.assertEqual(len(module.opened_paths), 1)

    def test_second_question_mark_returns_to_the_photo(self):
        self._patch(_BrokenFitz(n_pages=1))
        app = make_app(self)
        app._toggle_help_menu()
        self.assertTrue(app._pdf_panel_visible)
        app._toggle_help_menu()
        self.assertFalse(app._pdf_panel_visible)
        self.assertIn(app._middle_row, app._middle_row.master.children)

    def test_padded_rows_are_read_with_the_stride(self):
        self._patch(_PaddedFitz(n_pages=1))
        panel = gm.PdfManualPanel(fake_tkinter.Tk(), "/fake/manual.pdf")
        panel.show()
        self.assertEqual(len(panel._photo_refs), 1)
        self.assertFalse(_texts(panel), "no error message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
