"""test_v150_xml_reference.py

Headless tests for the 1.5.0 [XML reference] (xml_reference.py, code 133): damage regions of a VOC-style XML
('XML files/<photo>.xml', boxes and polygons) drawn over the photo as a tracing guide, cracks in yellow and the
other classes in cyan; on screen only, never in the JSON or the masks.

Run with:  python3 -m unittest test_v150_xml_reference -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported

import crack_segmentation_gui as gm
from crack_extremes import EXTREMES_KEEP_KEYS
from crack_relabel import RELABEL_KEEP_KEYS
from test_v105_link import _two_crack_app, H, W
from xml_reference import (XML_REFERENCE_CODE, XML_CRACK_BGR, XML_OTHER_BGR, find_xml_dir, parse_reference_xml,
                           scale_regions, is_crack_region)

XML_TEMPLATE = """<?xml version="1.0"?>
<annotation>
  <filename>damage/images/wall.jpg</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object element="Masonry wall" type="rect">
    <name damage="Shear crack">Crack</name>
    <bndbox><xmin>40</xmin><ymin>60</ymin><xmax>140</xmax><ymax>140</ymax></bndbox>
  </object>
  <object element="Plaster" type="polygon">
    <name damage="Masonry spalling">Spalling</name>
    <bndbox><xmin>180</xmin><ymin>20</ymin><xmax>260</xmax><ymax>90</ymax></bndbox>
    <polygon><x1>180</x1><y1>20</y1><x2>260</x2><y2>30</y2><x3>220</x3><y3>90</y3></polygon>
  </object>
</annotation>
"""


def _write_xml(folder, name="wall.xml", w=W, h=H, dir_name="XML files"):
    xml_dir = os.path.join(folder, dir_name)
    os.makedirs(xml_dir, exist_ok=True)
    path = os.path.join(xml_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(XML_TEMPLATE.format(w=w, h=h))
    return path


def _press(app, code):
    with mock.patch.object(app, "refresh_zoom_viewport"):
        app.process_keypress(code)


def _canvas():
    return np.zeros((900, 1200, 3), dtype=np.uint8)


def _has_color(win, bgr):
    return bool(np.all(win == np.array(bgr, dtype=np.uint8), axis=2).any())


class TestParsing(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_box_and_polygon(self):
        size, regions = parse_reference_xml(_write_xml(self.tmp))
        self.assertEqual(size, (W, H))
        self.assertEqual([r["name"] for r in regions], ["Crack", "Spalling"])
        self.assertEqual(regions[0]["points"], [(40, 60), (140, 60), (140, 140), (40, 140)])
        self.assertEqual(regions[1]["points"], [(180, 20), (260, 30), (220, 90)])
        self.assertEqual(regions[0]["damage"], "Shear crack")

    def test_pt_style_polygon(self):
        path = os.path.join(self.tmp, "pt.xml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<annotation><object><name>crack</name><polygon><pt><x>1</x><y>2</y></pt><pt><x>5</x><y>2</y>"
                     "</pt><pt><x>3</x><y>7</y></pt></polygon></object></annotation>")
        size, regions = parse_reference_xml(path)
        self.assertIsNone(size)
        self.assertEqual(regions[0]["points"], [(1, 2), (5, 2), (3, 7)])
        self.assertTrue(is_crack_region(regions[0]))

    def test_rescaled_to_a_resized_photo(self):
        regions = [{"name": "Crack", "damage": "", "points": [(100, 50), (200, 50), (200, 100)]}]
        out = scale_regions(regions, (400, 200), 200, 100)
        self.assertEqual(out[0]["points"], [(50, 25), (100, 25), (100, 50)])

    def test_other_orientation_not_shown(self):
        regions = [{"name": "Crack", "damage": "", "points": [(1, 1), (2, 1), (2, 2)]}]
        self.assertIsNone(scale_regions(regions, (400, 200), 100, 200))

    def test_folder_name_with_trailing_space(self):
        _write_xml(self.tmp, dir_name="XML files ")
        self.assertEqual(find_xml_dir(self.tmp), os.path.join(self.tmp, "XML files "))
        self.assertIsNone(find_xml_dir(os.path.join(self.tmp, "missing")))


class TestOnScreen(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.app = _two_crack_app(self.tmp)

    def test_regions_drawn_in_their_class_colors(self):
        _write_xml(self.tmp)
        win = _canvas()
        self.app._draw_xml_reference(win)
        self.assertTrue(_has_color(win, XML_CRACK_BGR))
        self.assertTrue(_has_color(win, XML_OTHER_BGR))

    def test_no_xml_draws_nothing(self):
        win = _canvas()
        self.app._draw_xml_reference(win)
        self.assertFalse(win.any())

    def test_button_hides_and_shows(self):
        _write_xml(self.tmp)
        _press(self.app, XML_REFERENCE_CODE)
        win = _canvas()
        self.app._draw_xml_reference(win)
        self.assertFalse(win.any())
        _press(self.app, XML_REFERENCE_CODE)
        self.app._draw_xml_reference(win)
        self.assertTrue(_has_color(win, XML_CRACK_BGR))

    def test_parsed_once_per_photo(self):
        _write_xml(self.tmp)
        with mock.patch("xml_reference.parse_reference_xml", wraps=parse_reference_xml) as parse:
            self.app._draw_xml_reference(_canvas())
            self.app._draw_xml_reference(_canvas())
        self.assertEqual(parse.call_count, 1)

    def test_broken_xml_is_skipped(self):
        path = _write_xml(self.tmp)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<annotation><object>")
        win = _canvas()
        self.app._draw_xml_reference(win)
        self.assertFalse(win.any())

    def test_never_written_to_json(self):
        _write_xml(self.tmp)
        self.app._draw_xml_reference(_canvas())
        self.app._save_annotations_now()
        with open(self.app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("Spalling", text)
        self.assertNotIn("Shear crack", text)


class TestGuiEntries(unittest.TestCase):

    def test_in_view_menu_and_sidebar(self):
        view = dict(gm._ALL_MENU_GROUPS)["\U0001F441 View"]
        self.assertIn(XML_REFERENCE_CODE, [code for _, code in view])
        self.assertIn(XML_REFERENCE_CODE, [code for *_, code in gm._SIDEBAR_SHORTCUTS])
        self.assertIn(XML_REFERENCE_CODE, gm._SIDEBAR_TOGGLE_STATES)

    def test_view_toggle_keeps_the_edit_modes_on(self):
        self.assertIn(XML_REFERENCE_CODE, RELABEL_KEEP_KEYS)
        self.assertIn(XML_REFERENCE_CODE, EXTREMES_KEEP_KEYS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
