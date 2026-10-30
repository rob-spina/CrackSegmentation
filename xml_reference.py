"""xml_reference.py
v1.5.0 [XML reference] (left sidebar View section and View menu, code 133), mixed into CrackSegmentation
(smart_segmentation.py). Shows the damage regions of a Pascal VOC-style XML (<photo name>.xml in the 'XML files'
folder next to 'Images') as outlines over the photo: rectangles (<bndbox>) and polygons (<polygon><x1><y1>...).
Crack regions are yellow, the other damage classes cyan, each with its class label. A guide for tracing only:
on screen, never saved to the JSON, never in the masks or in the exported images.
"""
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

XML_REFERENCE_CODE = 133
XML_DIR_NAME = "XML files"
XML_CRACK_BGR = (0, 255, 255)
XML_OTHER_BGR = (255, 255, 0)
XML_LABEL_BG_BGR = (0, 0, 0)


def find_xml_dir(base_dir):
    """'XML files' inside base_dir, also when the folder name has stray spaces ('XML files '). None if absent."""
    exact = os.path.join(str(base_dir), XML_DIR_NAME)
    if os.path.isdir(exact):
        return exact
    try:
        names = os.listdir(str(base_dir))
    except OSError:
        return None
    for name in sorted(names):
        if name.strip().lower() == XML_DIR_NAME.lower() and os.path.isdir(os.path.join(str(base_dir), name)):
            return os.path.join(str(base_dir), name)
    return None


def _polygon_points(poly_el):
    """Vertices of <polygon>: <x1><y1><x2><y2>... or <pt><x><y></pt> children."""
    pts = [(float(pt.findtext('x')), float(pt.findtext('y'))) for pt in poly_el.findall('pt')]
    if pts:
        return pts
    vals = {child.tag: float(child.text) for child in poly_el if child.text and child.text.strip()}
    i, out = 1, []
    while f"x{i}" in vals and f"y{i}" in vals:
        out.append((vals[f"x{i}"], vals[f"y{i}"]))
        i += 1
    return out


def _box_points(obj_el):
    bb = obj_el.find('bndbox')
    if bb is None:
        return []
    x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _parse_object(obj_el):
    """{'name', 'damage', 'points'} of one <object>; the polygon wins over the box when both are present."""
    name_el = obj_el.find('name')
    name = (name_el.text or "").strip() if name_el is not None else ""
    damage = (name_el.get('damage') or "").strip() if name_el is not None else ""
    poly_el = obj_el.find('polygon')
    points = _polygon_points(poly_el) if poly_el is not None else []
    if len(points) < 3:
        points = _box_points(obj_el)
    return {"name": name, "damage": damage, "points": points}


def parse_reference_xml(path):
    """(size, regions) of a VOC-style XML. size = (width, height) declared in <size> or None."""
    root = ET.parse(path).getroot()
    size_el = root.find('size')
    size = None
    if size_el is not None and size_el.findtext('width') and size_el.findtext('height'):
        size = (int(float(size_el.findtext('width'))), int(float(size_el.findtext('height'))))
    regions = [_parse_object(o) for o in root.findall('object')]
    return size, [r for r in regions if len(r["points"]) >= 3]


def scale_regions(regions, size, img_w, img_h):
    """Regions in image pixels: rescaled when the XML was made on a resized copy. None when the orientation differs."""
    if size is None or size == (img_w, img_h):
        return regions
    xml_w, xml_h = size
    if (xml_w > xml_h) != (img_w > img_h) and xml_w != xml_h and img_w != img_h:
        return None
    sx, sy = img_w / xml_w, img_h / xml_h
    return [dict(r, points=[(x * sx, y * sy) for x, y in r["points"]]) for r in regions]


def is_crack_region(region):
    return "crack" in region["name"].lower()


class XmlReferenceMixin:
    """On-screen XML damage regions. Relies on CrackSegmentation's cfg and transform_real_to_window_coords."""

    def _xml_reference_path(self):
        xml_dir = find_xml_dir(self.cfg.SCRIPT_DIR)
        if xml_dir is None or not self.cfg.CURRENT_IMAGE_PATH:
            return None
        stem = os.path.splitext(os.path.basename(self.cfg.CURRENT_IMAGE_PATH))[0]
        path = os.path.join(xml_dir, stem + ".xml")
        return path if os.path.isfile(path) else None

    def _load_xml_reference(self):
        """Regions of the current photo (image pixels), parsed once per photo; [] when there is no usable XML."""
        state = self.cfg.xml_reference_state
        if state["image"] == self.cfg.CURRENT_IMAGE_PATH:
            return state["regions"]
        state["image"], state["regions"] = self.cfg.CURRENT_IMAGE_PATH, []
        path = self._xml_reference_path()
        if path is None:
            return []
        state["regions"] = self._read_xml_regions(path)
        return state["regions"]

    def _read_xml_regions(self, path):
        """Parses and rescales one XML, printing what was found (or why it is not shown)."""
        try:
            size, regions = parse_reference_xml(path)
        except (ET.ParseError, ValueError, TypeError) as exc:
            print(f" [XML] {os.path.basename(path)} not readable ({exc}): no reference regions shown.")
            return []
        scaled = scale_regions(regions, size, self.cfg.W_img, self.cfg.H_img)
        if scaled is None:
            print(f" [XML] {os.path.basename(path)}: size {size[0]}x{size[1]} has a different orientation from the "
                  f"photo ({self.cfg.W_img}x{self.cfg.H_img}), regions not shown.")
            return []
        n_crack = sum(1 for r in scaled if is_crack_region(r))
        print(f" [XML] {os.path.basename(path)}: {len(scaled)} reference region(s), {n_crack} crack.")
        return scaled

    def _toggle_xml_reference(self):
        """Code 133: hides / shows the XML reference regions."""
        self.cfg.show_xml_reference = not self.cfg.show_xml_reference
        state = "visible" if self.cfg.show_xml_reference else "hidden"
        print(f" [XML] Reference regions are now {state}.")
        self.refresh_zoom_viewport()

    def _xml_window_pts(self, region):
        return np.array([self.transform_real_to_window_coords(int(round(x)), int(round(y)))
                         for x, y in region["points"]], dtype=np.int32)

    def _draw_xml_reference(self, win_out):
        """Outlines and labels of the XML regions (cracks last, so they stay on top)."""
        if not self.cfg.show_xml_reference:
            return
        regions = self._load_xml_reference()
        for region in sorted(regions, key=is_crack_region):
            crack = is_crack_region(region)
            pts = self._xml_window_pts(region)
            color = XML_CRACK_BGR if crack else XML_OTHER_BGR
            cv2.polylines(win_out, [pts], True, color, 2 if crack else 1, cv2.LINE_AA)
            self._draw_xml_label(win_out, region, pts, color)

    @staticmethod
    def _draw_xml_label(win_out, region, pts, color):
        """Class (and damage type) just above the region's top-left corner, on a dark band for legibility."""
        text = region["name"] + (f": {region['damage']}" if region["damage"] else "")
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + base + 2, y - 4)
        cv2.rectangle(win_out, (x, y - th - base - 2), (x + tw + 4, y + 2), XML_LABEL_BG_BGR, -1)
        cv2.putText(win_out, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
