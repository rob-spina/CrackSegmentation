"""Basic coverage for compare_and_filter_cracks.py's CrackComparator --
this module had NO existing tests before this session's refactor split
build_variable_width_crack_mask() into smaller helpers. Not exhaustive,
but enough to catch a mechanical-extraction mistake in the pieces that
were actually touched."""
import unittest
import os
import json
import tempfile
import numpy as np

from compare_and_filter_cracks import CrackComparator, LoadedJson


class TestBuildVariableWidthCrackMask(unittest.TestCase):

    def setUp(self):
        self.comparator = CrackComparator(cfg=None)
        self.comparator.auto_edge_enabled = False  # isolate the plain scalar-dilation path

    def _straight_crack_shape(self, width_px=None):
        points = [[10, 50], [20, 50], [30, 50], [40, 50]]
        shape = {"label": "crack", "shape_type": "linestrip", "points": points}
        if width_px is not None:
            shape["width_segments"] = [{"i0": 0, "i1": 3, "width_px": width_px}]
        return shape

    def test_empty_shape_list_returns_all_zero_mask(self):
        out = self.comparator.build_variable_width_crack_mask([], 100, 100, default_dilation_px=2)
        self.assertEqual(out.shape, (100, 100))
        self.assertFalse(np.any(out))

    def test_plain_crack_produces_a_nonempty_mask_along_its_path(self):
        shape = self._straight_crack_shape()
        out = self.comparator.build_variable_width_crack_mask([shape], 100, 100, default_dilation_px=2)
        # Every centerline point (and its dilation) should be covered.
        self.assertTrue(out[50, 10] or np.any(out[45:55, 10]))
        self.assertTrue(out[50, 40] or np.any(out[45:55, 40]))

    def test_wider_width_segment_covers_more_pixels_than_default(self):
        narrow = self.comparator.build_variable_width_crack_mask(
            [self._straight_crack_shape()], 100, 100, default_dilation_px=1)
        wide = self.comparator.build_variable_width_crack_mask(
            [self._straight_crack_shape(width_px=8)], 100, 100, default_dilation_px=1)
        self.assertGreater(int(np.count_nonzero(wide)), int(np.count_nonzero(narrow)))

    def test_shape_with_no_points_is_skipped_without_error(self):
        out = self.comparator.build_variable_width_crack_mask(
            [{"label": "crack", "points": []}], 50, 50, default_dilation_px=2)
        self.assertFalse(np.any(out))


class TestIsCrackShape(unittest.TestCase):
    def test_recognizes_crack_and_crepa_labels(self):
        self.assertTrue(CrackComparator.is_crack_shape({"label": "crack"}))
        self.assertTrue(CrackComparator.is_crack_shape({"label": "crepa"}))

    def test_recognizes_linestrip_shape_type(self):
        self.assertTrue(CrackComparator.is_crack_shape({"label": "", "shape_type": "linestrip"}))

    def test_detachment_is_not_a_crack_shape(self):
        self.assertFalse(CrackComparator.is_crack_shape({"label": "detachment", "shape_type": "polygon"}))


class TestApplyDirectionalMargin(unittest.TestCase):
    """apply_directional_margin() had no direct test coverage either --
    mirrors the equivalent test added for smart_segmentation.py's twin
    function, adapted to this module's simpler (no binary_mask contour
    cap) growth logic."""

    def _straight_horizontal_setup(self):
        comparator = CrackComparator(cfg=None)
        points = [(10, 50), (11, 50), (12, 50), (13, 50), (14, 50)]
        mask = np.zeros((100, 100), dtype=np.uint8)
        return comparator, mask, points

    def test_growing_one_side_leaves_the_other_side_untouched(self):
        comparator, mask, points = self._straight_horizontal_setup()
        comparator.apply_directional_margin(mask, points, 0, 4, side_sign=1, delta=3, height=100, width=100)
        self.assertTrue(any(mask[51:54, 12]), "the grown side should have some fill")
        self.assertFalse(any(mask[45:50, 12]), "the untouched side must stay empty")

    def test_shrink_retracts_from_the_current_outer_edge(self):
        comparator, mask, points = self._straight_horizontal_setup()
        comparator.apply_directional_margin(mask, points, 0, 4, side_sign=1, delta=3, height=100, width=100)
        filled_before = int(np.count_nonzero(mask[:, 12]))
        comparator.apply_directional_margin(mask, points, 0, 4, side_sign=1, delta=-1, height=100, width=100)
        filled_after = int(np.count_nonzero(mask[:, 12]))
        self.assertLess(filled_after, filled_before)

    def test_zero_delta_or_empty_points_is_a_no_op(self):
        comparator, mask, points = self._straight_horizontal_setup()
        result = comparator.apply_directional_margin(mask, points, 0, 4, side_sign=1, delta=0, height=100, width=100)
        self.assertFalse(np.any(result))
        result2 = comparator.apply_directional_margin(mask, [], 0, -1, side_sign=1, delta=5, height=100, width=100)
        self.assertFalse(np.any(result2))


class TestRenderWidthSegmentMask(unittest.TestCase):
    """render_width_segment_mask() likewise had no direct coverage."""

    def test_legacy_segment_with_no_margins_dilates_the_centerline(self):
        comparator = CrackComparator(cfg=None)
        points = [(10, 50), (20, 50), (30, 50)]
        seg = {"i0": 0, "i1": 2, "width_px": 3}
        mask = comparator.render_width_segment_mask(points, 0, 2, seg, height=100, width=100)
        self.assertTrue(mask[50, 20])
        self.assertTrue(np.any(mask[47:54, 20]), "expected ~3px dilation around the centerline")

    def test_empty_points_produce_an_empty_mask_without_error(self):
        comparator = CrackComparator(cfg=None)
        seg = {"i0": 0, "i1": 0, "width_px": 2}
        mask = comparator.render_width_segment_mask([], 0, -1, seg, height=50, width=50)
        self.assertFalse(np.any(mask))


class TestCompare(unittest.TestCase):
    """compare() had no direct test coverage either."""

    def _write_json(self, tmpdir, name, cracks_points, scale=0.05):
        shapes = [{"label": "crack", "shape_type": "linestrip", "points": pts} for pts in cracks_points]
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"shapes": shapes, "pixel_to_cm_scale": scale, "imageHeight": 100, "imageWidth": 100}, f)
        return path

    def test_identical_cracks_across_files_are_all_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pts = [[0, 0], [10, 0], [20, 0]]
            p1 = self._write_json(tmpdir, "a.json", [pts])
            p2 = self._write_json(tmpdir, "b.json", [pts])
            loaded = [LoadedJson(p1), LoadedJson(p2)]
            rows, incompatible, extra_note = CrackComparator.compare(loaded, sinuosity_threshold=0.30, length_threshold=None)
            self.assertEqual(len(rows), 1)
            self.assertEqual(incompatible, set())
            self.assertIsNone(extra_note)

    def test_wildly_different_shape_is_flagged_incompatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            straight = [[0, 0], [20, 0]]
            zigzag = [[0, 0], [2, 10], [4, 0], [6, 10], [8, 0], [20, 0]]  # much more sinuous
            p1 = self._write_json(tmpdir, "a.json", [straight])
            p2 = self._write_json(tmpdir, "b.json", [zigzag])
            loaded = [LoadedJson(p1), LoadedJson(p2)]
            rows, incompatible, extra_note = CrackComparator.compare(loaded, sinuosity_threshold=0.30, length_threshold=None)
            self.assertEqual(incompatible, {0})
            self.assertEqual(rows[0]["verdict"], "DISCARDED")

    def test_mismatched_crack_counts_produce_an_explanatory_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pts = [[0, 0], [10, 0]]
            p1 = self._write_json(tmpdir, "a.json", [pts])
            p2 = self._write_json(tmpdir, "b.json", [pts, pts])
            loaded = [LoadedJson(p1), LoadedJson(p2)]
            rows, incompatible, extra_note = CrackComparator.compare(loaded, sinuosity_threshold=0.30, length_threshold=None)
            self.assertEqual(len(rows), 1)  # only the common prefix is compared
            self.assertIsNotNone(extra_note)


if __name__ == "__main__":
    unittest.main()
