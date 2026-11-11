"""test_v210_features_off.py -- 2.1.0 ships some features switched off (config FEATURE_* = False): their commands
have no sidebar button and no menu entry, and Auto ignore does nothing; CRACKSEG_ALL_FEATURES=1 turns them on.

Run with:  python3 -m unittest test_v210_features_off -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import crack_segmentation_gui as gm
from config import Config
from active_learning import ACTIVE_LEARNING_CODE
from auto_ignore import AUTO_IGNORE_CODE
from leonardo_link import LEONARDO_CODE

OFF = {"FEATURE_GROUP_SIMILAR_PHOTOS", "FEATURE_ACTIVE_LEARNING", "FEATURE_LEONARDO", "FEATURE_AUTO_IGNORE"}


def _cfg_without_env():
    with mock.patch.dict(os.environ, {"CRACKSEG_ALL_FEATURES": ""}):
        return Config()


class TestFeaturesOff(unittest.TestCase):

    def test_release_defaults_off(self):
        cfg = _cfg_without_env()
        for flag in OFF:
            self.assertFalse(getattr(cfg, flag), flag)

    def test_env_turns_them_on(self):
        with mock.patch.dict(os.environ, {"CRACKSEG_ALL_FEATURES": "1"}):
            cfg = Config()
        self.assertTrue(all(getattr(cfg, f) for f in OFF))

    def test_hidden_codes(self):
        hidden = gm.hidden_feature_codes(_cfg_without_env())
        self.assertEqual(hidden, {gm.GROUP_PHOTOS_CODE, ACTIVE_LEARNING_CODE, LEONARDO_CODE, AUTO_IGNORE_CODE})

    def test_sidebar_sections_filtered(self):
        hidden = gm.hidden_feature_codes(_cfg_without_env())
        codes = [b[3] for sections in (gm._RIGHT_SIDEBAR_SECTIONS, gm._LEFT_SIDEBAR_SECTIONS)
                 for _, buttons in gm._visible_sections(sections, hidden) for b in buttons]
        self.assertFalse(hidden & set(codes))
        everything = [b[3] for _, buttons in gm._RIGHT_SIDEBAR_SECTIONS + gm._LEFT_SIDEBAR_SECTIONS for b in buttons]
        self.assertEqual(len(codes), len(everything) - 4)

    def test_gui_without_the_features(self):
        from test_gui_panel_wiring import make_app as make_gui_app
        with mock.patch.dict(os.environ, {"CRACKSEG_ALL_FEATURES": ""}):
            app = make_gui_app(self)
        self.assertNotIn(AUTO_IGNORE_CODE, app._sidebar.buttons_by_code)
        self.assertNotIn(LEONARDO_CODE, app._sidebar.buttons_by_code)
        self.assertIn(ord('i'), app._sidebar.buttons_by_code)          # Ignore region stays

    def test_auto_ignore_code_does_nothing_when_off(self):
        from test_gui_shell_parity import make_app
        app = make_app()
        app.cfg.FEATURE_AUTO_IGNORE = False
        with mock.patch.object(app, "run_auto_ignore") as run:
            app.process_keypress(AUTO_IGNORE_CODE)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
