"""test_v200_sam_download.py

Headless tests for the 2.0.0 automatic download of the SAM model (sam_weights.py, sam_autoseg.py):
  - sam_weights.download(): whole file, resume of an interrupted download (HTTP Range), a server that ignores
    Range, size / SHA-256 checks (a bad file is removed), cancel keeps the partial file, HTTP errors;
  - the button: asks once, downloads both files into '<data folder>/SAM model', stores settings pointing at them
    and at the bundled cracksam/ code; "No" goes to the manual settings; already-downloaded files are reused;
    the installed app (no Python of its own) asks for one;
  - bundled code and model card present; GUI dialogs.
A local HTTP server stands in for Meta's and GitHub's servers.

Run with:  python3 -m unittest test_v200_sam_download -v
"""
import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported

import crack_segmentation_gui as gm
import sam_weights
from test_gui_shell_parity import make_app

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with byte-range support (like Meta's and GitHub's servers); ?norange ignores Range."""
    served = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        path, _, query = self.path.partition("?")
        full = os.path.join(self.server.root, path.lstrip("/"))
        if not os.path.isfile(full):
            self.send_error(404)
            return
        data = _read(full)
        start = 0
        rng = self.headers.get("Range")
        if rng and query != "norange":
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
        else:
            self.send_response(200)
        _RangeHandler.served.append((path, start))
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


class _Server(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        cls.httpd.root = cls.root
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _RangeHandler.served.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _publish(self, name, size):
        data = os.urandom(size)
        with open(os.path.join(self.root, name), "wb") as f:
            f.write(data)
        return data, {"file": name, "url": f"{self.base}/{name}", "size": size,
                      "sha256": hashlib.sha256(data).hexdigest()}


class TestDownload(_Server):

    def test_whole_file_verified(self):
        data, spec = self._publish("a.bin", 3 * sam_weights.CHUNK + 7)
        seen = []
        dest = sam_weights.download(spec, os.path.join(self.tmp, "w", "a.bin"), lambda d, t: seen.append((d, t)))
        self.assertEqual(_read(dest), data)
        self.assertFalse(os.path.exists(dest + ".part"))
        self.assertEqual(seen[-1], (len(data), len(data)))

    def test_resume_after_interruption(self):
        data, spec = self._publish("b.bin", 2 * sam_weights.CHUNK + 100)
        dest = os.path.join(self.tmp, "b.bin")
        with open(dest + ".part", "wb") as f:
            f.write(data[:sam_weights.CHUNK + 5])
        sam_weights.download(spec, dest)
        self.assertEqual(_read(dest), data)
        self.assertEqual(_RangeHandler.served, [("/b.bin", sam_weights.CHUNK + 5)])

    def test_server_ignoring_range_restarts(self):
        data, spec = self._publish("c.bin", 5000)
        spec["url"] += "?norange"
        dest = os.path.join(self.tmp, "c.bin")
        with open(dest + ".part", "wb") as f:
            f.write(b"x" * 1000)
        sam_weights.download(spec, dest)
        self.assertEqual(_read(dest), data)

    def test_complete_part_file(self):
        data, spec = self._publish("d.bin", 4000)
        dest = os.path.join(self.tmp, "d.bin")
        with open(dest + ".part", "wb") as f:
            f.write(data)
        sam_weights.download(spec, dest)   # the server answers 416: the .part is already whole
        self.assertEqual(_read(dest), data)

    def test_bad_checksum_removed(self):
        _, spec = self._publish("e.bin", 4000)
        spec["sha256"] = "0" * 64
        dest = os.path.join(self.tmp, "e.bin")
        with self.assertRaises(sam_weights.DownloadError):
            sam_weights.download(spec, dest)
        self.assertFalse(os.path.exists(dest) or os.path.exists(dest + ".part"))

    def test_wrong_size_rejected(self):
        _, spec = self._publish("f.bin", 4000)
        spec.update(size=4001, sha256=None)
        with self.assertRaises(sam_weights.DownloadError):
            sam_weights.download(spec, os.path.join(self.tmp, "f.bin"))

    def test_cancel_keeps_partial_file(self):
        _, spec = self._publish("g.bin", 3 * sam_weights.CHUNK)
        dest = os.path.join(self.tmp, "g.bin")
        calls = []
        with self.assertRaises(sam_weights.DownloadCancelled):
            sam_weights.download(spec, dest, cancelled=lambda: calls.append(1) or len(calls) > 2)
        self.assertEqual(os.path.getsize(dest + ".part"), 2 * sam_weights.CHUNK)
        self.assertFalse(os.path.exists(dest))

    def test_http_error(self):
        spec = {"file": "x", "url": f"{self.base}/missing.bin"}
        with self.assertRaisesRegex(sam_weights.DownloadError, "404"):
            sam_weights.download(spec, os.path.join(self.tmp, "x"))

    def test_already_there_not_downloaded_again(self):
        data, spec = self._publish("h.bin", 1000)
        dest = os.path.join(self.tmp, "h.bin")
        _write(dest, data)
        sam_weights.download(spec, dest)
        self.assertEqual(_RangeHandler.served, [])

    def test_human_size(self):
        self.assertEqual(sam_weights.human_size(2564550879), "2.4 GB")
        self.assertEqual(sam_weights.human_size(500), "500 bytes")


class TestButtonDownload(_Server):

    def _app(self):
        app = make_app()
        app.cfg.SCRIPT_DIR = self.tmp
        self.base_data, app.cfg.SAM_BASE_WEIGHTS = self._publish("sam_vit_h_4b8939.pth", 3000)
        app.cfg.SAM_BASE_WEIGHTS.pop("sha256")   # like the real config: Meta's file is checked by size
        self.crack_data, app.cfg.SAM_CRACK_WEIGHTS = self._publish("cracksam_vit_h_lora_r4.pth", 800)
        app.cfg.SAM_CRACK_WEIGHTS.pop("size")
        return app

    def test_first_use_downloads_and_stores_settings(self):
        app = self._app()
        with mock.patch.object(app, "_confirm_sam_download", return_value=True) as ask, \
                mock.patch.object(app, "_pump_extra_events"):
            settings = app._usable_sam_settings()
            again = app._usable_sam_settings()
        ask.assert_called_once()
        model_dir = os.path.join(self.tmp, "SAM model")
        self.assertEqual(_read(os.path.join(model_dir, "sam_vit_h_4b8939.pth")), self.base_data)
        self.assertEqual(settings["backend"], "sam_lora")
        self.assertEqual(settings["model_type"], "vit_h")
        self.assertEqual(settings["lora_r"], 4)
        self.assertEqual(settings["code_dir"], os.path.join(HERE, "cracksam"))
        self.assertEqual(settings["lora_weights"], os.path.join(model_dir, "cracksam_vit_h_lora_r4.pth"))
        self.assertEqual(again, settings)
        with open(os.path.join(self.tmp, "sam_model.json")) as f:
            self.assertEqual(json.load(f), settings)

    def test_command_uses_the_downloaded_model(self):
        app = self._app()
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"):
            settings = app._usable_sam_settings()
        cmd = app._sam_command(settings, "/img.jpg", "/out.png")
        pairs = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual(pairs["--code-dir"], os.path.join(HERE, "cracksam"))
        self.assertTrue(pairs["--sam-checkpoint"].endswith("sam_vit_h_4b8939.pth"))

    def test_no_means_choose_files(self):
        app = self._app()
        with mock.patch.object(app, "_confirm_sam_download", return_value=False), \
                mock.patch.object(app, "_edit_sam_settings", return_value=None) as manual:
            self.assertIsNone(app._usable_sam_settings())
        manual.assert_called_once()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "SAM model")))

    def test_cancel(self):
        app = self._app()
        with mock.patch.object(app, "_confirm_sam_download", return_value=None), \
                mock.patch.object(app, "_edit_sam_settings") as manual:
            self.assertIsNone(app._usable_sam_settings())
        manual.assert_not_called()

    def test_already_downloaded_files_reused_without_asking(self):
        app = self._app()
        model_dir = os.path.join(self.tmp, "SAM model")
        os.makedirs(model_dir)
        _write(os.path.join(model_dir, "sam_vit_h_4b8939.pth"), self.base_data)
        _write(os.path.join(model_dir, "cracksam_vit_h_lora_r4.pth"), self.crack_data)
        with mock.patch.object(app, "_confirm_sam_download") as ask:
            settings = app._usable_sam_settings()
        ask.assert_not_called()
        self.assertEqual(settings["backend"], "sam_lora")
        self.assertEqual(_RangeHandler.served, [])

    def test_installed_app_asks_for_python(self):
        app = self._app()
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"), \
                mock.patch.object(app, "_default_sam_python", return_value=""), \
                mock.patch.object(app, "_ask_sam_python", return_value=sys.executable) as ask:
            settings = app._usable_sam_settings()
        ask.assert_called_once()
        self.assertEqual(settings["python"], sys.executable)

    def test_failed_download_reported(self):
        app = self._app()
        app.cfg.SAM_CRACK_WEIGHTS["url"] = f"{self.base}/not-published-yet.pth"
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"):
            self.assertIsNone(app._usable_sam_settings())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "sam_model.json")))

    def test_escape_stops_download(self):
        app = self._app()
        app._pending_keys.append(27)
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch("sam_weights.download", side_effect=lambda s, d, p, cancelled:
                           (_ for _ in ()).throw(sam_weights.DownloadCancelled())):
            self.assertIsNone(app._usable_sam_settings())

    def test_real_config_points_at_meta_and_the_release(self):
        cfg = make_app().cfg
        self.assertEqual(cfg.SAM_BASE_WEIGHTS["size"], 2564550879)
        self.assertTrue(cfg.SAM_BASE_WEIGHTS["url"].startswith("https://dl.fbaipublicfiles.com/segment_anything/"))
        self.assertIn("github.com/rob-spina/CrackSegmentation/releases/download/",
                      cfg.SAM_CRACK_WEIGHTS["url"])


class TestDownloadWindow(TestButtonDownload):
    """v2.0.1: the GUI shows a window with a progress bar while the model downloads."""

    def _app(self):
        import test_gui_panel_wiring as wiring
        app = wiring.make_app(self)
        app.cfg.SCRIPT_DIR = self.tmp
        self.base_data, app.cfg.SAM_BASE_WEIGHTS = self._publish("sam_vit_h_4b8939.pth", 3000)
        app.cfg.SAM_BASE_WEIGHTS.pop("sha256")
        self.crack_data, app.cfg.SAM_CRACK_WEIGHTS = self._publish("cracksam_vit_h_lora_r4.pth", 800)
        app.cfg.SAM_CRACK_WEIGHTS.pop("size")
        return app

    def test_window_shows_progress_and_closes(self):
        app = self._app()
        seen = []
        real_update = gm._SamDownloadWindow.update

        def spy(window, head, text, percent):
            seen.append((head, text, percent, window.bar.kw.get("value")))
            return real_update(window, head, text, percent)
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"), \
                mock.patch.object(gm._SamDownloadWindow, "update", spy), \
                mock.patch.object(gm._SamDownloadWindow, "close", autospec=True) as close:
            self.assertIsNotNone(app._usable_sam_settings())
        close.assert_called_once()
        self.assertIsNone(app._sam_download_window)
        heads = {h for h, _, _, _ in seen}
        self.assertIn("File 1 of 2: sam_vit_h_4b8939.pth", heads)
        self.assertIn("File 2 of 2: cracksam_vit_h_lora_r4.pth", heads)

    def test_cancel_button_stops_the_download(self):
        app = self._app()

        def slow(spec, dest, progress, cancelled):
            app._sam_download_window.cancel()      # the operator presses Cancel
            while not cancelled():
                pass
            raise sam_weights.DownloadCancelled()
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"), \
                mock.patch("sam_weights.download", side_effect=slow):
            self.assertIsNone(app._usable_sam_settings())
        self.assertIsNone(app._sam_download_window)

    def test_only_missing_files_downloaded(self):
        app = self._app()
        model_dir = os.path.join(self.tmp, "SAM model")
        os.makedirs(model_dir)
        _write(os.path.join(model_dir, "sam_vit_h_4b8939.pth"), self.base_data)
        with mock.patch.object(app, "_confirm_sam_download", return_value=True), \
                mock.patch.object(app, "_pump_extra_events"):
            self.assertIsNotNone(app._usable_sam_settings())
        self.assertEqual([p for p, _ in _RangeHandler.served], ["/cracksam_vit_h_lora_r4.pth"])

    def test_status_text(self):
        import time as _t
        status = gm.GuiCrackSegmentation.sam_download_status
        p = {"file": "sam_vit_h_4b8939.pth", "index": 1, "count": 2, "done": 1024 ** 3,
             "total": 2 * 1024 ** 3, "start": _t.time() - 64, "start_bytes": 0}
        head, text, percent = status(p)
        self.assertEqual(head, "File 1 of 2: sam_vit_h_4b8939.pth")
        self.assertEqual(percent, 50.0)
        self.assertTrue(text.startswith("1.0 GB of 2.0 GB (50%) - 16.0 MB/s - about 64 s left"), text)
        p["start"] = _t.time() - 120
        self.assertIn("about 2 min left", status(p)[1])
        p.update(total=None, count=1)
        head, text, percent = status(p)
        self.assertEqual(head, "sam_vit_h_4b8939.pth")
        self.assertIsNone(percent)


class TestBundledFiles(unittest.TestCase):

    def test_cracksam_code_bundled(self):
        for name in ("model.py", "sam_lora.py", "tiling.py"):
            self.assertTrue(os.path.isfile(os.path.join(HERE, "cracksam", name)), name)

    def test_model_card_states_the_training_data(self):
        card = _read(os.path.join(HERE, "docs", "sam_model", "MODEL_CARD.md")).decode("utf-8")
        self.assertIn("Trained on Facade390 (CC BY 4.0, Ge et al. 2024) and CrackSeg9K (CC0 1.0, Kulkarni et al. "
                      "2022). Some source images of CrackSeg9K (e.g. GAPS) carry their own terms; intended for "
                      "research use.", card)

    def test_spec_bundles_the_code(self):
        spec = _read(os.path.join(HERE, "packaging", "crack_segmentation_gui.spec")).decode("utf-8")
        for name in ("model.py", "sam_lora.py", "tiling.py"):
            self.assertIn(f'"cracksam" / "{name}"', spec)


class TestGuiDialogs(unittest.TestCase):

    def _gui_app(self):
        app = gm.GuiCrackSegmentation.__new__(gm.GuiCrackSegmentation)
        gm.CrackSegmentation.__init__(app)
        app._tk_root = fake_tkinter.Tk()
        return app

    def test_download_question(self):
        app = self._gui_app()
        with mock.patch.object(fake_tkinter.messagebox, "askyesnocancel", return_value=True) as ask:
            self.assertTrue(app._confirm_sam_download("About 2.4 GB"))
        self.assertIn("About 2.4 GB", ask.call_args[0][1])

    def test_python_question(self):
        app = self._gui_app()
        with mock.patch.object(fake_tkinter.simpledialog, "askstring", return_value="/venv/bin/python"):
            self.assertEqual(app._ask_sam_python(""), "/venv/bin/python")


if __name__ == "__main__":
    unittest.main()
