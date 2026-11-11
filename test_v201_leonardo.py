"""test_v201_leonardo.py

Headless tests for the 2.0.1 [Leonardo] button (leonardo_link.py, code 154). No network: the commands are checked as
built and _run_command is replaced by a scripted fake.
  - settings: defaults, kept in a local JSON (not in the data folder), unknown keys ignored, missing fields reported;
  - commands: ssh check, step login, rsync up (only the dataset folders, spaces escaped remotely), sbatch with
    ROUND / PREV / ... and git pull, squeue / sacct, rsync back of rounds and probability maps;
  - flows: connect with and without a valid certificate, copy retried after an interruption, job id remembered;
  - GUI: the window shows the fields, reads them back and keeps them when asked.

Run with:  python3 -m unittest test_v201_leonardo -v
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401

import crack_segmentation_gui as gm
import leonardo_link as ll
from test_gui_shell_parity import make_app


def _settings(local):
    s = ll.default_settings(local)
    s.update(user="rspina00", email="me@example.org")
    return s


class TestSettingsAndCommands(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crackseg_leo_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.local = os.path.join(self.tmp, "data")
        for d in ("Images", "Training set", "Test set", "SAM model", "Deleted images"):
            os.makedirs(os.path.join(self.local, d))

    def test_settings_saved_outside_the_data_folder_and_reloaded(self):
        path = os.path.join(self.tmp, "conf", "leonardo.json")
        s = _settings(self.local)
        s["round"], s["job_id"], s["password"] = "3", "123", "never"
        ll.save_settings(s, path)
        with open(path) as f:
            stored = json.load(f)
        self.assertNotIn("password", stored)
        again = ll.load_settings("/other", path)
        self.assertEqual((again["user"], again["round"], again["job_id"]), ("rspina00", "3", "123"))
        self.assertEqual(again["local_data"], self.local)
        self.assertFalse(ll.SETTINGS_PATH.startswith(self.local))
        self.assertEqual(ll.missing_fields(ll.default_settings(""), ("user", "host")), ["Leonardo username"])

    def test_commands(self):
        s = _settings(self.local)
        self.assertEqual(ll.check_command(s)[-2:], ["rspina00@login.leonardo.cineca.it", "echo crackseg-ok"])
        self.assertIn("BatchMode=yes", ll.check_command(s))
        self.assertEqual(ll.login_command(s), ["step", "ssh", "login", "me@example.org", "--provisioner", "cineca-hpc"])
        up = ll.upload_command(s)
        self.assertEqual(up[0], "rsync")
        self.assertIn(os.path.join(self.local, "Images"), up)
        self.assertIn(os.path.join(self.local, "Training set"), up)
        self.assertFalse(any("SAM model" in a or "Deleted images" in a for a in up))
        self.assertEqual(up[-1], "rspina00@login.leonardo.cineca.it:/leonardo_work/IscrC_AID-ECHO/drone_4k/")
        start = ll.start_command(s)[-1]
        self.assertNotIn("git pull", start)            # default: the code is copied with rsync
        self.assertIn("MAPS_ONLY=0", start)
        code = ll.code_upload_command(s, "/Users/me/CrackSegmentation")
        self.assertEqual(code[0], "rsync")
        self.assertIn("/Users/me/CrackSegmentation/", code)
        self.assertEqual(code[code.index("--exclude", code.index("*.py")) + 1], "*")
        self.assertIn("cracksam/***", code)
        self.assertEqual(code[-1], "rspina00@login.leonardo.cineca.it:/leonardo_work/IscrC_AID-ECHO/CrackSegmentation/")
        s["code_sync"] = "git"
        start = ll.start_command(s)[-1]
        self.assertIn("git pull --ff-only", start)
        self.assertIn("ROUND=1", start)
        self.assertIn("PREV=/leonardo_work/IscrC_AID-ECHO/cracksam_lora/runs/facade390_vit_h_fixed/best_lora.pth",
                      start)
        self.assertIn("sbatch docs/sam_model/active_learning_round_leonardo.sbatch", start)
        fetch = ll.fetch_commands(s)
        self.assertEqual(len(fetch), 2)
        self.assertTrue(fetch[0][1][-2].endswith("drone_4k/SAM\\ model/rounds/"))
        self.assertEqual(fetch[1][0], os.path.join(self.local, "SAM model/probability maps"))
        self.assertEqual(ll.parse_job_id("Already up to date.\nSubmitted batch job 987654\n"), "987654")


class _Scripted:
    """Fake _run_command: answers by the program name, in order."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, cmd, timeout=None):
        self.calls.append(cmd)
        key = cmd[0] if cmd[0] != "ssh" else "ssh:" + cmd[-1].split()[0]
        queue = self.answers.get(key) or self.answers.get(cmd[0])
        return queue.pop(0) if queue else (0, "")


class TestFlows(TestSettingsAndCommands):

    def _app(self, answers):
        app = make_app()
        app.cfg.SCRIPT_DIR = self.local
        fake = _Scripted(answers)
        return app, fake

    def _run(self, app, fake, action, s):
        with mock.patch.object(app, "_run_command", side_effect=fake), redirect_stdout(io.StringIO()):
            return action(s)

    def test_connect_with_a_valid_certificate(self):
        app, fake = self._app({"ssh:echo": [(0, "crackseg-ok")]})
        ok, msg = self._run(app, fake, app.leonardo_connect, _settings(self.local))
        self.assertTrue(ok)
        self.assertEqual([c[0] for c in fake.calls], ["ssh"])

    def test_connect_logs_in_when_the_certificate_expired(self):
        app, fake = self._app({"ssh:echo": [(255, "Permission denied"), (0, "crackseg-ok")], "step": [(0, "")]})
        ok, msg = self._run(app, fake, app.leonardo_connect, _settings(self.local))
        self.assertTrue(ok)
        self.assertIn("certificate valid 12 h", msg)
        self.assertEqual([c[0] for c in fake.calls], ["ssh", "step", "ssh"])

    def test_connect_needs_the_email_to_log_in(self):
        app, fake = self._app({"ssh:echo": [(255, "")]})
        s = _settings(self.local)
        s["email"] = ""
        ok, msg = self._run(app, fake, app.leonardo_connect, s)
        self.assertFalse(ok)
        self.assertIn("E-mail", msg)

    def test_upload_retried_then_job_started_and_remembered(self):
        app, fake = self._app({"ssh:echo": [(0, "crackseg-ok")], "ssh:mkdir": [(0, "")],
                               "rsync": [(23, "connection closed"), (0, "")],
                               "ssh:cd": [(0, "Submitted batch job 4242")]})
        s = _settings(self.local)
        ok, msg = self._run(app, fake, app.leonardo_send_and_start, s)
        self.assertTrue(ok, msg)
        self.assertEqual(s["job_id"], "4242")
        rsyncs = [c for c in fake.calls if c[0] == "rsync"]
        self.assertEqual(len(rsyncs), 3)                 # data twice (retried), then the code
        self.assertIn("*.py", rsyncs[-1])

    def test_maps_only_job(self):
        app, fake = self._app({"ssh:echo": [(0, "crackseg-ok")], "ssh:mkdir": [(0, "")], "rsync": [(0, "")],
                               "ssh:cd": [(0, "Submitted batch job 77")]})
        s = _settings(self.local)
        s.update(job_kind="maps", round="", epochs="", code_sync="git")
        ok, msg = self._run(app, fake, app.leonardo_send_and_start, s)
        self.assertTrue(ok, msg)
        self.assertIn("Maps x2", msg)
        start = [c for c in fake.calls if c[0] == "ssh" and any("sbatch" in a for a in c)][-1][-1]
        self.assertIn("MAPS_ONLY=1", start)
        self.assertEqual([c[0] for c in fake.calls].count("rsync"), 1)   # git: data only

    def test_round_script_maps_only(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "sam_model",
                                "active_learning_round_leonardo.sbatch")).read()
        self.assertIn('if [ "$MAPS_ONLY" = "1" ]', src)
        block = src[src.index('if [ "$MAPS_ONLY" = "1" ]; then\n    echo'):src.index("exit 0")]
        self.assertIn('--lora-weights "$PREV"', block)
        self.assertIn('--upscale "$UPSCALE"', block)
        self.assertNotIn("--skip-labelled", block)

    def test_status_and_fetch(self):
        app, fake = self._app({"ssh:squeue": [(0, "RUNNING 1:02:03")]})
        s = _settings(self.local)
        self.assertFalse(self._run(app, fake, app.leonardo_status, s)[0])   # no job yet
        s["job_id"] = "4242"
        ok, msg = self._run(app, fake, app.leonardo_status, s)
        self.assertEqual(msg, "Job 4242: RUNNING 1:02:03")
        ok, msg = self._run(app, fake, app.leonardo_fetch, s)
        self.assertTrue(ok)
        self.assertIn("Active learning", msg)
        self.assertTrue(os.path.isdir(os.path.join(self.local, "SAM model", "rounds")))

    def test_real_runner_reports_a_missing_program(self):
        app = make_app()
        with redirect_stdout(io.StringIO()):
            code, out = app._run_command(["crackseg-no-such-program-xyz"])
        self.assertIsNone(code)


class TestGui(TestSettingsAndCommands):

    def test_window_fields_and_remember(self):
        import test_gui_panel_wiring as wiring
        app = wiring.make_app(self)
        s = _settings(self.local)
        dialog = gm._LeonardoDialog(app._tk_root, app, s)
        self.assertEqual(dialog.entries["user"].get(), "rspina00")
        dialog.entries["round"].delete(0, "end")
        dialog.entries["round"].insert(0, "2")
        path = os.path.join(self.tmp, "kept.json")
        with mock.patch.object(gm, "save_leonardo_settings", side_effect=lambda st: ll.save_settings(st, path)), \
                mock.patch.object(app, "leonardo_status", return_value=(True, "Job 1: RUNNING")):
            dialog.run(app.leonardo_status)
            self.assertIn("RUNNING", dialog.message.kw["text"])
            dialog.close()
        with open(path) as f:
            self.assertEqual(json.load(f)["round"], "2")

    def test_button_and_menu(self):
        codes = [entry[3] for _, entries in gm._LEFT_SIDEBAR_SECTIONS for entry in entries]
        self.assertIn(ll.LEONARDO_CODE, codes)
        self.assertIn(ll.LEONARDO_CODE, [code for _, code in gm._TOOL_COMMANDS])


if __name__ == "__main__":
    unittest.main()
