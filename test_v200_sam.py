"""test_v200_sam.py

Headless tests for the 2.0.0 [SAM auto-segment] button (sam_autoseg.py, code 136; settings code 137):
  - mask -> cracks: blobs thresholded, skeletonised into ordered 8-connected centre lines (main line, then
    branches), every blob pixel given to the nearest line;
  - the button: runs the model in a separate process (here a stand-in script, so no PyTorch is needed), adds the
    cracks with "source": "sam", writes the JSON and the binary crack mask (exactly SAM's mask), keeps the
    operator's cracks and ignore regions, replaces an earlier SAM run after a confirmation, one [U] undoes it;
  - model settings: asked once, stored in sam_model.json, refused when a file is missing;
  - GUI: AI sidebar section, Ctrl/Cmd+click -> settings, Tools menu entries, file dialogs.
sam_infer.py itself needs PyTorch: its test runs only where torch is installed.

Run with:  python3 -m unittest test_v200_sam -v
"""
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support"))
import fake_tkinter  # noqa: F401 -- registers the Tk stand-in before the GUI module is imported

import crack_segmentation_gui as gm
import sam_autoseg as sa
from sam_autoseg import SAM_AUTOSEG_CODE, SAM_SETTINGS_CODE
from test_gui_shell_parity import make_app

H, W = 200, 300

# Stand-in for sam_infer.py: same command line, writes a fixed probability map of the photo's size.
# A strong crack (prob 0.9) along y=100 with a branch going down from x=150, a weak one (0.55) along y=40,
# and a speck too small to be a crack.
_FAKE_INFER = textwrap.dedent('''
    import argparse, sys, time
    import cv2, numpy as np
    p = argparse.ArgumentParser()
    for a in ("--image", "--out", "--checkpoint", "--model-factory", "--tile", "--overlap", "--scale",
              "--batch-size", "--device", "--backend", "--code-dir", "--sam-checkpoint", "--model-type",
              "--lora-r", "--lora-weights", "--upscale"):
        p.add_argument(a, default="")
    args = p.parse_args()
    name = args.checkpoint + args.lora_weights
    if "fail" in name:
        print("[SAM] boom"); sys.exit(3)
    if "slow" in name:
        time.sleep(30)
    h, w = cv2.imread(args.image).shape[:2]
    prob = np.zeros((h, w), np.uint8)
    cv2.line(prob, (20, 100), (280, 100), 230, 3)
    cv2.line(prob, (150, 100), (150, 180), 230, 3)
    cv2.line(prob, (50, 40), (200, 40), 140, 3)
    cv2.circle(prob, (260, 20), 2, 230, -1)
    print("[SAM] tiles 1/1", flush=True)
    cv2.imwrite(args.out, prob)
''')


def _press(app, code):
    with mock.patch.object(app, "refresh_zoom_viewport"):
        return app.process_keypress(code)


def _sam_app(tmpdir, checkpoint_name="model.pt"):
    """App on a blank 300x200 photo, with SAM settings pointing at the stand-in script."""
    app = make_app()
    cfg = app.cfg
    cfg.H_img, cfg.W_img = H, W
    cfg.zoom_box = [0, 0, W, H]
    cfg.binary_mask = np.zeros((H, W), np.uint8)
    cfg.skeleton_mask = np.zeros((H, W), np.uint8)
    cfg.blue_visual_mask = np.zeros((H, W), np.uint8)
    cfg.green_visual_mask = np.zeros((H, W), np.uint8)
    cfg.saved_cracks = []
    img = np.full((H, W, 3), 200, np.uint8)
    os.makedirs(os.path.join(tmpdir, "Images"))
    img_path = os.path.join(tmpdir, "Images", "wall.png")
    cv2.imwrite(img_path, img)
    cfg.SCRIPT_DIR = tmpdir
    cfg.OUTPUT_FOLDER = os.path.join(tmpdir, "JSON files")
    cfg.CURRENT_IMAGE_PATH = img_path
    cfg.JSON_OUTPUT_PATH = os.path.join(cfg.OUTPUT_FOLDER, "wall.json")
    cfg.img_original = img
    script = os.path.join(tmpdir, "fake_infer.py")
    with open(script, "w") as f:
        f.write(_FAKE_INFER)
    checkpoint = os.path.join(tmpdir, checkpoint_name)
    open(checkpoint, "wb").close()
    app._save_sam_settings({"checkpoint": checkpoint, "model_factory": "", "python": sys.executable})
    app._sam_script_path = lambda: script
    return app


def _max_step(path):
    return max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(path, path[1:]))


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestMaskToCracks(unittest.TestCase):

    def test_small_blobs_dropped(self):
        prob = np.zeros((50, 50), np.uint8)
        prob[10, 5:45] = 200
        prob[30:32, 30:32] = 200
        mask = sa.probability_to_mask(prob, 0.5, 10)
        self.assertEqual(int(mask[10, 20]), 255)
        self.assertEqual(int(mask[30, 30]), 0)

    def test_threshold(self):
        prob = np.full((10, 10), 120, np.uint8)
        self.assertFalse(np.any(sa.probability_to_mask(prob, 0.5, 1)))
        self.assertTrue(np.all(sa.probability_to_mask(prob, 0.4, 1)))

    def test_branching_crack_gives_main_line_and_branch(self):
        mask = np.zeros((H, W), np.uint8)
        cv2.line(mask, (20, 100), (280, 100), 255, 3)
        cv2.line(mask, (150, 100), (150, 180), 255, 3)
        shapes = sa.mask_to_crack_shapes(mask, 15, 100)
        self.assertEqual(len(shapes), 2)
        main, branch = shapes[0][0], shapes[1][0]
        self.assertGreater(len(main), 250)
        self.assertGreater(len(branch), 60)
        for line, *_ in shapes:
            self.assertLessEqual(_max_step(line), 1)   # ordered, 8-connected like a traced crack

    def test_regions_cover_the_mask_exactly(self):
        mask = np.zeros((H, W), np.uint8)
        cv2.line(mask, (20, 100), (280, 100), 255, 5)
        cv2.line(mask, (150, 100), (150, 180), 255, 3)
        cv2.line(mask, (30, 20), (100, 60), 255, 2)
        total = np.zeros_like(mask)
        for line, region, x0, y0 in sa.mask_to_crack_shapes(mask, 15, 100):
            total[y0:y0 + region.shape[0], x0:x0 + region.shape[1]] |= region
        np.testing.assert_array_equal(total > 0, mask > 0)

    def test_short_lines_are_not_cracks(self):
        mask = np.zeros((H, W), np.uint8)
        cv2.line(mask, (20, 100), (30, 100), 255, 2)
        self.assertEqual(sa.mask_to_crack_shapes(mask, 15, 100), [])

    def test_max_cracks(self):
        mask = np.zeros((H, W), np.uint8)
        for y in range(10, 190, 20):
            cv2.line(mask, (20, y), (100 + y, y), 255, 2)
        shapes = sa.mask_to_crack_shapes(mask, 15, 3)
        self.assertEqual(len(shapes), 3)
        self.assertGreaterEqual(len(shapes[0][0]), len(shapes[-1][0]))


class TestSamButton(_TmpDir):

    def _run(self, app):
        with mock.patch.object(app, "_pump_extra_events"):
            _press(app, SAM_AUTOSEG_CODE)

    def test_cracks_json_and_masks(self):
        app = _sam_app(self.tmp)
        self._run(app)
        sam = [f for f in app.cfg.saved_cracks if f.get("source") == "sam"]
        self.assertEqual(len(sam), 3)   # main line, its branch, the weak line; the speck is dropped
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        shapes = [s for s in data["shapes"] if s["shape_type"] == "linestrip"]
        self.assertEqual(len(shapes), 3)
        self.assertTrue(all(s["source"] == "sam" and s["width_segments"][0]["fill"] for s in shapes))
        mask = cv2.imread(os.path.join(self.tmp, "Binary files", "wall-crack_mask.png"), cv2.IMREAD_GRAYSCALE)
        expected = np.zeros((H, W), np.uint8)
        cv2.line(expected, (20, 100), (280, 100), 255, 3)
        cv2.line(expected, (150, 100), (150, 180), 255, 3)
        cv2.line(expected, (50, 40), (200, 40), 255, 3)
        np.testing.assert_array_equal(mask > 0, expected > 0)   # SAM's own shape, not a dilated centre line

    def test_weak_crack_saved_as_uncertain(self):
        app = _sam_app(self.tmp)
        self._run(app)
        weak = [f for f in app.cfg.saved_cracks if all(p[1] == 40 for p in f["path"])]
        strong = [f for f in app.cfg.saved_cracks if all(p[1] == 100 for p in f["path"])]
        self.assertEqual(len(weak), 1)
        self.assertTrue(weak[0].get("uncertain_manual"))
        self.assertFalse(strong[0].get("uncertain_manual"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "Binary files", "wall-crack_uncertain_mask.png")))

    def test_uncertain_can_be_switched_off(self):
        app = _sam_app(self.tmp)
        app.cfg.SAM_UNCERTAIN_MEAN_PROB = None
        self._run(app)
        self.assertFalse(any(f.get("uncertain_manual") for f in app.cfg.saved_cracks))

    def test_operator_cracks_kept_and_not_duplicated(self):
        app = _sam_app(self.tmp)
        own = [(x, 100) for x in range(20, 281)]
        app.cfg.saved_cracks = [{'start': own[0], 'end': own[-1], 'path': own, 'active': True, 'session_id': 'x'}]
        self._run(app)
        self.assertIs(app.cfg.saved_cracks[0]['path'], own)
        sam = [f for f in app.cfg.saved_cracks if f.get("source") == "sam"]
        self.assertFalse(any(abs(p[1] - 100) <= 2 for f in sam for p in f["path"]))

    def test_ignore_region_left_alone(self):
        app = _sam_app(self.tmp)
        app.cfg.saved_ignores = [{'path': [(0, 20), (299, 20), (299, 60), (0, 60)], 'active': True}]
        self._run(app)
        self.assertFalse(any(20 <= p[1] <= 60 for f in app.cfg.saved_cracks for p in f["path"]))

    def test_second_run_replaces_after_confirmation(self):
        app = _sam_app(self.tmp)
        self._run(app)
        own = [(x, 150) for x in range(10, 60)]
        app.cfg.saved_cracks.append({'start': own[0], 'end': own[-1], 'path': own, 'active': True, 'session_id': 'x'})
        with mock.patch.object(app, "_confirm_sam_replace", return_value=False) as ask:
            self._run(app)
        ask.assert_called_once_with(3)
        self.assertEqual(len(app.cfg.saved_cracks), 4)
        with mock.patch.object(app, "_confirm_sam_replace", return_value=True):
            self._run(app)
        self.assertEqual(sum(1 for f in app.cfg.saved_cracks if f.get("source") == "sam"), 3)
        self.assertIn(own, [f["path"] for f in app.cfg.saved_cracks])

    def test_one_undo_removes_the_run(self):
        app = _sam_app(self.tmp)
        self._run(app)
        self.assertEqual(len(app.cfg.saved_cracks), 3)
        _press(app, ord('u'))
        self.assertEqual(app.cfg.saved_cracks, [])
        _press(app, ord('r'))
        self.assertEqual(len(app.cfg.saved_cracks), 3)

    def test_json_round_trip_keeps_source(self):
        app = _sam_app(self.tmp)
        self._run(app)
        with open(app.cfg.JSON_OUTPUT_PATH, encoding="utf-8") as f:
            shape = json.load(f)["shapes"][0]
        crack = app._build_loaded_crack(shape, [tuple(int(v) for v in p) for p in shape["points"]])
        self.assertEqual(crack["source"], "sam")
        self.assertIn("sam_mean_prob", crack)
        self.assertTrue(crack["width_segments"][0]["fill"])

    def test_model_failure_changes_nothing(self):
        app = _sam_app(self.tmp, "fail.pt")
        self._run(app)
        self.assertEqual(app.cfg.saved_cracks, [])
        self.assertFalse(os.path.exists(app.cfg.JSON_OUTPUT_PATH))

    def test_escape_cancels_only_sam(self):
        app = _sam_app(self.tmp, "slow.pt")
        app._pending_keys.append(27)
        with mock.patch.object(app, "_pump_extra_events"):
            self.assertFalse(app._run_sam_autoseg())
        self.assertEqual(app._pending_keys, [])
        self.assertEqual(app.cfg.saved_cracks, [])

    def test_no_photo(self):
        app = make_app()
        with mock.patch.object(app, "_usable_sam_settings") as settings:
            self.assertFalse(app._run_sam_autoseg())
        settings.assert_not_called()

    def test_command_line_module(self):
        app = _sam_app(self.tmp)
        app._save_sam_settings({"checkpoint": "/m/model.pt", "model_factory": "/m/f.py:make", "python": "/py"})
        cmd = app._sam_command(app._load_sam_settings(), "/img.jpg", "/out.png")
        self.assertEqual(cmd[0], "/py")
        self.assertEqual(cmd[cmd.index("--backend") + 1], "module")
        self.assertEqual(cmd[cmd.index("--model-factory") + 1], "/m/f.py:make")
        self.assertEqual(cmd[cmd.index("--tile") + 1], str(app.cfg.SAM_TILE_PX))

    def test_command_line_sam_lora(self):
        app = _sam_app(self.tmp)
        settings = {"backend": "sam_lora", "lora_weights": "/r/best_lora.pth", "sam_checkpoint": "/w/sam_vit_h.pth",
                    "model_type": "vit_h", "lora_r": 4, "code_dir": "/echo-twin", "python": "/py"}
        cmd = app._sam_command(settings, "/img.jpg", "/out.png")
        pairs = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual(pairs["--backend"], "sam_lora")
        self.assertEqual(pairs["--lora-weights"], "/r/best_lora.pth")
        self.assertEqual(pairs["--sam-checkpoint"], "/w/sam_vit_h.pth")
        self.assertEqual(pairs["--model-type"], "vit_h")
        self.assertEqual(pairs["--lora-r"], "4")
        self.assertEqual(pairs["--code-dir"], "/echo-twin")
        self.assertEqual(pairs["--tile"], "512")   # train.py --tile-size default

    def test_sam_lora_settings_run(self):
        app = _sam_app(self.tmp)
        code = os.path.join(self.tmp, "echo-twin")
        os.makedirs(code)
        for name in ("model.py", "best_lora.pth", "sam_vit_h_4b8939.pth"):
            open(os.path.join(code, name), "w").close()
        app._save_sam_settings({"backend": "sam_lora", "lora_weights": os.path.join(code, "best_lora.pth"),
                                "sam_checkpoint": os.path.join(code, "sam_vit_h_4b8939.pth"), "model_type": "vit_h",
                                "lora_r": 4, "code_dir": code, "python": sys.executable})
        self._run(app)
        self.assertEqual(sum(1 for f in app.cfg.saved_cracks if f.get("source") == "sam"), 3)


class TestSamSettings(_TmpDir):

    def test_asked_once_when_missing_then_stored(self):
        app = make_app()
        app.cfg.SCRIPT_DIR = self.tmp
        ckpt = os.path.join(self.tmp, "m.pt")
        open(ckpt, "wb").close()
        chosen = {"checkpoint": ckpt, "model_factory": "", "python": sys.executable}
        with mock.patch.object(app, "_choose_sam_settings", return_value=chosen) as ask, \
                mock.patch.object(app, "_confirm_sam_download", return_value=False):   # "choose the files"
            self.assertEqual(app._usable_sam_settings()["checkpoint"], ckpt)
            app._usable_sam_settings()
        ask.assert_called_once()
        with open(os.path.join(self.tmp, "sam_model.json")) as f:
            self.assertEqual(json.load(f)["checkpoint"], ckpt)

    def test_missing_file_refused(self):
        app = make_app()
        app.cfg.SCRIPT_DIR = self.tmp
        bad = {"checkpoint": os.path.join(self.tmp, "none.pt"), "model_factory": "", "python": sys.executable}
        with mock.patch.object(app, "_choose_sam_settings", return_value=bad):
            self.assertIsNone(app._edit_sam_settings())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "sam_model.json")))

    def test_sam_lora_needs_every_file(self):
        code = os.path.join(self.tmp, "code")
        os.makedirs(code)
        files = {k: os.path.join(self.tmp, k + ".pth") for k in ("lora", "base")}
        for path in files.values():
            open(path, "w").close()
        ok = {"backend": "sam_lora", "lora_weights": files["lora"], "sam_checkpoint": files["base"],
              "model_type": "vit_h", "lora_r": 4, "code_dir": code, "python": sys.executable}
        problem = sa.SamAutoSegMixin._sam_settings_problem
        self.assertIn("code folder", problem(ok))      # no model.py yet
        open(os.path.join(code, "model.py"), "w").close()
        self.assertIsNone(problem(ok))
        self.assertIn("LoRA", problem(dict(ok, lora_weights=os.path.join(self.tmp, "none.pth"))))
        self.assertIn("base SAM", problem(dict(ok, sam_checkpoint="")))
        self.assertIn("model type", problem(dict(ok, model_type="vit_x")))

    def test_model_type_from_file_name(self):
        guess = sa.SamAutoSegMixin._guess_sam_model_type
        self.assertEqual(guess("/w/sam_vit_h_4b8939.pth"), "vit_h")
        self.assertEqual(guess("sam_vit_b_01ec64.pth"), "vit_b")
        self.assertEqual(guess("weights.pth", "vit_l"), "vit_l")

    def test_missing_factory_file_refused(self):
        ckpt = os.path.join(self.tmp, "m.pt")
        open(ckpt, "wb").close()
        problem = sa.SamAutoSegMixin._sam_settings_problem(
            {"checkpoint": ckpt, "model_factory": os.path.join(self.tmp, "no.py") + ":f", "python": sys.executable})
        self.assertIn("factory", problem)

    def test_settings_key(self):
        app = make_app()
        with mock.patch.object(app, "_edit_sam_settings") as edit:
            _press(app, SAM_SETTINGS_CODE)
        edit.assert_called_once()


class TestSamGui(_TmpDir):

    def test_sidebar_menu_and_ctrl_click(self):
        entries = dict(gm._RIGHT_SIDEBAR_SECTIONS)["AI"]
        self.assertEqual([code for *_, code in entries][:2], [SAM_AUTOSEG_CODE, 138])   # then the 2.0.1 accept / exclude buttons
        self.assertEqual(gm._SIDEBAR_CTRL_CLICK_CODES[SAM_AUTOSEG_CODE], SAM_SETTINGS_CODE)
        codes = [code for _, code in gm._TOOL_COMMANDS]
        self.assertIn(SAM_AUTOSEG_CODE, codes)
        self.assertIn(SAM_SETTINGS_CODE, codes)

    def _gui_app(self):
        app = gm.GuiCrackSegmentation.__new__(gm.GuiCrackSegmentation)
        gm.CrackSegmentation.__init__(app)
        app._tk_root = fake_tkinter.Tk()
        return app

    def test_file_dialogs_sam_lora(self):
        app = self._gui_app()
        fake_tkinter.filedialog.next_paths = ["/runs/best_lora.pth", "/w/sam_vit_h_4b8939.pth"]
        fake_tkinter.filedialog.next_dirs = ["/Users/me/echo-twin"]
        with mock.patch.object(fake_tkinter.messagebox, "askyesnocancel", return_value=True), \
                mock.patch.object(fake_tkinter.simpledialog, "askstring", side_effect=["vit_h", "/venv/bin/python3"]), \
                mock.patch.object(fake_tkinter.simpledialog, "askinteger", return_value=4):
            got = app._choose_sam_settings({})
        self.assertEqual(got, {"backend": "sam_lora", "lora_weights": "/runs/best_lora.pth",
                               "sam_checkpoint": "/w/sam_vit_h_4b8939.pth", "model_type": "vit_h", "lora_r": 4,
                               "code_dir": "/Users/me/echo-twin", "python": "/venv/bin/python3"})

    def test_file_dialogs_module_with_factory(self):
        app = self._gui_app()
        fake_tkinter.filedialog.next_paths = ["/m/model.pth", "/m/build.py"]
        with mock.patch.object(fake_tkinter.messagebox, "askyesnocancel", side_effect=[False, True]), \
                mock.patch.object(fake_tkinter.simpledialog, "askstring",
                                  side_effect=["make_sam", "/venv/bin/python3"]):
            got = app._choose_sam_settings({})
        self.assertEqual(got, {"backend": "module", "checkpoint": "/m/model.pth",
                               "model_factory": "/m/build.py:make_sam", "python": "/venv/bin/python3"})

    def test_file_dialogs_whole_model(self):
        app = self._gui_app()
        fake_tkinter.filedialog.next_paths = ["/m/model.pt"]
        with mock.patch.object(fake_tkinter.messagebox, "askyesnocancel", side_effect=[False, False]), \
                mock.patch.object(fake_tkinter.simpledialog, "askstring", return_value="/py"):
            got = app._choose_sam_settings({})
        self.assertEqual(got["model_factory"], "")

    def test_cancel_dialogs(self):
        app = self._gui_app()
        fake_tkinter.filedialog.next_paths = []
        self.assertIsNone(app._choose_sam_settings({}))   # LoRA weights dialog cancelled
        with mock.patch.object(fake_tkinter.messagebox, "askyesnocancel", return_value=None):
            self.assertIsNone(app._choose_sam_settings({}))


class TestSamInferBackend(unittest.TestCase):

    def test_photo_turned_like_the_app(self):
        """Portrait photos stored landscape with an EXIF Orientation tag: the map must have the app's size."""
        from PIL import Image
        import sam_infer
        tmp = tempfile.mkdtemp()
        try:
            img = Image.fromarray(np.zeros((100, 200, 3), np.uint8))
            exif = img.getexif()
            exif[0x0112] = 6   # rotate 90 degrees when shown
            path = os.path.join(tmp, "portrait.jpg")
            img.save(path, exif=exif)
            h, w = cv2.imread(path).shape[:2]
            self.assertEqual(sam_infer.read_photo_upright(path).size, (w, h))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_backend_chosen_from_arguments(self):
        import sam_infer
        with mock.patch.object(sam_infer, "run_sam_lora") as lora, \
                mock.patch.object(sam_infer.cda, "_require_torch", side_effect=SystemExit("module path")):
            with self.assertRaises(SystemExit):
                sam_infer.run(sam_infer.parse_args(["--image", "a", "--out", "b", "--checkpoint", "m.pt"]))
            lora.assert_not_called()
            sam_infer.run(sam_infer.parse_args(["--image", "a", "--out", "b", "--lora-weights", "l.pth"]))
            lora.assert_called_once()


@unittest.skipUnless(sa.__name__ and __import__("importlib").util.find_spec("torch"), "PyTorch not installed")
class TestSamInferWithTorch(_TmpDir):

    def test_whole_saved_model(self):
        import torch
        import sam_infer
        model = torch.nn.Conv2d(3, 1, 1)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.constant_(model.bias, 2.0)
        ckpt = os.path.join(self.tmp, "m.pt")
        torch.save(model, ckpt)
        img = os.path.join(self.tmp, "a.png")
        cv2.imwrite(img, np.zeros((300, 700, 3), np.uint8))
        out = os.path.join(self.tmp, "p.png")
        sam_infer.run(sam_infer.parse_args(["--image", img, "--out", out, "--backend", "module", "--checkpoint", ckpt,
                                            "--tile", "256", "--overlap", "32", "--device", "cpu"]))
        prob = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(prob.shape, (300, 700))
        self.assertEqual(int(prob.mean()), round(255 / (1 + np.exp(-2.0))))

    def test_sam_lora_backend_with_stand_in_project(self):
        """The cracksam_lora path end to end, with a tiny stand-in model.py (no SAM download needed)."""
        import torch
        import sam_infer
        code = os.path.join(self.tmp, "echo-twin")
        os.makedirs(code)
        with open(os.path.join(code, "model.py"), "w") as f:
            f.write(textwrap.dedent('''
                import torch
                class Tiny(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.mask_decoder = torch.nn.Conv2d(3, 1, 1)
                def build_model(checkpoint_path, model_type="vit_b", lora_r=4, device="cpu", grad_checkpoint=False):
                    return Tiny().to(device)
                def forward_batch(sam, images, device):
                    return sam.mask_decoder(images.to(device))
            '''))
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_support", "cracksam_tiling.py"),
                    os.path.join(code, "tiling.py"))
        state = {"mask_decoder.weight": torch.zeros(1, 3, 1, 1), "mask_decoder.bias": torch.full((1,), -2.0)}
        torch.save(state, os.path.join(self.tmp, "best_lora.pth"))
        img = os.path.join(self.tmp, "a.png")
        cv2.imwrite(img, np.zeros((300, 700, 3), np.uint8))
        out = os.path.join(self.tmp, "p.png")
        sam_infer.run(sam_infer.parse_args([
            "--image", img, "--out", out, "--backend", "sam_lora", "--code-dir", code,
            "--sam-checkpoint", img, "--lora-weights", os.path.join(self.tmp, "best_lora.pth"),
            "--tile", "256", "--overlap", "64", "--device", "cpu"]))
        prob = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(prob.shape, (300, 700))
        self.assertEqual(int(np.median(prob)), round(255 / (1 + np.exp(2.0))))


if __name__ == "__main__":
    unittest.main()
