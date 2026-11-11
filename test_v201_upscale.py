"""v2.0.1 [x2 tile upscale]: SAM sees each 512 tile enlarged to 1024 (thin cracks at twice the size), in training
(train.py --upscale, written in train_args.json) and in inference (sam_infer.py --upscale, read from that file);
maps at different factors go to different cache folders."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sam_weights  # noqa: E402
import sam_autoseg  # noqa: E402
import leonardo_link as ll  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        sam_autoseg._TAG_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def weights(self, upscale=None, content=b"lora"):
        path = os.path.join(self.tmp, "best_lora.pth")
        with open(path, "wb") as f:
            f.write(content)
        if upscale is not None:
            with open(os.path.join(self.tmp, "train_args.json"), "w") as f:
                json.dump({"tile_size": 512, "upscale": upscale}, f)
        return path


class TestWeightsUpscale(_Tmp):

    def test_read_from_train_args(self):
        self.assertEqual(sam_weights.lora_upscale(self.weights(2.0)), 2.0)

    def test_missing_or_broken_file_is_one(self):
        path = self.weights()
        self.assertEqual(sam_weights.lora_upscale(path), 1.0)
        with open(os.path.join(self.tmp, "train_args.json"), "w") as f:
            f.write("not json")
        self.assertEqual(sam_weights.lora_upscale(path), 1.0)
        self.assertEqual(sam_weights.lora_upscale(""), 1.0)

    def test_override_wins(self):
        path = self.weights(2.0)
        self.assertEqual(sam_weights.resolve_upscale(None, path), 2.0)
        self.assertEqual(sam_weights.resolve_upscale(1, path), 1.0)
        self.assertEqual(sam_weights.resolve_upscale("bad", path), 2.0)
        self.assertEqual(sam_weights.resolve_upscale(None, ""), 1.0)

    def test_suffix(self):
        self.assertEqual(sam_weights.upscale_suffix(1), "")
        self.assertEqual(sam_weights.upscale_suffix(2.0), "-x2")
        self.assertEqual(sam_weights.upscale_suffix(1.5), "-x1.5")


class TestMapsFolder(_Tmp):

    def test_x2_weights_get_their_own_folder(self):
        path = self.weights(2)
        tag = sam_autoseg.model_tag(path)
        self.assertTrue(tag.startswith("lora-") and tag.endswith("-x2"), tag)
        self.assertEqual(sam_autoseg.model_tag(path, upscale=1), tag[:-3])

    def test_x1_weights_keep_the_old_folder(self):
        """Maps computed before (e.g. lora-d3a8d5a27cc9 on Leonardo) stay valid."""
        path = self.weights()
        tag = sam_autoseg.model_tag(path)
        self.assertRegex(tag, r"^lora-[0-9a-f]{12}$")
        self.assertEqual(sam_autoseg.model_tag(path, upscale=2), tag + "-x2")

    def test_config_default_is_auto(self):
        from config import Config
        self.assertIsNone(Config().SAM_UPSCALE)


class TestAppCommand(_Tmp):

    def _app(self):
        from test_v200_sam import _sam_app
        return _sam_app(self.tmp)

    def _cmd(self, app, weights):
        settings = {"backend": "sam_lora", "lora_weights": weights, "sam_checkpoint": "/w/sam_vit_h.pth",
                    "model_type": "vit_h", "lora_r": 4, "code_dir": "/echo-twin", "python": "/py"}
        cmd = app._sam_command(settings, "/img.jpg", "/out.png")
        return cmd[cmd.index("--upscale") + 1]

    def test_upscale_from_the_weights(self):
        app = self._app()
        self.assertEqual(self._cmd(app, self.weights(2)), "2")

    def test_upscale_one_without_train_args(self):
        app = self._app()
        self.assertEqual(self._cmd(app, self.weights()), "1")

    def test_config_forces_it(self):
        app = self._app()
        app.cfg.SAM_UPSCALE = 1
        self.assertEqual(self._cmd(app, self.weights(2)), "1")


class TestSamInfer(_Tmp):

    def test_argument(self):
        import sam_infer
        args = sam_infer.parse_args(["--image", "a", "--out", "b", "--lora-weights", "l.pth"])
        self.assertIsNone(args.upscale)
        args = sam_infer.parse_args(["--image", "a", "--out", "b", "--lora-weights", "l.pth", "--upscale", "2"])
        self.assertEqual(args.upscale, 2.0)

    def test_forward_dispatch(self):
        import sam_infer
        calls = []
        mod = SimpleNamespace(forward_batch=lambda *a: calls.append(a) or "logits")
        self.assertEqual(sam_infer._forward(mod, "sam", "batch", "cpu", 1.0), "logits")
        self.assertEqual(calls[-1], ("sam", "batch", "cpu"))   # older model.py: no 4th argument
        sam_infer._forward(mod, "sam", "batch", "cpu", 2.0)
        self.assertEqual(calls[-1], ("sam", "batch", "cpu", 2.0))

    def _predictor(self, argv):
        import sam_infer
        args = sam_infer.parse_args(["--image", "a", "--out", "b"] + argv)
        with mock.patch.object(sam_infer.cda, "_require_torch"), \
                mock.patch.object(sam_infer.cda, "pick_device", return_value="cpu"), \
                mock.patch.object(sam_infer, "load_sam_lora", return_value=("sam", "mod", "tiling")), \
                mock.patch.object(sam_infer, "predict_sam_lora", return_value="prob") as predict:
            predict_fn = sam_infer.make_predictor(args)
            predict_fn("photo.jpg")
        return predict.call_args[0]

    def test_predictor_reads_the_weights_upscale(self):
        call = self._predictor(["--lora-weights", self.weights(2)])
        self.assertEqual(call[-1], 2.0)

    def test_predictor_explicit_upscale(self):
        call = self._predictor(["--lora-weights", self.weights(2), "--upscale", "1"])
        self.assertEqual(call[-1], 1.0)

    def test_tile_too_large(self):
        with self.assertRaises(SystemExit):
            self._predictor(["--lora-weights", self.weights(2), "--tile", "1024"])


class TestModelCode(unittest.TestCase):

    def test_bundled_model_has_upscale(self):
        src = open(os.path.join(HERE, "cracksam", "model.py")).read()
        self.assertIn("upscale: float = 1.0", src)
        self.assertIn("input_size=(in_h, in_w), original_size=(H, W)", src)
        self.assertIn('mode="bilinear"', src)


class TestLeonardo(unittest.TestCase):

    def test_round_script(self):
        src = open(os.path.join(HERE, "docs", "sam_model", "active_learning_round_leonardo.sbatch")).read()
        self.assertIn('UPSCALE=${UPSCALE:-2}', src)
        self.assertIn('--upscale "$UPSCALE" \\', src)                      # train.py
        self.assertIn('--overlap 128 --upscale "$UPSCALE"', src)          # sam_batch.py maps
        self.assertIn('cp "$RUN_DIR/train_args.json" "$ROUNDS_DIR/train_args.json"', src)

    def test_start_command_passes_upscale(self):
        s = ll.default_settings("/data")
        s.update(user="u", email="e@x", round="1")
        self.assertIn("UPSCALE=2", ll.start_command(s)[-1])
        s["upscale"] = "1"
        self.assertIn("UPSCALE=1", ll.start_command(s)[-1])
        del s["upscale"]   # settings saved before this field existed
        self.assertIn("UPSCALE=2", ll.start_command(s)[-1])


if __name__ == "__main__":
    unittest.main()
