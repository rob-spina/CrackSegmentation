"""sam_infer.py -- v2.0.0 [SAM auto-segment]: crack probability map of ONE photo with the fine-tuned SAM model.

Run by CrackSegmentation in a separate Python process (the one chosen in [SAM model settings], with PyTorch
installed), so PyTorch is never loaded inside the app: the installed app does not bundle it, and the model
never shares the process with the Tk window. Two kinds of model:

  --backend sam_lora (default): the SAM + LoRA model of the cracksam_lora project (model.py, sam_lora.py,
      tiling.py, infer.py in --code-dir, e.g. ~/echo-twin): base SAM weights (sam_vit_h_4b8939.pth ...) plus the
      trained LoRA + mask-decoder weights (best_lora.pth). Inference is the same as that project's infer.py:
      native-resolution tiles from tiling.py, forward_batch() from model.py, the same soft blending.

    python3 sam_infer.py --image photo.jpg --out photo_prob.png --code-dir ~/echo-twin \
        --sam-checkpoint sam_vit_h_4b8939.pth --model-type vit_h --lora-r 4 --lora-weights best_lora.pth \
        [--tile 512 --overlap 128 --batch-size 4 --device auto] [--upscale 2]

  --backend module: any torch.nn.Module mapping a [B,3,H,W] ImageNet-normalised image to crack logits, as in
      cv_domain_adapt.py: --checkpoint is a whole saved module (torch.save(model) / TorchScript), or a state_dict
      that needs --model-factory FILE.py:FUNCTION.

Output: an 8-bit grey PNG of the photo's size, probability x 255. Progress lines start with "[SAM]".
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_domain_adapt as cda  # noqa: E402 -- tiling, normalisation and model helpers shared with training
import sam_weights  # noqa: E402 -- train_args.json upscale of the LoRA weights (standard library only)


def _log(text):
    print(f"[SAM] {text}", flush=True)


def load_model(factory, checkpoint):
    """The model: built by the factory and filled with the checkpoint, or the checkpoint itself when it is a
    whole saved module / TorchScript file."""
    torch = cda.torch
    if factory:
        return cda.build_model(factory, checkpoint)
    path = os.path.expanduser(checkpoint)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 1.13 has no weights_only
        obj = torch.load(path, map_location="cpu")
    except RuntimeError:
        obj = torch.jit.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    sys.exit("[SAM] The checkpoint holds only weights (a state_dict): set the model factory "
             "(the Python file that builds the model) in SAM model settings.")


def predict_with_progress(model, img_rgb, tile, overlap, device, batch_size):
    """cv_domain_adapt.predict_photo, printing how many tiles are done."""
    torch = cda.torch
    h, w = img_rgb.shape[:2]
    padded = cda.pad_to_tile(img_rgb, tile)
    ph, pw = padded.shape[:2]
    positions = cda.tile_grid(ph, pw, tile, overlap)
    probs, last = [], 0.0
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            chunk = positions[i:i + batch_size]
            batch = np.stack([cda.normalize(padded[y:y + tile, x:x + tile]) for y, x in chunk])
            logits = cda.crack_logits(model(torch.from_numpy(batch).to(device)))
            probs.extend(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
            if time.time() - last > 2 or i + batch_size >= len(positions):
                _log(f"tiles {min(i + batch_size, len(positions))}/{len(positions)}")
                last = time.time()
    return cda.stitch(probs, positions, ph, pw, tile, overlap)[:h, :w]


# --------------------------------------------------------------------------------------
# SAM + LoRA (cracksam_lora project)
# --------------------------------------------------------------------------------------

def _import_cracksam(code_dir):
    """model.py / tiling.py of the cracksam_lora project, imported from code_dir (it wins over this folder)."""
    code_dir = os.path.abspath(os.path.expanduser(code_dir))
    if not os.path.isfile(os.path.join(code_dir, "model.py")):
        sys.exit(f"[SAM] model.py not found in {code_dir} (the cracksam_lora code folder)")
    sys.path.insert(0, code_dir)
    import model as cracksam_model       # noqa: E402
    import tiling as cracksam_tiling     # noqa: E402
    return cracksam_model, cracksam_tiling


def load_sam_lora(args, device):
    """Same as infer.py: build_model() on the base SAM weights, then the LoRA + mask-decoder weights on top."""
    torch = cda.torch
    model_mod, tiling_mod = _import_cracksam(args.code_dir)
    sam = model_mod.build_model(checkpoint_path=os.path.expanduser(args.sam_checkpoint), model_type=args.model_type,
                                lora_r=args.lora_r, device=str(device))
    state = torch.load(os.path.expanduser(args.lora_weights), map_location=str(device))
    if isinstance(state, dict) and isinstance(state.get("model_state"), dict):
        state = state["model_state"]   # latest_checkpoint.pth / checkpoint_epoch_N.pth instead of best_lora.pth
    missing, unexpected = sam.load_state_dict(state, strict=False)
    lora_keys = [k for k in state if "linear_" in k] if isinstance(state, dict) else []
    _log(f"LoRA weights loaded ({len(state)} tensors, {len(lora_keys)} LoRA; {len(unexpected)} unexpected)")
    if not lora_keys:
        _log("WARNING: no LoRA tensor in --lora-weights: is it best_lora.pth / last_lora.pth?")
    return sam.eval(), model_mod, tiling_mod


def read_photo_upright(image_path):
    """The photo as a PIL RGB image turned by its EXIF orientation, like cv2.imread (and so the app) shows it:
    portrait photos from phones and drones are stored landscape with an 'Orientation' tag."""
    from PIL import Image, ImageOps
    with Image.open(os.path.expanduser(image_path)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _forward(model_mod, sam, batch, device, upscale):
    if abs(upscale - 1.0) < 1e-6:
        return model_mod.forward_batch(sam, batch, device)   # also an older model.py without the upscale argument
    return model_mod.forward_batch(sam, batch, device, upscale)


def predict_sam_lora(sam, model_mod, tiling_mod, image_path, tile, overlap, device, batch_size, upscale=1.0):
    """infer.py's tiling loop: probability map (float, original size) of the photo. upscale: each tile is
    enlarged by this factor inside forward_batch (train.py --upscale of the weights)."""
    torch = cda.torch
    orig = read_photo_upright(image_path)
    orig_w, orig_h = orig.size
    image = tiling_mod.pad_to_tile_rgb(orig, tile)
    width, height = image.size
    xs = tiling_mod.compute_tile_origins(width, tile, overlap)
    ys = tiling_mod.compute_tile_origins(height, tile, overlap)
    coords = [(x, y) for y in ys for x in xs]
    _log(f"photo {orig_w}x{orig_h}: {len(coords)} tiles of {tile}x{tile} (overlap {overlap})"
         + (f", enlarged x{upscale:g}" if abs(upscale - 1.0) > 1e-6 else ""))
    blend = tiling_mod.make_blend_weight(tile, overlap)
    acc = np.zeros((height, width), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)
    arr = np.asarray(image)
    use_amp = device.type == "cuda"
    last = 0.0
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        # torchvision's ToTensor: HWC uint8 -> CHW float in [0, 1] (sam.preprocess normalises).
        batch = torch.from_numpy(np.stack([arr[y:y + tile, x:x + tile].transpose(2, 0, 1) for x, y in chunk])
                                 .astype(np.float32) / 255.0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = _forward(model_mod, sam, batch, device, upscale)
            probs = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
        for (x, y), prob in zip(chunk, probs):
            acc[y:y + tile, x:x + tile] += prob * blend
            wsum[y:y + tile, x:x + tile] += blend
        if time.time() - last > 2 or start + batch_size >= len(coords):
            _log(f"tiles {min(start + batch_size, len(coords))}/{len(coords)}")
            last = time.time()
    return (acc / np.clip(wsum, 1e-6, None))[:orig_h, :orig_w]


def _write_probability(prob, out_path):
    out = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(out_path, out):
        sys.exit(f"[SAM] cannot write {out_path}")
    _log(f"probability map written: {out_path}")


def _backend(args):
    return args.backend or ("module" if args.checkpoint and not args.lora_weights else "sam_lora")


def make_predictor(args):
    """Loads the model ONCE and returns predict(image_path) -> float32 crack-probability map (photo size).
    Used for one photo here and for a whole folder by sam_batch.py."""
    backend = _backend(args)
    if backend == "module" and not args.checkpoint:
        sys.exit("[SAM] --checkpoint is required with --backend module")
    cda._require_torch()
    device = cda.pick_device(args.device)
    if backend == "sam_lora":
        _log(f"loading SAM {args.model_type} + LoRA r={args.lora_r} on {device} ...")
        sam, model_mod, tiling_mod = load_sam_lora(args, device)
        upscale = sam_weights.resolve_upscale(getattr(args, "upscale", None), args.lora_weights)
        if args.tile * upscale > 1024:
            sys.exit(f"[SAM] --tile {args.tile} x upscale {upscale:g} exceeds the SAM input (1024)")
        if abs(upscale - 1.0) > 1e-6:
            _log(f"tiles enlarged x{upscale:g} before SAM")
        return lambda image_path: predict_sam_lora(sam, model_mod, tiling_mod, image_path, args.tile,
                                                   args.overlap, device, args.batch_size, upscale)
    _log(f"loading the model on {device} ...")
    model = load_model(args.model_factory, args.checkpoint).to(device).eval()

    def predict(image_path):
        img = cda.read_image_rgb(os.path.expanduser(image_path))
        full_h, full_w = img.shape[:2]
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        _log(f"photo {full_w}x{full_h}, inference at {img.shape[1]}x{img.shape[0]}, "
             f"tile {args.tile}, overlap {args.overlap}")
        prob = predict_with_progress(model, img, args.tile, args.overlap, device, args.batch_size)
        if prob.shape != (full_h, full_w):
            prob = cv2.resize(prob, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        return prob
    return predict


def run_sam_lora(args):
    _write_probability(make_predictor(args)(args.image), args.out)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def run(args):
    if _backend(args) == "sam_lora":
        return run_sam_lora(args)
    _write_probability(make_predictor(args)(args.image), args.out)


def add_model_args(p):
    """The model options, shared with sam_batch.py."""
    p.add_argument("--backend", choices=["sam_lora", "module"], default=None,
                   help="default: module when only --checkpoint is given, otherwise sam_lora")
    # sam_lora: the cracksam_lora project (base SAM + best_lora.pth)
    p.add_argument("--code-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cracksam"),
                   help="folder with model.py / sam_lora.py / tiling.py (default: the bundled cracksam/)")
    p.add_argument("--sam-checkpoint", default="", help="base SAM weights, e.g. sam_vit_h_4b8939.pth")
    p.add_argument("--model-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-weights", default="", help="crack weights, e.g. cracksam_vit_h_lora_r4.pth")
    # module: cv_domain_adapt.py interface
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model-factory", default="", help="FILE.py:FUNCTION, needed when the checkpoint is a state_dict")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--upscale", type=float, default=None,
                   help="enlarge each tile by this factor before SAM (train.py --upscale of the weights); "
                        "default: read from train_args.json next to --lora-weights, 1 when absent")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True, help="8-bit PNG, probability x 255")
    add_model_args(p)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
