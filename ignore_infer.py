"""ignore_infer.py -- v2.0.1 [Auto ignore]: scene-class map (ADE20K: tree, plant, grass, fence, person, car, rock ...)
of one photo or of many, with a SegFormer semantic-segmentation model (Hugging Face transformers).

Run by CrackSegmentation in a separate Python process -- the same Python as SAM (PyTorch installed), with
`transformers` added once:  <that python> -m pip install transformers
The model (default nvidia/segformer-b5-finetuned-ade-640-640, about 340 MB) is downloaded by transformers the first
time into ~/.cache/huggingface and reused after that.

    python3 ignore_infer.py --image photo.jpg --out photo_classes.png --labels-out labels.json
    python3 ignore_infer.py --list photos.txt --out-dir "SAM model/ignore maps/<tag>" --labels-out labels.json

Output: an 8-bit PNG per photo, of the photo's proportions with its long side --long-side px, each pixel the ADE20K
class id (0-149); labels.json maps the ids to the class names. Progress lines start with "[IGNORE]". The photo is
read with OpenCV (EXIF orientation applied), exactly as the app shows it.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

DEFAULT_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"


def _log(text):
    print(f"[IGNORE] {text}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image")
    p.add_argument("--out")
    p.add_argument("--list", help="text file with one photo path per line (batch: the model is loaded once)")
    p.add_argument("--out-dir", help="batch output folder: <photo stem>.png")
    p.add_argument("--labels-out", help="writes {id: class name} of the model here")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--long-side", type=int, default=1024, help="the photo's long side is resized to this")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)
    if not ((args.image and args.out) or (args.list and args.out_dir)):
        p.error("give --image and --out, or --list and --out-dir")
    return args


def pick_device(torch, wanted):
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, device):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as e:
        _log(f"missing package: {e}. Install it once in this Python:  {sys.executable} -m pip install transformers")
        sys.exit(4)
    device = pick_device(torch, device)
    _log(f"loading {name} on {device} (the first time it is downloaded, ~340 MB)")
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForSemanticSegmentation.from_pretrained(name).to(device).eval()
    return torch, processor, model, device


def resized(image, long_side):
    h, w = image.shape[:2]
    scale = float(long_side) / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def classify(torch, processor, model, device, image_bgr, long_side):
    """uint8 class-id map of the resized photo."""
    small = resized(image_bgr, long_side)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt", do_resize=False)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        try:
            logits = model(pixel_values=pixel_values).logits
        except RuntimeError as e:          # e.g. an operation MPS does not have: CPU instead
            if device == "cpu":
                raise
            _log(f"{device} failed ({e}); running on the CPU")
            model.to("cpu")
            logits = model(pixel_values=pixel_values.to("cpu")).logits
        logits = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            photos = [line.strip() for line in f if line.strip()]
        os.makedirs(args.out_dir, exist_ok=True)
        jobs = [(p, os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")) for p in photos]
    else:
        jobs = [(args.image, args.out)]
    if args.skip_existing:
        jobs = [(p, o) for p, o in jobs if not os.path.isfile(o)]
    torch, processor, model, device = load_model(args.model, args.device)
    if args.labels_out:
        labels = {int(k): v for k, v in model.config.id2label.items()}
        os.makedirs(os.path.dirname(os.path.abspath(args.labels_out)), exist_ok=True)
        with open(args.labels_out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "id2label": labels}, f, indent=1)
    failed = 0
    for i, (photo, out) in enumerate(jobs, 1):
        start = time.time()
        image = cv2.imread(photo, cv2.IMREAD_COLOR)
        if image is None:
            _log(f"cannot read {photo}")
            failed += 1
            continue
        classes = classify(torch, processor, model, device, image, args.long_side)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        cv2.imwrite(out, classes)
        _log(f"photo {i}/{len(jobs)} {os.path.basename(photo)}: {time.time() - start:.1f} s")
    _log(f"done: {len(jobs) - failed} map(s)" + (f", {failed} failed" if failed else ""))
    return 1 if failed and failed == len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
