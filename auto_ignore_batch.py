"""auto_ignore_batch.py -- v2.0.1 [Auto ignore] for many photos, and its comparison with the hand-drawn ignore regions.

1. Class maps of every photo (the model is loaded once; on a Mac ~5-15 s a photo, on a GPU well under 1 s):

    python3 auto_ignore_batch.py --data-dir ~/Documents/CrackSegmentation --python ~/venv-sam/bin/python

   They go to '<data>/SAM model/ignore maps/<model>/', where [Auto ignore] finds them (no wait in the app).
   --only-ignored: only the photos that already have a '<photo>-ignore_mask.png' (for step 2, quick).

2. Comparison with the ignore masks drawn by hand ('Binary files/<photo>-ignore_mask.png'):

    python3 auto_ignore_batch.py --data-dir ~/Documents/CrackSegmentation --evaluate [--report report.csv]

   For each photo: IoU, precision (share of the automatic regions inside the hand-drawn ones) and recall (share of the
   hand-drawn regions found). Then, over all photos, the ADE20K classes under the hand-drawn regions: a class that is
   mostly inside them and missing from AUTO_IGNORE_CLASSES is a candidate to add; a class of the list that is mostly
   outside them, a candidate to remove.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, IMAGES_DIR_NAME, BINARY_DIR_NAME   # noqa: E402
import auto_ignore as ai   # noqa: E402

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SETS = ("", "Training set", "Validation set", "Test set")


def find_photos(data_dir):
    out = []
    for sub in SETS:
        folder = os.path.join(data_dir, sub, IMAGES_DIR_NAME)
        if os.path.isdir(folder):
            out += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                    if f.lower().endswith(PHOTO_EXTENSIONS) and not f.startswith(".")]
    return out


def manual_mask_path(data_dir, photo):
    base = os.path.splitext(os.path.basename(photo))[0]
    set_dir = os.path.dirname(os.path.dirname(photo))
    for folder in (os.path.join(set_dir, BINARY_DIR_NAME), os.path.join(data_dir, BINARY_DIR_NAME)):
        path = os.path.join(folder, f"{base}-ignore_mask.png")
        if os.path.isfile(path):
            return path
    return None


def maps_dir(data_dir, cfg):
    return os.path.join(data_dir, "SAM model", ai.IGNORE_MAPS_DIR, ai.model_slug(cfg.AUTO_IGNORE_MODEL,
                                                                                 cfg.AUTO_IGNORE_LONG_SIDE))


def compute_maps(data_dir, cfg, python, photos, device="auto", skip_existing=True):
    out_dir = maps_dir(data_dir, cfg)
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(photos))
        listing = f.name
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ignore_infer.py")
    cmd = [python, "-u", script, "--list", listing, "--out-dir", out_dir,
           "--labels-out", os.path.join(out_dir, "labels.json"), "--model", cfg.AUTO_IGNORE_MODEL,
           "--long-side", str(cfg.AUTO_IGNORE_LONG_SIDE), "--device", device]
    if skip_existing:
        cmd.append("--skip-existing")
    try:
        return subprocess.call(cmd)
    finally:
        os.remove(listing)


def evaluate_photo(classes, id2label, manual, cfg):
    """(iou, precision, recall, n polygons) of the automatic regions against a hand-drawn ignore mask."""
    H, W = manual.shape[:2]
    polys, _ = ai.regions_from_classes(classes, id2label, cfg.AUTO_IGNORE_CLASSES, (W, H),
                                       close_fraction=cfg.AUTO_IGNORE_CLOSE_FRACTION,
                                       min_area_fraction=cfg.AUTO_IGNORE_MIN_AREA_FRACTION,
                                       margin_px=cfg.AUTO_IGNORE_MARGIN_PX,
                                       epsilon_fraction=cfg.AUTO_IGNORE_EPSILON_FRACTION,
                                       max_regions=cfg.AUTO_IGNORE_MAX_REGIONS)
    auto = ai.polygons_mask(polys, (W, H)) > 0
    hand = manual > 127
    inter = float(np.count_nonzero(auto & hand))
    union = float(np.count_nonzero(auto | hand))
    a, h = float(np.count_nonzero(auto)), float(np.count_nonzero(hand))
    return (inter / union if union else 1.0, inter / a if a else (1.0 if not h else 0.0),
            inter / h if h else 1.0, len(polys))


def class_coverage(classes, manual, totals):
    """Adds, per class id, [pixels inside the hand-drawn regions, pixels outside] at the class-map size."""
    small = cv2.resize((manual > 127).astype(np.uint8), (classes.shape[1], classes.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
    for cid in np.unique(classes):
        sel = classes == cid
        t = totals.setdefault(int(cid), [0, 0])
        t[0] += int(np.count_nonzero(sel & small))
        t[1] += int(np.count_nonzero(sel & ~small))


def evaluate(data_dir, cfg, report=None, out=print):
    mdir = maps_dir(data_dir, cfg)
    id2label = ai.read_labels(os.path.join(mdir, "labels.json"))
    if not id2label:
        out(f"[AUTO IGNORE] No class maps in {mdir}: run without --evaluate first.")
        return []
    rows, totals = [], {}
    for photo in find_photos(data_dir):
        manual_path = manual_mask_path(data_dir, photo)
        map_path = os.path.join(mdir, os.path.splitext(os.path.basename(photo))[0] + ".png")
        if not manual_path or not os.path.isfile(map_path):
            continue
        manual = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
        classes = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if manual is None or classes is None:
            continue
        iou, prec, rec, n = evaluate_photo(classes, id2label, manual, cfg)
        class_coverage(classes, manual, totals)
        rows.append((os.path.basename(photo), iou, prec, rec, n))
    if not rows:
        out("[AUTO IGNORE] No photo has both a hand-drawn ignore mask and a class map.")
        return rows
    out(f"{'photo':48s} {'IoU':>5s} {'prec':>5s} {'recall':>6s} {'regions':>7s}")
    for name, iou, prec, rec, n in rows:
        out(f"{name[:48]:48s} {iou:5.2f} {prec:5.2f} {rec:6.2f} {n:7d}")
    mean = np.mean([[r[1], r[2], r[3]] for r in rows], axis=0)
    out(f"{'MEAN (' + str(len(rows)) + ' photos)':48s} {mean[0]:5.2f} {mean[1]:5.2f} {mean[2]:6.2f}")
    chosen = ai.class_ids(id2label, cfg.AUTO_IGNORE_CLASSES)
    hand_total = float(sum(v[0] for v in totals.values())) or 1.0
    out("\nADE20K classes under the hand-drawn ignore regions (share of those regions; share of the class inside them):")
    for cid, (inside, outside) in sorted(totals.items(), key=lambda kv: -kv[1][0])[:20]:
        if inside / hand_total < 0.005:
            continue
        name = str(id2label.get(cid, cid)).split(";")[0]
        inside_share = inside / float(inside + outside)
        flag = "in the list" if cid in chosen else ("ADD?" if inside_share >= 0.5 else "")
        out(f"  {name:22s} {100 * inside / hand_total:5.1f}% of the regions  {100 * inside_share:5.1f}% of it inside  {flag}")
    for cid in chosen:
        inside, outside = totals.get(cid, (0, 0))
        if inside + outside and inside / float(inside + outside) < 0.2:
            out(f"  {str(id2label.get(cid)).split(';')[0]:22s} in the list but mostly OUTSIDE the hand-drawn regions "
                f"({100 * inside / float(inside + outside):.0f}% inside): REMOVE?")
    if report:
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["photo", "iou", "precision", "recall", "regions"])
            w.writerows(rows)
        out(f"\n[AUTO IGNORE] Report written to {report}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--python", default=sys.executable, help="Python with PyTorch + transformers (the SAM one)")
    p.add_argument("--device", default="auto")
    p.add_argument("--only-ignored", action="store_true", help="only photos with a hand-drawn ignore mask")
    p.add_argument("--redo", action="store_true", help="compute again the maps already there")
    p.add_argument("--evaluate", action="store_true", help="compare with the hand-drawn ignore masks")
    p.add_argument("--report", help="CSV of the comparison")
    args = p.parse_args(argv)
    data_dir = os.path.expanduser(args.data_dir)
    cfg = Config()
    if args.evaluate:
        evaluate(data_dir, cfg, args.report)
        return 0
    photos = find_photos(data_dir)
    if args.only_ignored:
        photos = [ph for ph in photos if manual_mask_path(data_dir, ph)]
    print(f"[AUTO IGNORE] {len(photos)} photo(s) -> {maps_dir(data_dir, cfg)}")
    return compute_maps(data_dir, cfg, os.path.expanduser(args.python), photos, args.device, not args.redo)


if __name__ == "__main__":
    sys.exit(main())
