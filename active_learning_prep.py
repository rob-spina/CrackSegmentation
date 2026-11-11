"""active_learning_prep.py
v2.0.1 [Active learning]: turns the labelled photos of a CrackSegmentation data folder into the folder layout the
cracksam_lora train.py reads (--data-root with images/ and masks/, same file stem, crack = 255):

    python3 active_learning_prep.py --data-dir $WORK/drone_4k --out $WORK/cracksam_lora/al_round_01

Takes the photos of 'Training set/Images' and 'Validation set/Images' (never the Test set: it stays for comparing the
rounds) with their '<photo>-crack_mask.png' (in the set's 'Binary files' or in the data folder's). Photos are linked
(symlinks, copies where links are not possible) as <stem>.jpg -- the file content decides the format, so .JPG / .png
photos work with train.py --image-ext .jpg. Prints how many pairs, and which photos had no mask.

Uncertain cracks ('<photo>-crack_uncertain_mask.png') and [I] ignore regions ('<photo>-ignore_mask.png') are written
as a grey 128 band in the mask (crack = 255, background = 0): train.py --ignore-uncertain leaves them out of the loss
and of the IoU, so a doubt -- or an area with more cracks than could be traced -- is taught neither as crack nor as
background. --uncertain background turns them into background (older train.py).
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGES_DIR_NAME, BINARY_DIR_NAME, IGNORED_MASK_SUFFIXES   # noqa: E402

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TRAINING_SETS = ("Training set", "Validation set")
DETACHMENT_SUFFIX = "-detachment_mask.png"   # damage, never background: ignored even with --uncertain background


def find_mask(data_dir, set_dir, base, suffix="-crack_mask.png"):
    for folder in (os.path.join(set_dir, BINARY_DIR_NAME), os.path.join(data_dir, BINARY_DIR_NAME)):
        path = os.path.join(folder, f"{base}{suffix}")
        if os.path.isfile(path):
            return path
    return None


def write_mask_with_ignore(mask_path, ignore_paths, dest):
    """crack 255, ignored 128, background 0. Ignored = the uncertain cracks (the app's crack mask holds every crack,
    uncertain ones included; '-crack_uncertain_mask.png' those only) and the [I] ignore regions
    ('-ignore_mask.png', e.g. an area with too many small cracks to trace them all) and the [D] detachments
    ('-detachment_mask.png': damage, but not a crack, so never taught as background)."""
    import cv2
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    out = (mask > 127).astype("uint8") * 255
    for path in ([ignore_paths] if isinstance(ignore_paths, str) else ignore_paths):
        extra = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        if extra is not None and extra.shape == out.shape:
            out[extra > 127] = 128
    if os.path.lexists(dest):
        os.remove(dest)
    cv2.imwrite(dest, out)
    return int((out == 128).sum())


def collect_pairs(data_dir, sets=TRAINING_SETS):
    """[(photo, mask)] of the labelled photos of these sets, and [photos without a mask]."""
    pairs, missing = [], []
    for name in sets:
        set_dir = os.path.join(data_dir, name)
        images = os.path.join(set_dir, IMAGES_DIR_NAME)
        if not os.path.isdir(images):
            continue
        for f in sorted(os.listdir(images)):
            if not f.lower().endswith(PHOTO_EXTENSIONS) or f.startswith("."):
                continue
            base = os.path.splitext(f)[0]
            mask = find_mask(data_dir, set_dir, base)
            (pairs.append((os.path.join(images, f), mask)) if mask else missing.append(f))
    return pairs, missing


def _link(src, dest):
    if os.path.lexists(dest):
        os.remove(dest)
    try:
        os.symlink(os.path.abspath(src), dest)
    except OSError:
        shutil.copy2(src, dest)


def export(data_dir, out_dir, sets=TRAINING_SETS, uncertain="ignore"):
    pairs, missing = collect_pairs(data_dir, sets)
    export.ignored_photos = 0
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        for f in os.listdir(os.path.join(out_dir, sub)):   # a fresh export each round
            os.remove(os.path.join(out_dir, sub, f))
    for photo, mask in pairs:
        stem = os.path.splitext(os.path.basename(photo))[0]
        _link(photo, os.path.join(out_dir, "images", stem + ".jpg"))
        dest = os.path.join(out_dir, "masks", stem + ".png")
        set_dir = os.path.dirname(os.path.dirname(photo))
        suffixes = IGNORED_MASK_SUFFIXES if uncertain == "ignore" else (DETACHMENT_SUFFIX,)
        ignore = [m for m in (find_mask(data_dir, set_dir, stem, sfx) for sfx in suffixes) if m]
        if ignore and write_mask_with_ignore(mask, ignore, dest):
            export.ignored_photos += 1
        else:
            _link(mask, dest)
    return pairs, missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True, help="CrackSegmentation data folder")
    p.add_argument("--out", required=True, help="folder to create (images/ and masks/) for train.py --data-root")
    p.add_argument("--sets", nargs="*", default=list(TRAINING_SETS))
    p.add_argument("--uncertain", choices=["ignore", "background"], default="ignore",
                   help="uncertain cracks and [I] regions: grey 128 band for train.py --ignore-uncertain (default), "
                        "or background; [D] detachments are the grey band either way")
    args = p.parse_args(argv)
    pairs, missing = export(os.path.expanduser(args.data_dir), os.path.expanduser(args.out), args.sets,
                            args.uncertain)
    print(f"[AL prep] {len(pairs)} labelled photo(s) exported to {args.out} (images/ + masks/); "
          f"{export.ignored_photos} with uncertain cracks / ignore regions / detachments left out of the loss.")
    if missing:
        print(f"[AL prep] {len(missing)} photo(s) without a crack mask were skipped: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
    if not pairs:
        sys.exit("[AL prep] Nothing to train on: label photos and run Auto split first.")


if __name__ == "__main__":
    main()
