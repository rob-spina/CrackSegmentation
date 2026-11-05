"""cv_domain_adapt.py -- domain adaptation of a crack-segmentation model to 4K drone photos, by tiling,
with cross-validation by building (the folds written by CrackSegmentation's [CV folds] button).

The model was first fine-tuned on low-resolution datasets (Facade390, CrackSeg9K); here it is adapted to the
annotated drone photos. Each 4K photo is cut into overlapping tiles (optionally after a rescale, so cracks have
about the width in pixels the model saw before); training samples tiles around cracks and on clean wall,
validation runs on the WHOLE photo by sliding window and stitches the tiles back with a smooth weighting.

Labels (built from the CrackSegmentation exports):
    crack_mask > 0      -> 1 (crack)          everything else -> 0 (background)
    uncertain_mask > 0  -> 255 (ignored)      ignore_mask > 0 -> 255 (ignored, applied last)
Ignored pixels count neither in the loss nor in the metrics.

Usage (run from anywhere; paths are examples):
    # 1. cross-validation: one adaptation per fold, metrics per fold and mean +- std
    python3 cv_domain_adapt.py cv --training-set "~/Documents/CrackSegmentation/Training set" \\
        --model-factory my_model.py:build_model --checkpoint finetuned_crackseg9k.pt --out runs/cv1

    # 2. final model on every fold (epochs = the mean best epoch found by cross-validation)
    python3 cv_domain_adapt.py final --training-set ... --model-factory ... --checkpoint ... \\
        --epochs 30 --out runs/final

    # 3. one evaluation on the test set (photos in 'Test set/Images', masks still in 'Binary files')
    python3 cv_domain_adapt.py test --images "~/Documents/CrackSegmentation/Test set/Images" \\
        --masks "~/Documents/CrackSegmentation/Binary files" --model-factory ... --checkpoint runs/final/model.pt

The model: --model-factory FILE.py:FUNCTION must return a torch.nn.Module mapping a [B,3,H,W] float image
(0-1, ImageNet-normalised) to [B,1,H,W] logits (or [B,2,H,W]: channel 1 is taken as the crack). Without it,
segmentation_models_pytorch Unet(resnet34) is built. --checkpoint loads a state_dict (or a dict holding one
under 'state_dict' / 'model'). Parameters whose name starts with --encoder-prefix get a lower learning rate
and can be frozen for the first epochs (--freeze-encoder-epochs), the usual recipe for domain adaptation.

Requires: numpy, opencv-python, torch (and segmentation_models_pytorch only for the default model).
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:          # the numpy helpers below stay usable (and testable) without torch
    torch = None
    F = None

IGNORE = 255
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ======================================================================================
# Data: folds, images, labels (numpy only)
# ======================================================================================

def read_folds(training_set):
    """Rows of 'Training set/folds.csv' that have a crack mask, with absolute paths."""
    path = os.path.join(training_set, "folds.csv")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found: press 'CV folds' in CrackSegmentation first.")
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("crack_mask"):
                print(f"[skip] {row['photo']}: no crack mask")
                continue
            rec = {"photo": row["photo"], "group": row["group"], "fold": int(row["fold"])}
            for key in ("image", "crack_mask", "uncertain_mask", "ignore_mask"):
                rec[key] = os.path.join(training_set, row[key]) if row.get(key) else ""
            records.append(rec)
    return records


def read_image_rgb(path, scale=1.0):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _read_mask(path, shape, scale):
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if scale != 1.0 or m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def build_label(shape, crack_path, uncertain_path="", ignore_path="", scale=1.0):
    """uint8 label: 1 crack, 0 background, 255 ignored (uncertain cracks first, ignore regions last)."""
    label = np.zeros(shape, dtype=np.uint8)
    crack = _read_mask(crack_path, shape, scale)
    if crack is not None:
        label[crack > 0] = 1
    for path in (uncertain_path, ignore_path):
        m = _read_mask(path, shape, scale)
        if m is not None:
            label[m > 0] = IGNORE
    return label


def load_record(rec, scale=1.0):
    img = read_image_rgb(rec["image"], scale)
    label = build_label(img.shape[:2], rec["crack_mask"], rec.get("uncertain_mask", ""),
                        rec.get("ignore_mask", ""), scale)
    return img, label


def normalize(img_rgb):
    """HxWx3 uint8 -> 3xHxW float32, ImageNet-normalised."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ======================================================================================
# Tiling (numpy only)
# ======================================================================================

def tile_starts(length, tile, overlap):
    """Start offsets covering [0, length) with tiles of `tile` px overlapping by at least `overlap`;
    the last tile ends exactly at the border. A side shorter than a tile gives one start (0)."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return sorted(set(starts))


def tile_grid(height, width, tile, overlap):
    return [(y, x) for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]


def pad_to_tile(arr, tile, value=0):
    """Pads an image/label on the bottom/right up to at least tile x tile (small photos)."""
    h, w = arr.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph == 0 and pw == 0:
        return arr
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad, mode="constant", constant_values=value)


def blend_window(tile, overlap):
    """tile x tile weights: 1 in the middle, ramping down linearly over `overlap` px towards the borders
    (never 0), so stitched tiles join without seams."""
    if overlap <= 0:
        return np.ones((tile, tile), dtype=np.float32)
    ramp = np.ones(tile, dtype=np.float32)
    edge = np.linspace(1.0 / (overlap + 1), 1.0, overlap, endpoint=False, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def stitch(tile_probs, positions, height, width, tile, overlap):
    """Weighted average of the tile predictions placed at their (y, x) positions -> height x width."""
    acc = np.zeros((height, width), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)
    w = blend_window(tile, overlap)
    for prob, (y, x) in zip(tile_probs, positions):
        h, wd = min(tile, height - y), min(tile, width - x)
        acc[y:y + h, x:x + wd] += prob[:h, :wd] * w[:h, :wd]
        wsum[y:y + h, x:x + wd] += w[:h, :wd]
    return acc / np.maximum(wsum, 1e-6)


def sample_tile_origin(label, tile, rng, crack_fraction):
    """Top-left corner of a training tile: with probability crack_fraction around a random crack pixel
    (so thin cracks are not drowned by wall), otherwise anywhere."""
    h, w = label.shape
    if rng.random() < crack_fraction:
        ys, xs = np.nonzero(label == 1)
        if len(ys):
            i = rng.integers(len(ys))
            y = int(ys[i]) - tile // 2 + int(rng.integers(-tile // 4, tile // 4 + 1))
            x = int(xs[i]) - tile // 2 + int(rng.integers(-tile // 4, tile // 4 + 1))
            return int(np.clip(y, 0, max(0, h - tile))), int(np.clip(x, 0, max(0, w - tile)))
    return int(rng.integers(0, max(1, h - tile + 1))), int(rng.integers(0, max(1, w - tile + 1)))


def augment(img, label, rng):
    """Flips, 90-degree rotations and a mild brightness/contrast change (drone light varies a lot)."""
    k = int(rng.integers(4))
    img, label = np.rot90(img, k), np.rot90(label, k)
    if rng.random() < 0.5:
        img, label = img[:, ::-1], label[:, ::-1]
    alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


class TileDataset:
    """Map-style dataset for torch's DataLoader: `tiles_per_image` random tiles per photo and epoch.
    Photos are kept in memory after the first read (93 photos at 4K are about 3 GB; use --scale < 1 or
    fewer photos per fold if RAM is short)."""

    def __init__(self, records, tile, tiles_per_image, crack_fraction, scale, seed=0):
        self.records, self.tile, self.n = records, tile, tiles_per_image
        self.crack_fraction, self.scale, self.seed, self.epoch = crack_fraction, scale, seed, 0
        self._cache = {}

    def __len__(self):
        return len(self.records) * self.n

    def _photo(self, i):
        if i not in self._cache:
            img, label = load_record(self.records[i], self.scale)
            self._cache[i] = (pad_to_tile(img, self.tile), pad_to_tile(label, self.tile, IGNORE))
        return self._cache[i]

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, self.epoch, index))
        img, label = self._photo(index // self.n)
        y, x = sample_tile_origin(label, self.tile, rng, self.crack_fraction)
        t = self.tile
        img_t, lab_t = augment(img[y:y + t, x:x + t], label[y:y + t, x:x + t], rng)
        return normalize(img_t), lab_t.astype(np.int64)


# ======================================================================================
# Metrics (numpy only)
# ======================================================================================

def confusion(pred, label, tolerance_px=0):
    """(tp, fp, fn) over non-ignored pixels. With tolerance_px > 0, a predicted pixel within that distance
    of a crack counts as correct and vice versa (usual for thin cracks, whose borders are uncertain)."""
    valid = label != IGNORE
    gt = (label == 1) & valid
    pr = (pred > 0) & valid
    if tolerance_px <= 0:
        tp = int(np.count_nonzero(pr & gt))
        return tp, int(np.count_nonzero(pr)) - tp, int(np.count_nonzero(gt)) - tp
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    gt_near = cv2.dilate(gt.astype(np.uint8), k) > 0
    pr_near = cv2.dilate(pr.astype(np.uint8), k) > 0
    tp_pred = int(np.count_nonzero(pr & gt_near))
    tp_gt = int(np.count_nonzero(gt & pr_near))
    return tp_pred, int(np.count_nonzero(pr)) - tp_pred, int(np.count_nonzero(gt)) - tp_gt


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def evaluate_probs(prob_label_pairs, thresholds, tolerance_px):
    """Dataset-level scores (pixels summed over photos) for each threshold."""
    out = {}
    for thr in thresholds:
        tp = fp = fn = 0
        for prob, label in prob_label_pairs:
            a, b, c = confusion(prob >= thr, label, tolerance_px)
            tp, fp, fn = tp + a, fp + b, fn + c
        out[thr] = scores(tp, fp, fn)
    return out


def mean_std(values):
    values = list(values)
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ======================================================================================
# Model (torch)
# ======================================================================================

def _require_torch():
    if torch is None:
        sys.exit("PyTorch is not installed: pip install torch (and segmentation-models-pytorch for the default model).")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(factory, checkpoint):
    if factory:
        file, _, func = factory.partition(":")
        spec = importlib.util.spec_from_file_location("user_model", os.path.expanduser(file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, func or "build_model")()
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    if checkpoint:
        state = torch.load(os.path.expanduser(checkpoint), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[model] checkpoint loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    return model


def crack_logits(output):
    """[B,1,H,W] logits from the model output (also [B,2,H,W] or a dict with 'out')."""
    if isinstance(output, dict):
        output = output.get("out", next(iter(output.values())))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.shape[1] == 2:
        output = output[:, 1:2] - output[:, 0:1]
    return output


def masked_loss(logits, target, dice_weight, pos_weight):
    """BCE + soft Dice over non-ignored pixels."""
    target = target.unsqueeze(1)
    valid = (target != IGNORE).float()
    y = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * valid).sum() + 1)
    return bce + dice_weight * dice


def set_encoder_trainable(model, prefix, trainable):
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = trainable


def make_optimizer(model, args):
    enc = [p for n, p in model.named_parameters() if n.startswith(args.encoder_prefix)]
    rest = [p for n, p in model.named_parameters() if not n.startswith(args.encoder_prefix)]
    groups = [{"params": rest, "lr": args.lr}]
    if enc:
        groups.append({"params": enc, "lr": args.lr * args.encoder_lr_factor})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad() if torch is not None else (lambda f: f)
def predict_photo(model, img_rgb, tile, overlap, device, batch_size):
    """Probability map of a whole photo by sliding window (tiles stitched with blend_window)."""
    h, w = img_rgb.shape[:2]
    padded = pad_to_tile(img_rgb, tile)
    ph, pw = padded.shape[:2]
    positions = tile_grid(ph, pw, tile, overlap)
    probs = []
    for i in range(0, len(positions), batch_size):
        batch = np.stack([normalize(padded[y:y + tile, x:x + tile]) for y, x in positions[i:i + batch_size]])
        logits = crack_logits(model(torch.from_numpy(batch).to(device)))
        probs.extend(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
    return stitch(probs, positions, ph, pw, tile, overlap)[:h, :w]


def evaluate_records(model, records, args, device):
    model.eval()
    pairs = []
    for rec in records:
        img, label = load_record(rec, args.scale)
        pairs.append((predict_photo(model, img, args.tile, args.overlap, device, args.batch_size), label))
    return evaluate_probs(pairs, args.thresholds, args.tolerance)


# ======================================================================================
# Training
# ======================================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(train_records, val_records, args, out_dir, epochs):
    """Adapts a fresh copy of the model on train_records. With val_records, keeps the epoch with the best
    F1 (at the best threshold) and returns its scores; without them, keeps the last epoch."""
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(args.device)
    seed_everything(args.seed)
    model = build_model(args.model_factory, args.checkpoint).to(device)
    data = TileDataset(train_records, args.tile, args.tiles_per_image, args.crack_fraction, args.scale, args.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, drop_last=True)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best = {"f1": -1.0}
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "threshold", "precision", "recall", "f1", "iou", "seconds"])
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            set_encoder_trainable(model, args.encoder_prefix, epoch > args.freeze_encoder_epochs)
            data.epoch = epoch
            model.train()
            total, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = masked_loss(crack_logits(model(x)), y, args.dice_weight, args.pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, n = total + float(loss), n + 1
            scheduler.step()
            row = [epoch, round(total / max(1, n), 5)]
            if val_records:
                res = evaluate_records(model, val_records, args, device)
                thr = max(res, key=lambda t: res[t]["f1"])
                s = res[thr]
                row += [thr, *(round(s[k], 4) for k in ("precision", "recall", "f1", "iou"))]
                if s["f1"] > best["f1"]:
                    best = dict(s, threshold=thr, epoch=epoch)
                    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                row += ["", "", "", "", ""]
            row.append(round(time.time() - t0, 1))
            writer.writerow(row)
            log.flush()
            print(f"  epoch {epoch}/{epochs}  " + "  ".join(str(v) for v in row[1:]))
    if not val_records:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return best


def run_cv(args):
    records = read_folds(args.training_set)
    folds = sorted({r["fold"] for r in records}) if not args.folds else args.folds
    summary = []
    for fold in folds:
        train = [r for r in records if r["fold"] != fold]
        val = [r for r in records if r["fold"] == fold]
        print(f"\n=== fold {fold}: {len(train)} training photos, {len(val)} validation photos "
              f"({len({r['group'] for r in val})} buildings) ===")
        best = train_one(train, val, args, os.path.join(args.out, f"fold_{fold}"), args.epochs)
        summary.append(dict(best, fold=fold, n_val=len(val)))
    write_summary(args.out, summary)


def write_summary(out, summary):
    keys = ["fold", "n_val", "epoch", "threshold", "precision", "recall", "f1", "iou"]
    with open(os.path.join(out, "cv_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
        for label, fn in (("mean", 0), ("std", 1)):
            writer.writerow({"fold": label, **{k: round(mean_std([s[k] for s in summary])[fn], 4)
                                               for k in ("epoch", "threshold", "precision", "recall", "f1", "iou")}})
    print("\n=== cross-validation ===")
    for k in ("f1", "iou", "precision", "recall"):
        m, s = mean_std([r[k] for r in summary])
        print(f"  {k:9s} {m:.3f} +- {s:.3f}")
    m_epoch, _ = mean_std([r["epoch"] for r in summary])
    m_thr, _ = mean_std([r["threshold"] for r in summary])
    print(f"  best epoch on average {m_epoch:.1f}, threshold {m_thr:.2f}  -> use them for 'final'")


def run_final(args):
    records = read_folds(args.training_set)
    print(f"Final adaptation on all {len(records)} training photos for {args.epochs} epochs")
    train_one(records, [], args, args.out, args.epochs)
    with open(os.path.join(args.out, "settings.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "func"}, f, indent=2)


def test_records(images_dir, masks_dir):
    """Test photos with their masks found by name in 'Binary files' (the Test set keeps only photos)."""
    records = []
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        base = os.path.splitext(name)[0]
        mask = lambda suffix: (os.path.join(masks_dir, base + suffix)
                               if os.path.isfile(os.path.join(masks_dir, base + suffix)) else "")
        if not mask("-crack_mask.png"):
            print(f"[skip] {name}: no crack mask in {masks_dir}")
            continue
        records.append({"photo": name, "image": os.path.join(images_dir, name),
                        "crack_mask": mask("-crack_mask.png"), "uncertain_mask": mask("-crack_uncertain_mask.png"),
                        "ignore_mask": mask("-ignore_mask.png")})
    return records


def run_test(args):
    records = test_records(os.path.expanduser(args.images), os.path.expanduser(args.masks))
    device = pick_device(args.device)
    model = build_model(args.model_factory, args.checkpoint).to(device)
    res = evaluate_records(model, records, args, device)
    print(f"\n=== test set: {len(records)} photos ===")
    for thr, s in res.items():
        print(f"  threshold {thr:.2f}: " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))


# ======================================================================================
# Command line
# ======================================================================================

def _common(p):
    p.add_argument("--model-factory", default="", help="FILE.py:FUNCTION returning the nn.Module (default: smp Unet resnet34)")
    p.add_argument("--checkpoint", default="", help="weights of the model fine-tuned on Facade390 / CrackSeg9K")
    p.add_argument("--tile", type=int, default=512, help="tile side in px (use the input size of the fine-tuning)")
    p.add_argument("--overlap", type=int, default=128, help="overlap between tiles at inference, px")
    p.add_argument("--scale", type=float, default=1.0,
                   help="rescale the photos first, e.g. 0.5, so crack widths in px match the fine-tuning data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--tolerance", type=int, default=2, help="px of tolerance in precision/recall (0 = strict)")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--seed", type=int, default=42)


def _training(p):
    p.add_argument("--training-set", required=True, help="the 'Training set' folder (with folds.csv)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4, help="decoder learning rate (low: adaptation, not training)")
    p.add_argument("--encoder-lr-factor", type=float, default=0.1, help="encoder lr = lr x this")
    p.add_argument("--encoder-prefix", default="encoder.", help="parameter-name prefix of the encoder")
    p.add_argument("--freeze-encoder-epochs", type=int, default=3, help="epochs with the encoder frozen at first")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tiles-per-image", type=int, default=32, help="random tiles per photo and epoch")
    p.add_argument("--crack-fraction", type=float, default=0.7, help="share of tiles centred on a crack")
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=3.0, help="BCE weight of crack pixels (cracks are rare)")
    p.add_argument("--workers", type=int, default=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cv_p = sub.add_parser("cv", help="cross-validation by building, one adaptation per fold")
    _common(cv_p)
    _training(cv_p)
    cv_p.add_argument("--folds", type=int, nargs="*", help="only these folds (default: all)")
    cv_p.set_defaults(func=run_cv)
    final_p = sub.add_parser("final", help="adapt on every training photo (no validation)")
    _common(final_p)
    _training(final_p)
    final_p.set_defaults(func=run_final)
    test_p = sub.add_parser("test", help="evaluate a model once on the test set")
    _common(test_p)
    test_p.add_argument("--images", required=True, help="'Test set/Images'")
    test_p.add_argument("--masks", required=True, help="'Binary files' (where the test masks stayed)")
    test_p.set_defaults(func=run_test)
    args = parser.parse_args(argv)
    if hasattr(args, "training_set"):
        args.training_set = os.path.expanduser(args.training_set)
    return args


def main(argv=None):
    args = parse_args(argv)
    _require_torch()
    if hasattr(args, "out"):
        os.makedirs(args.out, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
