"""sam_batch.py -- v2.0.0: [SAM auto-segment] for a whole folder of photos, from the command line (no window).

Does for every photo of '<data folder>/Images' what the SAM auto-segment button does for one: the fine-tuned SAM
model finds the cracks, they become editable cracks of the app ("source": "sam", low-confidence ones uncertain),
and the LabelMe JSON ('JSON files') and the binary masks ('Binary files') are written. Open the data folder in
CrackSegmentation afterwards (mode 2) to check and correct them.

The model is loaded ONCE for the whole folder, so this is the way to process hundreds of 4K photos, e.g. on a GPU
node of Leonardo (see docs/sam_model/sam_batch_leonardo.sbatch). It needs the packages of the app (opencv,
scikit-image) and of the model (torch, torchvision, segment-anything) in the same Python.

    python3 sam_batch.py --data-dir ~/Documents/CrackSegmentation \\
        --sam-checkpoint sam_vit_h_4b8939.pth --lora-weights cracksam_vit_h_lora_r4.pth [--limit 30]

Photos that already have a JSON are left alone (--existing skip, the default). --existing update loads their
JSON, keeps every crack and detachment of the operator and replaces only the cracks of an earlier SAM run.
A report (sam_batch_report.csv) with the cracks found per photo, an uncertainty score (share of crack-like pixels
the model hesitates on) and the time taken is written in the data folder. The crack maps are kept in
'<data folder>/SAM model/probability maps/<model>/', where the app reuses them: with --prob-only nothing else is
written, so that [SAM click to accept] is instant in the app and the most uncertain photos can be annotated first.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import Config, JSON_DIR_NAME, IMAGES_DIR_NAME  # noqa: E402
from smart_segmentation import CrackSegmentation  # noqa: E402
from sam_autoseg import (PROBABILITY_DIR, model_tag, probability_cache_path, probability_uncertainty,  # noqa: E402
                         read_cached_probability, write_cached_probability)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
REPORT_NAME = "sam_batch_report.csv"
REPORT_FIELDS = ["photo", "status", "sam_cracks", "uncertain", "operator_cracks_kept", "uncertainty", "seconds",
                 "message"]


class BatchCrackSegmentation(CrackSegmentation):
    """CrackSegmentation without a window: the editing pipeline and the JSON / mask export, nothing interactive."""

    def __init__(self, data_dir, predict, cfg=None, tag="model", prob_only=False):
        super().__init__(cfg)
        self.cfg.SCRIPT_DIR = os.path.abspath(os.path.expanduser(data_dir))
        self._set_queue_folders()
        self._predict = predict
        self._tag = tag
        self._prob_only = prob_only

    def _batch_probability(self, image_path, shape):
        """The photo's crack map (uint8): from the cache when this model already did it, else from the model.
        Always left in the cache, where [SAM auto-segment] and [SAM click to accept] find it."""
        cache = probability_cache_path(self._sam_model_dir(), self._tag, image_path)
        prob_u8 = read_cached_probability(cache, image_path, shape)
        if prob_u8 is None:
            prob = self._predict(image_path)
            prob_u8 = np.clip(np.rint(np.asarray(prob, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
            if prob_u8.shape != tuple(shape):
                raise ValueError(f"probability map {prob_u8.shape[::-1]} does not match the photo {shape[::-1]}")
            write_cached_probability(cache, prob_u8)
        return prob_u8

    # No screen, no questions: the batch decides by its own options.
    def refresh_zoom_viewport(self):
        return None

    def _pump_extra_events(self):
        return None

    def auto_assign_building_group_by_similarity(self, target_path):
        """Never renames photos into building groups on its own: the operator does that in the app."""
        return None

    def _confirm_sam_replace(self, count):
        return True

    def process(self, image_path, update_existing):
        """Segments one photo and writes its files. Returns a report row (dict)."""
        name = os.path.basename(image_path)
        row = {"photo": name, "status": "", "sam_cracks": 0, "uncertain": 0, "operator_cracks_kept": 0,
               "uncertainty": "", "seconds": 0, "message": ""}
        has_json = os.path.isfile(os.path.join(self.cfg.OUTPUT_FOLDER, os.path.splitext(name)[0] + ".json"))
        if self._prob_only:
            started = time.time()
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("the photo cannot be read")
            prob_u8 = self._batch_probability(image_path, img.shape)
            row.update(status="map", uncertainty=probability_uncertainty(prob_u8),
                       seconds=round(time.time() - started, 1))
            return row
        if has_json and not update_existing:
            row["status"] = "skipped"
            row["message"] = "already has a JSON (use --existing update to add SAM cracks to it)"
            return row
        started = time.time()
        self.cfg.modalita_scelta = "2" if has_json else "1"
        self.initialize_image_session(image_path, mode=self.cfg.modalita_scelta)
        prob_u8 = self._batch_probability(image_path, (self.cfg.H_img, self.cfg.W_img))
        keep = [f for f in self.cfg.saved_cracks if f.get('active', True) and not self._is_sam_crack(f)]
        new_cracks = self._sam_cracks_from_probability(prob_u8, keep)
        self._apply_sam_cracks(new_cracks)
        row.update(status="updated" if has_json else "segmented", sam_cracks=len(new_cracks),
                   uncertain=sum(1 for f in new_cracks if f.get('uncertain_manual')),
                   operator_cracks_kept=len(keep), uncertainty=probability_uncertainty(prob_u8),
                   seconds=round(time.time() - started, 1))
        return row


def list_photos(images_dir, only=None):
    names = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith("."))
    if only:
        wanted = {n.lower() for n in only}
        names = [n for n in names if n.lower() in wanted or os.path.splitext(n)[0].lower() in wanted]
    return [os.path.join(images_dir, n) for n in names]


def write_report(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args, predictor_factory=None):
    """Processes the folder. predictor_factory(args) -> predict(image_path); default: sam_infer.make_predictor."""
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    images_dir = os.path.join(data_dir, IMAGES_DIR_NAME)
    if not os.path.isdir(images_dir):
        sys.exit(f"[SAM batch] No '{IMAGES_DIR_NAME}' folder in {data_dir}")
    photos = list_photos(images_dir, args.only)
    json_dir = os.path.join(data_dir, JSON_DIR_NAME)
    todo = [p for p in photos if args.prob_only or args.existing == "update" or not os.path.isfile(
        os.path.join(json_dir, os.path.splitext(os.path.basename(p))[0] + ".json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[SAM batch] {len(photos)} photo(s) in {images_dir}; {len(todo)} to segment "
          f"({'adding to existing JSONs' if args.existing == 'update' else 'photos with a JSON are skipped'}).")
    if not todo:
        return []
    if predictor_factory is None:
        import sam_infer
        predictor_factory = sam_infer.make_predictor
    app = BatchCrackSegmentation(data_dir, predictor_factory(args), Config(),
                                 tag=model_tag(args.lora_weights, args.checkpoint), prob_only=args.prob_only)
    cfg = app.cfg
    cfg.SAM_THRESHOLD = args.threshold
    if args.uncertain_below is not None:
        cfg.SAM_UNCERTAIN_MEAN_PROB = None if args.uncertain_below <= 0 else args.uncertain_below
    rows = []
    report = os.path.join(data_dir, REPORT_NAME)
    for i, path in enumerate(todo, 1):
        print(f"\n[SAM batch] {i}/{len(todo)}  {os.path.basename(path)}")
        try:
            row = app.process(path, args.existing == "update")
        except Exception as e:   # one bad photo must not stop the other 599
            row = {"photo": os.path.basename(path), "status": "error", "sam_cracks": 0, "uncertain": 0,
                   "operator_cracks_kept": 0, "uncertainty": "", "seconds": 0, "message": str(e)}
            print(f"[SAM batch] ERROR on {os.path.basename(path)}: {e}", file=sys.stderr)
        rows.append(row)
        write_report(report, rows)   # rewritten after every photo: usable even if the job is stopped
    if args.prob_only:
        ranked = sorted((r for r in rows if r["status"] == "map"), key=lambda r: -r["uncertainty"])
        print(f"\n[SAM batch] Crack maps of {len(ranked)} photo(s) in '<data folder>/SAM model/{PROBABILITY_DIR}'. "
              f"Most uncertain first (annotate these first): {', '.join(r['photo'] for r in ranked[:10])}")
        return rows
    done = [r for r in rows if r["status"] in ("segmented", "updated")]
    errors = [r for r in rows if r["status"] == "error"]
    print(f"\n[SAM batch] Done: {len(done)} photo(s) segmented, {sum(r['sam_cracks'] for r in done)} crack(s), "
          f"{len(errors)} error(s). Report: {report}")
    return rows


def parse_args(argv=None):
    import sam_infer
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True,
                   help="CrackSegmentation data folder (with 'Images'; 'JSON files' and 'Binary files' are written)")
    p.add_argument("--existing", choices=["skip", "update"], default="skip",
                   help="photos that already have a JSON: skip them (default) or add SAM cracks to them")
    p.add_argument("--limit", type=int, default=0, help="process at most N photos (e.g. a pilot of 30)")
    p.add_argument("--only", nargs="*", default=None, help="process only these photos (file names)")
    p.add_argument("--prob-only", action="store_true",
                   help="only compute the crack maps (no JSON): [SAM click to accept] then works instantly in the app; "
                        "the report ranks the photos by uncertainty (active learning)")
    p.add_argument("--threshold", type=float, default=0.5, help="probability that makes a pixel crack")
    p.add_argument("--uncertain-below", type=float, default=None,
                   help="cracks with a lower mean probability are saved as uncertain (default 0.6; 0 = never)")
    sam_infer.add_model_args(p)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
