"""sam_autoseg.py
v2.0.0 [SAM auto-segment] (right sidebar, Tools menu, code 136), mixed into CrackSegmentation (smart_segmentation.py).
Runs the fine-tuned SAM crack model (non-promptable) on the photo on screen and turns its mask into real cracks
of the app, then writes the JSON ('JSON files') and the binary masks ('Binary files') exactly as [S] does.

  1. sam_infer.py runs in the Python chosen in [SAM model settings] (code 137, or Ctrl/Cmd+click on the button),
     which has PyTorch: the app itself never imports torch. Progress goes to the status bar; [Esc] cancels.
     The model is the SAM + LoRA of the cracksam_lora project (base SAM weights + best_lora.pth, with that
     project's model.py / tiling.py, bundled in cracksam/, so inference matches its infer.py), or any single
     checkpoint with the cv_domain_adapt.py interface. The first time, the default model (Meta's SAM ViT-H + the
     crack weights of the GitHub Release, sam_weights.py) is downloaded into '<data folder>/SAM model'.
  2. The probability map is thresholded (SAM_THRESHOLD); small blobs are dropped; pixels already covered by the
     operator's cracks or inside ignore regions are removed, so nothing is segmented twice.
  3. Each blob is skeletonised; its longest centre line becomes a crack, then the longest line of what is left
     (branches), until the lines get shorter than SAM_MIN_CRACK_LENGTH_PX. Each blob pixel goes to the nearest
     line, and that exact shape is kept as the crack's width (an auto-filled width_segments tract), so the exported
     crack mask is SAM's mask, not a dilated centre line.
  4. Cracks whose mean probability is below SAM_UNCERTAIN_MEAN_PROB are saved as uncertain (red).
Every SAM crack has "source": "sam" in the JSON and stays editable (Cut, Join, Eraser, Uncertain ...). Running the
button again replaces the previous SAM cracks of the photo (after a confirmation); the operator's cracks stay.
[U] undoes the whole run in one step.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

import cv2
import numpy as np
from skimage.morphology import skeletonize

import sam_weights

SAM_AUTOSEG_CODE = 136   # 133 XML reference, 134/135 Merge crack branches (v1.5.0)
SAM_SETTINGS_CODE = 137
SAM_SOURCE = "sam"

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ======================================================================================
# Mask -> cracks (numpy / OpenCV only, no torch)
# ======================================================================================

def probability_to_mask(prob_u8, threshold, min_area_px):
    """255 where prob >= threshold, without connected blobs smaller than min_area_px."""
    mask = (prob_u8.astype(np.float32) >= threshold * 255.0).astype(np.uint8) * 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area_px
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def _bfs_farthest(start, alive):
    """Farthest skeleton pixel from start (8-connected steps) and the predecessor map of the search."""
    h, w = alive.shape
    prev = {start: None}
    queue = deque([start])
    last = start
    while queue:
        y, x = last = queue.popleft()
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and alive[ny, nx] and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                queue.append((ny, nx))
    return last, prev


def _longest_line(start, alive):
    """Longest geodesic line through the skeleton part containing start (two searches), as [(y, x), ...],
    plus every pixel of that part."""
    a, _ = _bfs_farthest(start, alive)
    b, prev = _bfs_farthest(a, alive)
    line, node = [], b
    while node is not None:
        line.append(node)
        node = prev[node]
    return line, list(prev.keys())


def skeleton_lines(mask, min_length_px, max_lines=10000):
    """Centre lines of a binary mask: the longest line of each skeleton part, then of the branches left, until
    lines get shorter than min_length_px. Returns [[(x, y), ...], ...], longest first within each part."""
    alive = skeletonize(mask > 0)
    lines = []
    ys, xs = np.nonzero(alive)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not alive[y, x]:
            continue
        pending = [(y, x)]
        while pending and len(lines) < max_lines:
            seed = pending.pop()
            if not alive[seed]:
                continue
            line, part = _longest_line(seed, alive)
            if len(line) < min_length_px:
                for p in part:
                    alive[p] = False
                continue
            for p in line:
                alive[p] = False
            lines.append([(int(px), int(py)) for py, px in line])
            pending.extend(p for p in part if alive[p])
    return lines


def assign_pixels_to_lines(mask, lines):
    """For each line, the mask pixels nearest to it: list of uint8 masks (mask's size). Meant for ONE blob."""
    h, w = mask.shape
    seeds = np.full((h, w), 255, dtype=np.uint8)
    line_of = np.full((h, w), -1, dtype=np.int32)
    for i, line in enumerate(lines):
        pts = np.asarray(line, dtype=np.int64)
        seeds[pts[:, 1], pts[:, 0]] = 0
        line_of[pts[:, 1], pts[:, 0]] = i
    _, nearest = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    zero = seeds == 0
    lut = np.full(int(nearest.max()) + 1, -1, dtype=np.int32)
    lut[nearest[zero]] = line_of[zero]
    owner = np.where(mask > 0, lut[nearest], -1)
    regions = []
    for i, line in enumerate(lines):
        region = (owner == i).astype(np.uint8) * 255
        pts = np.asarray(line, dtype=np.int64)
        region[pts[:, 1], pts[:, 0]] = 255
        regions.append(region)
    return regions


def mask_to_crack_shapes(mask, min_length_px, max_cracks):
    """[(line [(x, y), ...], region, x0, y0), ...] for every crack found in the mask, longest first. Each blob is
    handled in its own bounding box: region is that box's mask (crack pixels = 255), placed at (x0, y0)."""
    shapes = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x0, y0, bw, bh = (int(v) for v in stats[k, :4])
        blob = (labels[y0:y0 + bh, x0:x0 + bw] == k).astype(np.uint8) * 255
        blob = np.pad(blob, 1)   # skeletonize needs a background border
        lines = skeleton_lines(blob, min_length_px)
        for line, region in zip(lines, assign_pixels_to_lines(blob, lines)):
            shapes.append(([(x + x0 - 1, y + y0 - 1) for x, y in line], region[1:-1, 1:-1], x0, y0))
    shapes.sort(key=lambda s: len(s[0]), reverse=True)
    return shapes[:max_cracks]


# ======================================================================================
# Probability-map cache: '<data folder>/SAM model/probability maps/<model>/<photo>.png'
# ======================================================================================

PROBABILITY_DIR = "probability maps"


_TAG_CACHE = {}


def model_tag(lora_weights="", checkpoint=""):
    """Folder name of one model's maps. For LoRA weights, their SHA-256 (the same weights give the same folder on the
    Mac and on Leonardo, whatever the file is called); for a single checkpoint, its file name."""
    path = os.path.expanduser(lora_weights or "")
    if path and os.path.isfile(path):
        key = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
        if key not in _TAG_CACHE:
            _TAG_CACHE[key] = "lora-" + sam_weights.file_digest(path, "sha256")[:12]
        return _TAG_CACHE[key]
    name = os.path.splitext(os.path.basename(lora_weights or checkpoint or "model"))[0]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "model"


def probability_cache_path(model_dir, tag, photo_path):
    return os.path.join(model_dir, PROBABILITY_DIR, tag, os.path.splitext(os.path.basename(photo_path))[0] + ".png")


def read_cached_probability(cache_path, photo_path, shape):
    """The cached map (uint8), or None when missing, older than the photo or of another size."""
    try:
        if os.path.getmtime(cache_path) < os.path.getmtime(photo_path):
            return None
    except OSError:
        return None
    prob = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
    return prob if prob is not None and prob.shape == tuple(shape) else None


def write_cached_probability(cache_path, prob_u8):
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cv2.imwrite(cache_path, prob_u8)
    except (OSError, cv2.error):
        pass   # a cache only: never blocks the segmentation


def probability_uncertainty(prob_u8, low=0.3, high=0.7):
    """Share of the crack-like pixels (p >= low) whose probability is in [low, high]: high = the model hesitates.
    Used to pick the photos worth annotating first (active learning)."""
    candidate = prob_u8 >= low * 255
    n = int(candidate.sum())
    if not n:
        return 0.0
    return round(float((candidate & (prob_u8 <= high * 255)).sum()) / n, 3)


# ======================================================================================
# The button
# ======================================================================================

class SamAutoSegMixin:
    """SAM automatic segmentation. Relies on CrackSegmentation's cfg, mask, undo and save helpers."""

    # --- Model settings (sam_model.json in the data folder) -------------------------------

    def _sam_settings_path(self):
        return os.path.join(str(self.cfg.SCRIPT_DIR), self.cfg.SAM_SETTINGS_FILE)

    def _load_sam_settings(self):
        try:
            with open(self._sam_settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_sam_settings(self, settings):
        with open(self._sam_settings_path(), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

    @staticmethod
    def _default_sam_python():
        """The Python running the app when it runs from source (it may have torch); none for the installed app."""
        return "" if getattr(sys, "frozen", False) else sys.executable

    @staticmethod
    def _sam_backend(settings):
        """'sam_lora' (base SAM + best_lora.pth of the cracksam_lora project) or 'module' (one checkpoint)."""
        return settings.get("backend") or ("module" if settings.get("checkpoint") else "sam_lora")

    @staticmethod
    def _guess_sam_model_type(path, default="vit_h"):
        """vit_b / vit_l / vit_h from a base SAM file name such as sam_vit_h_4b8939.pth."""
        name = os.path.basename(path or "").lower()
        return next((t for t in ("vit_b", "vit_l", "vit_h") if t in name), default)

    @staticmethod
    def _missing_file(settings, key, what):
        path = os.path.expanduser(settings.get(key) or "")
        return None if path and os.path.isfile(path) else f"{what} is not set or not found"

    @classmethod
    def _sam_settings_problem(cls, settings):
        """Why these settings cannot run (None = they look usable)."""
        if cls._sam_backend(settings) == "sam_lora":
            problem = (cls._missing_file(settings, "lora_weights", "the LoRA weights (best_lora.pth)")
                       or cls._missing_file(settings, "sam_checkpoint", "the base SAM weights (sam_vit_*.pth)"))
            if problem:
                return problem
            code_dir = os.path.expanduser(settings.get("code_dir") or "")
            if not os.path.isfile(os.path.join(code_dir, "model.py")):
                return "the cracksam_lora code folder (with model.py, sam_lora.py, tiling.py) is not set or not found"
            if settings.get("model_type") not in ("vit_b", "vit_l", "vit_h"):
                return "the SAM model type must be vit_b, vit_l or vit_h"
        else:
            problem = cls._missing_file(settings, "checkpoint", "the model checkpoint (.pt / .pth)")
            if problem:
                return problem
            factory_file = os.path.expanduser((settings.get("model_factory") or "").partition(":")[0])
            if factory_file and not os.path.isfile(factory_file):
                return f"the model factory file is not found: {factory_file}"
        return cls._missing_file(settings, "python", "the Python interpreter with PyTorch")

    def _choose_sam_settings(self, current):
        """Asks which model to run and its files. Returns the new settings, or None if cancelled.
        Default: terminal input(). A GUI wrapper overrides this with file dialogs."""
        def ask(label, default):
            try:
                answer = input(f"{label} [{default}]: ").strip()
            except EOFError:
                return None
            return answer or default
        backend = ask("Model kind: sam_lora (base SAM + best_lora.pth) or module (one checkpoint)",
                      self._sam_backend(current))
        if backend not in ("sam_lora", "module"):
            return None
        if backend == "sam_lora":
            answers = [ask("LoRA weights (best_lora.pth)", current.get("lora_weights", "")),
                       ask("Base SAM weights (sam_vit_h_4b8939.pth ...)", current.get("sam_checkpoint", ""))]
            if None in answers:
                return None
            answers += [ask("SAM model type (vit_b / vit_l / vit_h)",
                            current.get("model_type") or self._guess_sam_model_type(answers[1])),
                        ask("LoRA rank used in training", str(current.get("lora_r", 4))),
                        ask("cracksam_lora code folder (model.py, sam_lora.py, tiling.py)",
                            current.get("code_dir") or self._bundled_cracksam_dir())]
            python = ask("Python interpreter with PyTorch and segment-anything",
                         current.get("python") or self._default_sam_python())
            if None in answers or python is None:
                return None
            try:
                lora_r = int(answers[3])
            except ValueError:
                return None
            return {"backend": "sam_lora", "lora_weights": answers[0], "sam_checkpoint": answers[1],
                    "model_type": answers[2], "lora_r": lora_r, "code_dir": answers[4], "python": python}
        checkpoint = ask("Model checkpoint (.pt/.pth)", current.get("checkpoint", ""))
        if checkpoint is None:
            return None
        factory = ask("Model factory FILE.py:FUNCTION (empty if the checkpoint is a whole model)",
                      current.get("model_factory", ""))
        python = ask("Python interpreter with PyTorch", current.get("python") or self._default_sam_python())
        if factory is None or python is None:
            return None
        return {"backend": "module", "checkpoint": checkpoint, "model_factory": factory, "python": python}

    def _sam_settings_summary(self, settings):
        if self._sam_backend(settings) == "sam_lora":
            return (f"SAM {settings['model_type']} + LoRA r={settings['lora_r']} "
                    f"({os.path.basename(settings['lora_weights'])}), code {settings['code_dir']}, "
                    f"Python {settings['python']}")
        return (f"{os.path.basename(settings['checkpoint'])}"
                + (f" (factory {settings['model_factory']})" if settings.get("model_factory") else "")
                + f", Python {settings['python']}")

    def _edit_sam_settings(self):
        """Code 137 [SAM model settings]: asks for the model files and stores them. Returns the settings or None."""
        current = self._load_sam_settings()
        chosen = self._choose_sam_settings(current)
        if not chosen:
            print(" [SAM] Model settings unchanged.")
            return None
        settings = dict(current, **chosen)
        problem = self._sam_settings_problem(settings)
        if problem:
            print(f" [SAM] Settings not saved: {problem}.", file=sys.stderr)
            return None
        try:
            self._save_sam_settings(settings)
        except OSError as e:
            print(f" [SAM] Could not save the model settings: {e}", file=sys.stderr)
            return None
        print(f" [SAM] Model: {self._sam_settings_summary(settings)}.")
        return settings

    def _usable_sam_settings(self):
        """Stored settings if usable; otherwise the default model, downloaded on first use (after a confirmation);
        otherwise the operator chooses the files."""
        settings = self._load_sam_settings()
        if settings and not settings.get("python"):
            settings["python"] = self._default_sam_python()
        if not self._sam_settings_problem(settings):
            return settings
        if self.cfg.SAM_AUTO_DOWNLOAD:
            choice = True if self._default_sam_weights_present() else self._confirm_sam_download(self._sam_download_text())
            if choice is None:
                print(" [SAM] Cancelled: nothing changed.")
                return None
            if choice:
                return self._setup_default_sam_model()
        print(" [SAM] Choose the SAM model first (LoRA weights, base SAM weights, code folder, Python with PyTorch).")
        return self._edit_sam_settings()

    # --- Default model: automatic download ---------------------------------------------------

    def _sam_model_dir(self):
        return os.path.join(str(self.cfg.SCRIPT_DIR), self.cfg.SAM_MODEL_DIR)

    def _sam_weight_specs(self):
        """[(spec, destination path)] of the base SAM weights and the crack weights."""
        return [(spec, os.path.join(self._sam_model_dir(), spec["file"]))
                for spec in (self.cfg.SAM_BASE_WEIGHTS, self.cfg.SAM_CRACK_WEIGHTS)]

    def _default_sam_weights_present(self):
        """Both files already downloaded (sizes checked; the hashes were checked when they were downloaded)."""
        return all(sam_weights.verify(dest, {"size": spec.get("size")}) is None and os.path.isfile(dest)
                   for spec, dest in self._sam_weight_specs())

    @staticmethod
    def _bundled_cracksam_dir():
        """cracksam/ (model.py, sam_lora.py, tiling.py): next to this file, or in the installed app's resources."""
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "cracksam")

    def _sam_download_text(self):
        sizes = [spec.get("size") for spec, dest in self._sam_weight_specs() if not os.path.isfile(dest)]
        known = sum(s for s in sizes if s)
        return (f"The SAM crack model is downloaded once (about {sam_weights.human_size(known)}) into\n"
                f"{self._sam_model_dir()}\n\nIt needs a Python with PyTorch and segment-anything to run.")

    def _confirm_sam_download(self, text):
        """True = download the default model, False = choose the files, None = cancel.
        Default: terminal input(). A GUI wrapper overrides this with a dialog."""
        try:
            answer = input(f"{text}\nDownload it now? [Y = download / n = choose the files / c = cancel]: ")
        except EOFError:
            return None
        answer = answer.strip().lower()
        return None if answer in ("c", "cancel") else answer not in ("n", "no")

    def _ask_sam_python(self, current):
        """Path of a Python with PyTorch (the installed app has none of its own). A GUI wrapper uses a dialog."""
        try:
            return input(f"Python interpreter with PyTorch and segment-anything [{current}]: ").strip() or current
        except EOFError:
            return None

    def _download_sam_weights(self):
        """Downloads the missing weights, with progress in the status bar; [Esc] cancels. True when both are there."""
        for spec, dest in self._sam_weight_specs():
            name = spec["file"]
            print(f" [SAM] Downloading {name} from {spec['url'].split('/')[2]} ... ([Esc] cancels, it resumes next time)")
            try:
                self._run_in_background(lambda cancelled, s=spec, d=dest:
                                        sam_weights.download(s, d, self._sam_progress_printer(name), cancelled))
            except sam_weights.DownloadCancelled:
                print(f" [SAM] Download of {name} stopped; press the button again to resume it.")
                return False
            except sam_weights.DownloadError as e:
                print(f" [SAM] Download failed: {e}", file=sys.stderr)
                return False
            print(f" [SAM] {name} ready.")
        return True

    @staticmethod
    def _sam_progress_printer(name):
        last = [0.0]

        def progress(done, total):
            if time.time() - last[0] < 2 and (not total or done < total):
                return
            last[0] = time.time()
            share = f" ({100.0 * done / total:.0f}%)" if total else ""
            of = f" of {sam_weights.human_size(total)}" if total else ""
            print(f" [SAM] {name}: {sam_weights.human_size(done)}{of}{share}")
        return progress

    def _run_in_background(self, work):
        """Runs work(cancelled) in a thread, keeping the window alive; [Esc] sets cancelled. Re-raises its error."""
        result, stop = {}, threading.Event()

        def target():
            try:
                result["value"] = work(stop.is_set)
            except BaseException as e:   # handed back to the main thread
                result["error"] = e
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        while thread.is_alive():
            if self._sam_cancel_requested():
                stop.set()
            self._pump_extra_events()
            thread.join(0.05)
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _setup_default_sam_model(self):
        """Downloads the default model if needed and stores settings pointing at it. None if it did not work."""
        if not self._default_sam_weights_present() and not self._download_sam_weights():
            return None
        (base_spec, base), (_, crack) = self._sam_weight_specs()
        current = self._load_sam_settings()
        python = current.get("python") or self._default_sam_python()
        if not python:
            python = self._ask_sam_python("")
            if not python:
                print(" [SAM] No Python with PyTorch chosen: the model was downloaded but cannot run yet.")
                return None
        settings = {"backend": "sam_lora", "lora_weights": crack, "sam_checkpoint": base,
                    "model_type": self.cfg.SAM_DEFAULT_MODEL_TYPE, "lora_r": self.cfg.SAM_DEFAULT_LORA_R,
                    "code_dir": self._bundled_cracksam_dir(), "python": os.path.expanduser(python.strip())}
        problem = self._sam_settings_problem(settings)
        if problem:
            print(f" [SAM] Cannot use the downloaded model: {problem}.", file=sys.stderr)
            return None
        self._save_sam_settings(settings)
        print(f" [SAM] Model ready: {self._sam_settings_summary(settings)}.")
        return settings

    # --- Inference in a separate process ----------------------------------------------------

    @staticmethod
    def _sam_script_path():
        """sam_infer.py: next to this file from source, in the bundle's resource folder in the installed app."""
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "sam_infer.py")

    def _sam_command(self, settings, image_path, out_path):
        cfg = self.cfg
        exp = os.path.expanduser
        cmd = [exp(settings["python"]), "-u", self._sam_script_path(), "--image", image_path, "--out", out_path,
               "--tile", str(cfg.SAM_TILE_PX), "--overlap", str(cfg.SAM_OVERLAP_PX),
               "--batch-size", str(cfg.SAM_BATCH_SIZE), "--device", cfg.SAM_DEVICE]
        if self._sam_backend(settings) == "sam_lora":
            return cmd + ["--backend", "sam_lora", "--code-dir", exp(settings["code_dir"]),
                          "--sam-checkpoint", exp(settings["sam_checkpoint"]), "--model-type", settings["model_type"],
                          "--lora-r", str(settings.get("lora_r", 4)), "--lora-weights", exp(settings["lora_weights"])]
        cmd += ["--backend", "module", "--checkpoint", exp(settings["checkpoint"]), "--scale", str(cfg.SAM_SCALE)]
        if settings.get("model_factory"):
            factory_file, _, func = settings["model_factory"].partition(":")
            cmd += ["--model-factory", exp(factory_file) + (f":{func}" if func else "")]
        return cmd

    def _sam_cancel_requested(self):
        """[Esc] (or closing the window) while SAM runs cancels SAM only; one more press quits as usual."""
        if 27 in self._pending_keys:
            self._pending_keys.remove(27)
            return True
        return False

    def _run_sam_process(self, cmd):
        """Runs sam_infer.py, relaying its lines to the status bar and keeping the window alive.
        Returns the exit code (None if cancelled)."""
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            lines = deque()
            reader = threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True)
            reader.start()
            while proc.poll() is None or lines or reader.is_alive():
                while lines:
                    text = lines.popleft().rstrip()
                    if text:
                        print(f" {text}")
                if proc.poll() is None and self._sam_cancel_requested():
                    proc.kill()
                    proc.wait()
                    reader.join(1)
                    return None
                self._pump_extra_events()
                if proc.poll() is not None and not reader.is_alive() and not lines:
                    break
                time.sleep(0.05)
            return proc.returncode

    def _sam_probability_cache_path(self, settings):
        tag = model_tag(settings.get("lora_weights") if self._sam_backend(settings) == "sam_lora" else "",
                        settings.get("checkpoint", ""))
        return probability_cache_path(self._sam_model_dir(), tag, self.cfg.CURRENT_IMAGE_PATH)

    def _sam_probability_map(self, settings):
        """uint8 probability map (x 255) of the photo on screen, or None (failure / cancelled). Reused from the cache
        when this model already processed this photo (here, by [SAM click to accept] or by sam_batch.py)."""
        cache = self._sam_probability_cache_path(settings)
        prob = read_cached_probability(cache, self.cfg.CURRENT_IMAGE_PATH, (self.cfg.H_img, self.cfg.W_img))
        if prob is not None:
            print(" [SAM] Using the crack map already computed for this photo.")
            return prob
        prob = self._sam_run_model_on_photo(settings)
        if prob is not None:
            write_cached_probability(cache, prob)
        return prob

    def _sam_run_model_on_photo(self, settings):
        with tempfile.TemporaryDirectory(prefix="crackseg_sam_") as tmp:
            out_path = os.path.join(tmp, "probability.png")
            try:
                code = self._run_sam_process(self._sam_command(settings, self.cfg.CURRENT_IMAGE_PATH, out_path))
            except OSError as e:
                print(f" [SAM] Could not start {settings.get('python')}: {e}", file=sys.stderr)
                return None
            if code is None:
                print(" [SAM] Cancelled: nothing changed.")
                return None
            prob = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        if code != 0 or prob is None:
            print(f" [SAM] The model did not run (exit code {code}); see the lines above. "
                  "Are PyTorch and segment-anything installed in that Python? Ctrl/Cmd+click the button to change the model settings.",
                  file=sys.stderr)
            return None
        if prob.shape != (self.cfg.H_img, self.cfg.W_img):
            prob = cv2.resize(prob, (self.cfg.W_img, self.cfg.H_img), interpolation=cv2.INTER_LINEAR)
        return prob

    # --- Mask -> cracks ------------------------------------------------------------------------

    def _sam_existing_coverage(self, keep):
        """Pixels SAM must leave alone: the operator's cracks (with a margin) and the ignore regions."""
        cfg = self.cfg
        covered = self._build_variable_width_crack_mask(keep, cfg.H_img, cfg.W_img, cfg.CRACK_MASK_DILATION_PX)
        margin = int(cfg.SAM_EXISTING_MARGIN_PX)
        if margin > 0 and np.any(covered):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            covered = cv2.dilate(covered, k)
        if cfg.saved_ignores:
            covered = cv2.bitwise_or(covered, self._build_ignore_area_mask())
        return covered

    def _sam_crack_from_shape(self, line, region, x0, y0, prob):
        cfg = self.cfg
        window = prob[y0:y0 + region.shape[0], x0:x0 + region.shape[1]]
        mean_prob = float(window[region > 0].mean()) / 255.0 if np.any(region) else 0.0
        crack = {'start': line[0], 'end': line[-1], 'path': list(line), 'active': True,
                 'session_id': str(time.time()), 'source': SAM_SOURCE, 'sam_mean_prob': round(mean_prob, 3),
                 'width_segments': [{'i0': 0, 'i1': len(line) - 1, 'width_px': 0, 'margin_left_px': 0,
                                     'margin_right_px': 0, 'fill': self._mask_to_rle_rows(region, y0, x0)}]}
        if cfg.SAM_UNCERTAIN_MEAN_PROB is not None and mean_prob < cfg.SAM_UNCERTAIN_MEAN_PROB:
            crack['uncertain_manual'] = True
        return crack

    def _sam_cracks_from_probability(self, prob, keep):
        cfg = self.cfg
        mask = probability_to_mask(prob, cfg.SAM_THRESHOLD, cfg.SAM_MIN_AREA_PX)
        mask[self._sam_existing_coverage(keep) > 0] = 0
        mask = probability_to_mask(mask, 0.5, cfg.SAM_MIN_AREA_PX)   # drops slivers left by the removal
        shapes = mask_to_crack_shapes(mask, cfg.SAM_MIN_CRACK_LENGTH_PX, cfg.SAM_MAX_CRACKS)
        return [self._sam_crack_from_shape(line, region, x0, y0, prob) for line, region, x0, y0 in shapes]

    @staticmethod
    def _is_sam_crack(crack):
        return crack.get('source') == SAM_SOURCE

    def _confirm_sam_replace(self, count):
        """Asks before replacing the SAM cracks of an earlier run. A GUI wrapper overrides this with a dialog."""
        try:
            answer = input(f"Replace the {count} crack(s) of the previous SAM run? [y/N]: ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes", "s", "si")

    def _apply_sam_cracks(self, new_cracks):
        """Replaces the previous SAM cracks with new_cracks (one Undo step), then writes JSON and masks."""
        before = self._snapshot_crack_edit_state()
        kept = [f for f in self.cfg.saved_cracks if not self._is_sam_crack(f)]
        replaced = len(self.cfg.saved_cracks) - len(kept)
        self.cfg.saved_cracks[:] = kept + new_cracks
        self._commit_crack_edit(before)
        n_unc = sum(1 for f in new_cracks if f.get('uncertain_manual'))
        print(f" [SAM] {len(new_cracks)} crack(s) found ({n_unc} uncertain, red)"
              + (f"; {replaced} crack(s) of the previous SAM run replaced" if replaced else "") + ".")
        self.recalculate_masks()
        self._pump_extra_events()
        print(f" [SAM] Writing the JSON and the binary masks (quality scores of {len(self.cfg.saved_cracks)} crack(s)) ...")
        self._save_annotations_now("SAM", "the automatic segmentation")

    def _run_sam_autoseg(self):
        """Code 136 [SAM auto-segment]: segments the photo on screen with SAM, then writes JSON and masks."""
        cfg = self.cfg
        if not cfg.CURRENT_IMAGE_PATH or cfg.img_original is None or getattr(cfg, "queue_idle", False):
            print(" [SAM] No photo on screen.")
            return False
        previous = sum(1 for f in cfg.saved_cracks if self._is_sam_crack(f))
        if previous and not self._confirm_sam_replace(previous):
            print(" [SAM] Cancelled: nothing changed.")
            return False
        settings = self._usable_sam_settings()
        if not settings:
            return False
        print(f" [SAM] Segmenting {os.path.basename(cfg.CURRENT_IMAGE_PATH)} ... ([Esc] cancels)")
        started = time.time()
        prob = self._sam_probability_map(settings)
        if prob is None:
            return False
        keep = [f for f in cfg.saved_cracks if f.get('active', True) and not self._is_sam_crack(f)]
        new_cracks = self._sam_cracks_from_probability(prob, keep)
        self._apply_sam_cracks(new_cracks)
        print(f" [SAM] Done in {time.time() - started:.0f} s. Check the cracks, edit them if needed, [U] undoes the run.")
        self.refresh_zoom_viewport()
        return True
