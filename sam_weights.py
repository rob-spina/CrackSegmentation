"""sam_weights.py -- v2.0.0 automatic download of the SAM crack model's weights (sam_autoseg.py).

Two files, saved in '<data folder>/SAM model' the first time [SAM auto-segment] is pressed:
  - the base SAM ViT-H weights, published by Meta (Apache 2.0);
  - the crack LoRA + mask-decoder weights trained on Facade390 and CrackSeg9K (sam_weights/MODEL_CARD.md),
    published as an asset of a GitHub Release of CrackSegmentation.
Both are checked (size and/or hash) before use. An interrupted download resumes from where it stopped when the
server supports byte ranges (both do). Pure standard library: no torch, no requests.
"""
import hashlib
import os
import ssl
import urllib.error
import urllib.request

CHUNK = 1 << 20   # 1 MiB


class DownloadCancelled(Exception):
    pass


class DownloadError(Exception):
    pass


def _ssl_context():
    """certifi's CA bundle when it is available: the Python bundled in the macOS app may have no system CAs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _open(url, start):
    request = urllib.request.Request(url, headers={"User-Agent": "CrackSegmentation"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    context = _ssl_context() if url.lower().startswith("https:") else None
    return urllib.request.urlopen(request, timeout=60, context=context)


def file_digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, spec):
    """None if the file at path matches spec (size, md5 prefix, sha256 -- each checked only when given),
    otherwise why not."""
    if not os.path.isfile(path):
        return "file not found"
    size = spec.get("size")
    if size and os.path.getsize(path) != size:
        return f"size {os.path.getsize(path)} bytes instead of {size}"
    if spec.get("md5_prefix") and not file_digest(path, "md5").startswith(spec["md5_prefix"].lower()):
        return "MD5 checksum does not match"
    if spec.get("sha256") and file_digest(path, "sha256") != spec["sha256"].lower():
        return "SHA-256 checksum does not match"
    return None


def download(spec, dest, progress=None, cancelled=None):
    """Downloads spec['url'] to dest through dest + '.part' (kept on interruption, resumed next time), then
    verifies it. progress(done_bytes, total_bytes_or_None) is called while downloading; cancelled() returning
    True stops it (DownloadCancelled). Raises DownloadError on a network or verification problem."""
    if verify(dest, spec) is None:
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    start = os.path.getsize(part) if os.path.isfile(part) else 0
    if spec.get("size") and start > spec["size"]:
        os.remove(part)
        start = 0
    try:
        response = _open(spec["url"], start)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 416 and start:   # nothing left to fetch: the .part may already be complete
            response = None
        else:
            raise DownloadError(f"{spec['url']}: HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, OSError) as e:
        raise DownloadError(f"{spec['url']}: {getattr(e, 'reason', e)}") from e
    if response is not None:
        with response:
            resumed = start and getattr(response, "status", 200) == 206
            if not resumed:
                start = 0   # the server sent the whole file again
            length = response.headers.get("Content-Length")
            total = start + int(length) if length and length.isdigit() else spec.get("size")
            done = start
            try:
                with open(part, "ab" if resumed else "wb") as out:
                    while True:
                        if cancelled and cancelled():
                            raise DownloadCancelled()
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
            except (urllib.error.URLError, OSError) as e:
                raise DownloadError(f"{spec['url']}: download interrupted ({e}); press the button again to resume") from e
    problem = verify(part, spec)
    if problem:
        os.remove(part)
        raise DownloadError(f"{os.path.basename(dest)}: {problem}; the file was removed, press the button again")
    os.replace(part, dest)
    return dest


def human_size(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024.0


# v2.0.1 [x2 tile upscale]: a model trained with train.py --upscale U must be run with the same U. train.py writes
# its options in train_args.json next to best_lora.pth (the Leonardo round script copies it into round_NN/).
TRAIN_ARGS_FILE = "train_args.json"


def lora_upscale(lora_weights):
    """train.py --upscale of these LoRA weights (train_args.json in the same folder), 1.0 when unknown."""
    import json
    path = os.path.join(os.path.dirname(os.path.expanduser(lora_weights or "")), TRAIN_ARGS_FILE)
    try:
        with open(path) as f:
            value = float(json.load(f).get("upscale", 1.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 1.0
    return value if value > 0 else 1.0


def resolve_upscale(override, lora_weights):
    """The upscale to run these weights with: the setting when given (not None / 0), else the weights' own."""
    try:
        if override is not None and float(override) > 0:
            return float(override)
    except (TypeError, ValueError):
        pass
    return lora_upscale(lora_weights) if lora_weights else 1.0


def upscale_suffix(upscale):
    """'' for 1, '-x2' for 2, '-x1.5' for 1.5: appended to the maps folder name so maps of the same weights at
    different upscales never mix."""
    return "" if abs(float(upscale) - 1.0) < 1e-6 else f"-x{float(upscale):g}"
