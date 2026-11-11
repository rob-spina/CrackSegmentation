"""leonardo_link.py
v2.0.1 [Leonardo] (left sidebar "Dataset" section, Tools menu, code 154), mixed into CrackSegmentation.
Runs an active-learning round on Leonardo (CINECA) from the app, with the operator's own account:

  Connect               checks the SSH certificate; when missing or expired runs
                        `step ssh login <e-mail> --provisioner cineca-hpc` (the browser opens for the CINECA login with
                        two factors; the certificate then lasts 12 hours)
  Send data + start     copies the dataset folders (Images, JSON files, Binary files, Training / Validation / Test
                        set) to the remote data folder with rsync (retried, the login node may cut long transfers),
                        copies this computer's code there (code_sync = rsync: the *.py files, cracksam/, docs/ --
                        no git push needed; code_sync = git: `git pull`) and submits
                        docs/sam_model/active_learning_round_leonardo.sbatch with ROUND / PREV / DATA_DIR / CODE_DIR /
                        LORA_DIR / EPOCHS / UPSCALE (job = maps: MAPS_ONLY=1, the maps of every photo with PREV, no
                        training); the job id is remembered
  Job status            squeue / sacct of that job
  Fetch results         copies back 'SAM model/rounds' and 'SAM model/probability maps' -- then [Active learning]

The fields (user, e-mail, hosts, folders, round, previous weights ...) are shown in a window and, if wished, kept in
~/.crack_segmentation/leonardo.json on this computer (outside the data folder, so never committed with the code).
No password is ever stored or typed here: the login happens in the browser.
"""
import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections import deque

LEONARDO_CODE = 154
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".crack_segmentation", "leonardo.json")
SYNC_FOLDERS = ("Images", "JSON files", "Binary files", "Training set", "Validation set", "Test set")
FETCH_FOLDERS = ("SAM model/rounds", "SAM model/probability maps")
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # the code copied to Leonardo with code_sync = rsync
ROUND_SCRIPT = "docs/sam_model/active_learning_round_leonardo.sbatch"
EXTRA_PATH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")   # step / rsync when started from Finder

# (key, label, default). The order is the order of the window.
FIELDS = [
    ("user", "Leonardo username", ""),
    ("email", "E-mail for 'step ssh login'", ""),
    ("host", "Login host", "login.leonardo.cineca.it"),
    ("transfer_host", "Host for copying data (login or data.leonardo.cineca.it)", "login.leonardo.cineca.it"),
    ("provisioner", "step provisioner", "cineca-hpc"),
    ("local_data", "Local data folder (source)", ""),
    ("remote_data", "Remote data folder (destination)", "/leonardo_work/IscrC_AID-ECHO/drone_4k"),
    ("remote_code", "Remote CrackSegmentation folder", "/leonardo_work/IscrC_AID-ECHO/CrackSegmentation"),
    ("remote_lora", "Remote cracksam_lora folder", "/leonardo_work/IscrC_AID-ECHO/cracksam_lora"),
    ("round", "Round to train (ROUND)", "1"),
    ("prev", "Starting weights (PREV best_lora.pth)",
     "/leonardo_work/IscrC_AID-ECHO/cracksam_lora/runs/facade390_vit_h_fixed/best_lora.pth"),
    ("epochs", "Epochs", "20"),
    ("upscale", "Tile upscale before SAM (UPSCALE: 2 = thin cracks x2, 1 = off)", "2"),
    ("job_kind", "Job: round (train + maps) or maps (maps of every photo with the starting weights)", "round"),
    ("code_sync", "Code on Leonardo: rsync (copied from this computer) or git (git pull)", "rsync"),
]
FIELD_KEYS = [k for k, _, _ in FIELDS]


def default_settings(local_data=""):
    s = {k: d for k, _, d in FIELDS}
    s["local_data"] = local_data
    s["job_id"] = ""
    return s


def load_settings(local_data="", path=SETTINGS_PATH):
    s = default_settings(local_data)
    try:
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            s.update({k: str(v) for k, v in stored.items() if k in s})
    except (OSError, ValueError):
        pass
    if not s.get("local_data"):
        s["local_data"] = local_data
    return s


def save_settings(settings, path=SETTINGS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = {k: settings.get(k, "") for k in FIELD_KEYS + ["job_id"]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def missing_fields(settings, keys):
    return [label for k, label, _ in FIELDS if k in keys and not str(settings.get(k, "")).strip()]


def remote_path(path):
    """A remote path for rsync: spaces escaped for the remote shell ('SAM model' -> 'SAM\\ model')."""
    return path.replace(" ", "\\ ")


def ssh_base(s, host_key="host"):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", f"{s['user']}@{s[host_key]}"]


def check_command(s):
    return ssh_base(s) + ["echo crackseg-ok"]


def login_command(s):
    return ["step", "ssh", "login", s["email"], "--provisioner", s["provisioner"]]


def mkdir_command(s):
    return ssh_base(s) + [f"mkdir -p {shlex.quote(s['remote_data'])}"]


def upload_command(s):
    sources = [os.path.join(s["local_data"], d) for d in SYNC_FOLDERS
               if os.path.isdir(os.path.join(s["local_data"], d))]
    return (["rsync", "-av", "-e", "ssh -o BatchMode=yes", "--exclude", ".DS_Store"] + sources
            + [f"{s['user']}@{s['transfer_host']}:{remote_path(s['remote_data'].rstrip('/'))}/"])


def code_sync_mode(s):
    return "git" if str(s.get("code_sync") or "rsync").strip().lower() == "git" else "rsync"


def maps_only(s):
    return str(s.get("job_kind") or "round").strip().lower().startswith("map")


def code_upload_command(s, local_code=None):
    """rsync of the code only (the *.py files, cracksam/ and docs/) to the remote CrackSegmentation folder: no git
    push / pull needed. The data folders, 'SAM model', packaging/ ... are not copied."""
    src = os.path.join(local_code or APP_DIR, "")
    return (["rsync", "-av", "-e", "ssh -o BatchMode=yes", "--exclude", "__pycache__/", "--exclude", ".DS_Store",
             "--include", "cracksam/***", "--include", "docs/***", "--include", "*.py", "--exclude", "*", src,
             f"{s['user']}@{s['transfer_host']}:{remote_path(s['remote_code'].rstrip('/'))}/"])


def start_command(s):
    env = (f"ROUND={shlex.quote(str(s['round']))} PREV={shlex.quote(s['prev'])} "
           f"DATA_DIR={shlex.quote(s['remote_data'])} CODE_DIR={shlex.quote(s['remote_code'])} "
           f"LORA_DIR={shlex.quote(s['remote_lora'])} EPOCHS={shlex.quote(str(s['epochs']))} "
           f"UPSCALE={shlex.quote(str(s.get('upscale') or '2'))} MAPS_ONLY={1 if maps_only(s) else 0}")
    pull = "git pull --ff-only && " if code_sync_mode(s) == "git" else ""
    script = f"cd {shlex.quote(s['remote_code'])} && {pull}{env} sbatch {ROUND_SCRIPT}"
    return ssh_base(s) + [script]


def status_command(s):
    job = shlex.quote(s["job_id"])
    return ssh_base(s) + [f"squeue -h -j {job} -o '%T %M' 2>/dev/null || true; "
                          f"sacct -n -X -j {job} -o State,Elapsed 2>/dev/null | head -1"]


def fetch_commands(s):
    cmds = []
    for sub in FETCH_FOLDERS:
        local = os.path.join(s["local_data"], sub)
        src = f"{s['user']}@{s['transfer_host']}:{remote_path(s['remote_data'].rstrip('/') + '/' + sub)}/"
        cmds.append((local, ["rsync", "-av", "-e", "ssh -o BatchMode=yes", src, local + "/"]))
    return cmds


def parse_job_id(output):
    m = re.search(r"Submitted batch job (\d+)", output or "")
    return m.group(1) if m else ""


class LeonardoLinkMixin:

    # --- Running a command ---------------------------------------------------------------------

    def _leonardo_env(self):
        env = dict(os.environ)
        parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(parts + [p for p in EXTRA_PATH if p not in parts])
        return env

    def _run_command(self, cmd, timeout=None):
        """Runs cmd, relaying its lines (status bar / terminal) and keeping the window alive; [Esc] stops it.
        Returns (exit code or None if stopped / not started, output text)."""
        print(" [LEONARDO] $ " + " ".join(shlex.quote(c) for c in cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=self._leonardo_env())
        except OSError as e:
            print(f" [LEONARDO] Cannot run {cmd[0]}: {e}")
            return None, str(e)
        lines, out = deque(), []
        reader = threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True)
        reader.start()
        start = time.time()
        with proc:
            while True:
                while lines:
                    text = lines.popleft().rstrip()
                    out.append(text)
                    if text:
                        print(f" [LEONARDO] {text}")
                stop = self._sam_cancel_requested() or (timeout and time.time() - start > timeout)
                if proc.poll() is None and stop:
                    proc.kill()
                    proc.wait()
                    reader.join(1)
                    return None, "\n".join(out)
                if proc.poll() is not None and not reader.is_alive() and not lines:
                    break
                self._pump_extra_events()
                time.sleep(0.05)
        return proc.returncode, "\n".join(out)

    # --- Actions (each returns (ok, message)) ---------------------------------------------------

    def leonardo_connect(self, s):
        missing = missing_fields(s, ("user", "host"))
        if missing:
            return False, "Fill in: " + ", ".join(missing)
        code, out = self._run_command(check_command(s), timeout=60)
        if code == 0 and "crackseg-ok" in out:
            return True, f"Connected to {s['host']} as {s['user']} (certificate valid)."
        missing = missing_fields(s, ("email", "provisioner"))
        if missing:
            return False, "No valid certificate: fill in " + ", ".join(missing) + " to log in."
        print(" [LEONARDO] No valid certificate: logging in -- complete the CINECA login in the browser.")
        code, out = self._run_command(login_command(s), timeout=600)
        if code != 0:
            hint = " Is the 'step' CLI installed (brew install step)?" if code is None and "No such file" in out \
                else ""
            return False, "Login not completed." + hint
        code, out = self._run_command(check_command(s), timeout=60)
        if code == 0 and "crackseg-ok" in out:
            return True, f"Logged in: connected to {s['host']} as {s['user']} (certificate valid 12 h)."
        return False, "Logged in, but the connection to Leonardo still fails: see the lines above."

    def leonardo_upload(self, s, attempts=5):
        missing = missing_fields(s, ("local_data", "remote_data", "transfer_host"))
        if missing:
            return False, "Fill in: " + ", ".join(missing)
        if not any(os.path.isdir(os.path.join(s["local_data"], d)) for d in SYNC_FOLDERS):
            return False, f"No dataset folder (Images, Training set ...) in {s['local_data']}."
        code, _ = self._run_command(mkdir_command(s), timeout=120)
        if code != 0:
            return False, "Could not create the remote data folder."
        for attempt in range(1, attempts + 1):
            code, _ = self._run_command(upload_command(s))
            if code == 0:
                if code_sync_mode(s) == "rsync":
                    ok, message = self.leonardo_upload_code(s)
                    if not ok:
                        return False, message
                    return True, "Dataset and code copied to Leonardo."
                return True, "Dataset copied to Leonardo."
            if code is None:
                return False, "Copy stopped."
            print(f" [LEONARDO] Copy interrupted (attempt {attempt}/{attempts}): resuming ...")
        return False, "The copy kept failing: try the data host data.leonardo.cineca.it."

    def leonardo_upload_code(self, s, attempts=3):
        """code_sync = rsync: the app's code (this folder) to the remote CrackSegmentation folder."""
        missing = missing_fields(s, ("remote_code", "transfer_host"))
        if missing:
            return False, "Fill in: " + ", ".join(missing)
        self._run_command(ssh_base(s) + [f"mkdir -p {shlex.quote(s['remote_code'])}"], timeout=120)
        for _ in range(attempts):
            code, _ = self._run_command(code_upload_command(s))
            if code == 0:
                return True, "Code copied to Leonardo."
            if code is None:
                return False, "Copy stopped."
        return False, "The code could not be copied: see the lines above."

    def leonardo_start_round(self, s):
        keys = ("remote_code", "remote_lora", "prev") if maps_only(s) else \
            ("remote_code", "remote_lora", "round", "prev", "epochs")
        missing = missing_fields(s, keys)
        if missing:
            return False, "Fill in: " + ", ".join(missing)
        code, out = self._run_command(start_command(s), timeout=300)
        job = parse_job_id(out)
        if code != 0 or not job:
            return False, "The job was not submitted: see the lines above."
        s["job_id"] = job
        what = f"Maps x{s.get('upscale') or 2} of every photo" if maps_only(s) else f"Round {s['round']}"
        return True, f"{what} submitted: job {job}. 'Job status' tells when it ends, then 'Fetch results'."

    def leonardo_send_and_start(self, s):
        for step in (self.leonardo_connect, self.leonardo_upload, self.leonardo_start_round):
            ok, message = step(s)
            print(f" [LEONARDO] {message}")
            if not ok:
                return False, message
        return True, message

    def leonardo_status(self, s):
        if not s.get("job_id"):
            return False, "No job submitted from here yet."
        code, out = self._run_command(status_command(s), timeout=60)
        if code != 0:
            return False, "Could not read the job status (connect first)."
        text = " ".join(line.strip() for line in out.splitlines() if line.strip()) or "unknown"
        return True, f"Job {s['job_id']}: {text}"

    def leonardo_fetch(self, s):
        missing = missing_fields(s, ("local_data", "remote_data"))
        if missing:
            return False, "Fill in: " + ", ".join(missing)
        for local, cmd in fetch_commands(s):
            os.makedirs(local, exist_ok=True)
            code, _ = self._run_command(cmd)
            if code != 0:
                return False, f"Could not copy back {os.path.relpath(local, s['local_data'])} (connect first)."
        return True, "New weights and crack maps copied back: now press Active learning."

    # --- The window ----------------------------------------------------------------------------

    def _leonardo_window(self, settings):
        """Terminal version: a numbered menu. The GUI shows a window with the fields and the action buttons."""
        while True:
            print("\n[LEONARDO] " + ", ".join(f"{k}={settings.get(k, '')}" for k in FIELD_KEYS))
            print("  1) connect   2) send data + start round   3) job status   4) fetch results   ENTER) close")
            try:
                answer = input("Choice: ").strip()
            except Exception:
                return
            action = {"1": self.leonardo_connect, "2": self.leonardo_send_and_start, "3": self.leonardo_status,
                      "4": self.leonardo_fetch}.get(answer)
            if action is None:
                return
            print(" [LEONARDO] " + action(settings)[1])
            save_settings(settings)

    def run_leonardo(self):
        """Code 154."""
        settings = load_settings(str(self.cfg.SCRIPT_DIR))
        self._leonardo_window(settings)
        return False
