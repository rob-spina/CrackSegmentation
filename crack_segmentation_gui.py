"""crack_segmentation_gui.py: single-window Tk GUI for the crack/detachment segmentation tool (see smart_segmentation.py) -- reuses render_scene()/mouse_callback() unchanged via display/input hooks. Run with: python3 crack_segmentation_gui.py
"""
import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk

from config import Config
from smart_segmentation import CrackSegmentation
from go_to_image import PagedChooser
from crack_relabel import MARK_UNCERTAIN_CODE, MARK_NORMAL_CODE


# Keyboard translation: Tk <KeyPress> event -> the same integer code
# cv2.waitKey() would have produced for the same physical keypress.

# Mapped explicitly by keysym (not event.char, unreliable for non-printable
# keys across platforms) for arrows, Enter, Backspace, Delete, and Escape.
_SPECIAL_KEYSYM_TO_CODE = {
    "Up": 0, "Down": 1, "Left": 2, "Right": 3,
    "Return": 13, "KP_Enter": 13,
    "BackSpace": 8,
    "Delete": 127,
    "Escape": 27,
}


# v1.3.0: Ctrl (Cmd on macOS) + key -> command code. C = make a blue crack uncertain, U = make a red crack normal.
_CTRL_KEYSYM_TO_CODE = {"c": MARK_UNCERTAIN_CODE, "u": MARK_NORMAL_CODE}


def ctrl_or_command_held(state):
    """Ctrl held, or Cmd (Mod1) on macOS, in a Tk event state."""
    mask = 0x0004 | (0x0008 if sys.platform == "darwin" else 0)
    return bool((state or 0) & mask)


def tk_key_event_to_code(event):
    """Translates one Tk <KeyPress> event into the integer code process_keypress() expects, matching what cv2.waitKey() would return. None if unmapped.
    """
    if event.keysym in _SPECIAL_KEYSYM_TO_CODE:
        return _SPECIAL_KEYSYM_TO_CODE[event.keysym]
    if ctrl_or_command_held(getattr(event, "state", 0)):
        code = _CTRL_KEYSYM_TO_CODE.get((event.keysym or "").lower())
        if code is not None:
            return code
        if event.char and ord(event.char[0]) < 32:
            # Any other Ctrl+letter gives a control character (Ctrl+N = 14 ...) that would fire an unrelated
            # sidebar command: ignored.
            return None
    if event.char:
        return ord(event.char[0]) & 0xFF
    return None


# Embedded canvas: the photo, living inside the main window.
CANVAS_W, CANVAS_H = 1200, 900  # matches transform_*_coords()'s baked-in assumption exactly


class EmbeddedCanvas:
    """Tk Canvas showing the fixed 1200x900 logical content (see CANVAS_W/CANVAS_H, matching transform_*_coords()) inside a vertically-scrollable, horizontally-fixed viewport -- see GuiCrackSegmentation.__init__ for window sizing.
    """

    def _build_canvas_and_scrollbar(self, master):
        """Builds the scrollable canvas + its vertical scrollbar inside
        a container frame."""
        self.container = ttk.Frame(master)
        self.container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.container, bg="black", highlightthickness=0, cursor="crosshair",
                                 scrollregion=(0, 0, CANVAS_W, CANVAS_H))

        v_scroll = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

    def _bind_canvas_events(self):
        """Wires mouse click/motion/wheel handlers onto the canvas."""
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        # Wheel binding at the root level with bind_all() (only while hovering this
        # canvas) is more reliable than binding <MouseWheel> directly on the widget.
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        if sys.platform == "darwin":
            # v1.3.0: some macOS Tk builds turn Ctrl+click into a secondary click; it still means Ctrl+click here.
            for sequence in ("<Control-Button-2>", "<Control-Button-3>"):
                self.canvas.bind(sequence, self._on_press)

    def __init__(self, master, app):
        self.app = app
        self._build_canvas_and_scrollbar(master)

        self._image_id = None
        self._current_photo = None  # keeps a live reference -- Tk drops the image otherwise
        self._content_offset = (0, 0)

        self._bind_canvas_events()

    MIN_LAID_OUT_WIDTH = 200   # below this the canvas is not laid out yet: no scaling
    MAX_DISPLAY_SCALE = 2.0    # v1.2.0: upper bound when the photo grows into the space of hidden sidebars

    def _compute_display_scale(self):
        """v1.0.8: < 1 when the canvas is narrower than the 1200 px frame (e.g. between the two sidebars on a
        laptop screen): the frame is shrunk to the width so no part of the photo is cut off. Height scrolls.
        v1.2.0: > 1 when the canvas is wider and taller than the frame (sidebars hidden): it fills the space."""
        viewport_w = self.canvas.winfo_width()
        if viewport_w < self.MIN_LAID_OUT_WIDTH:
            return 1.0
        if viewport_w < CANVAS_W:
            return viewport_w / float(CANVAS_W)
        # v1.2.0: with a sidebar hidden the photo grows into the freed space, as long as it still fits the height.
        viewport_h = self.canvas.winfo_height()
        return max(1.0, min(viewport_w / float(CANVAS_W), viewport_h / float(CANVAS_H), self.MAX_DISPLAY_SCALE))

    def _compute_content_offset(self, scale=1.0):
        """Offset (x, y) to center the (scaled) 1200x900 photo within a larger viewport. Zero in either axis once the viewport isn't bigger."""
        viewport_w = max(1, self.canvas.winfo_width())
        viewport_h = max(1, self.canvas.winfo_height())
        x_offset = max(0, (viewport_w - int(CANVAS_W * scale)) // 2)
        y_offset = max(0, (viewport_h - int(CANVAS_H * scale)) // 2)
        return x_offset, y_offset

    def _fit_frame(self, bgr_frame):
        """The frame shrunk by the display scale (unchanged when it fits); keeps the scroll region in step."""
        scale = self._compute_display_scale()
        if scale != getattr(self, "_display_scale", 1.0):
            self.canvas.configure(scrollregion=(0, 0, int(CANVAS_W * scale), int(CANVAS_H * scale)))
        self._display_scale = scale
        if scale >= 1.0:
            return bgr_frame
        h, w = bgr_frame.shape[:2]
        return cv2.resize(bgr_frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def update_frame(self, bgr_frame):
        bgr_frame = self._fit_frame(bgr_frame)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_image)
        self._content_offset = self._compute_content_offset(self._display_scale)
        x_offset, y_offset = self._content_offset
        if self._image_id is None:
            self._image_id = self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
        else:
            self.canvas.coords(self._image_id, x_offset, y_offset)
            self.canvas.itemconfig(self._image_id, image=photo)
        self._current_photo = photo

    def _canvas_xy(self, event):
        # Subtracts the same centering offset update_frame() positions the
        # photo with, so a click still maps onto the exact pixel under the cursor.
        # v1.0.8: and divides by the display scale, back to the 1200x900 frame the app works in.
        x_offset, y_offset = getattr(self, "_content_offset", (0, 0))
        scale = getattr(self, "_display_scale", 1.0) or 1.0
        cx = int(round((self.canvas.canvasx(event.x) - x_offset) / scale))
        cy = int(round((self.canvas.canvasy(event.y) - y_offset) / scale))
        return cx, cy

    @staticmethod
    def _mouse_flags(event):
        """Shift held -> cv2.EVENT_FLAG_SHIFTKEY (Shift+click tracing); Ctrl (or Cmd on macOS) held ->
        cv2.EVENT_FLAG_CTRLKEY (v1.3.0: click next to a crack marker without deleting that crack)."""
        state = getattr(event, "state", 0) or 0
        flags = cv2.EVENT_FLAG_SHIFTKEY if state & 0x0001 else 0
        if ctrl_or_command_held(state):   # Control, + Command (Mod1) on macOS
            flags |= cv2.EVENT_FLAG_CTRLKEY
        return flags

    def _on_press(self, event):
        self.canvas.focus_set()
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, self._mouse_flags(event), None)

    def _on_release(self, event):
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, self._mouse_flags(event), None)

    def _on_motion(self, event):
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_MOUSEMOVE, x, y, 0, None)

    _WHEEL_SEQUENCES = ("<MouseWheel>", "<Button-4>", "<Button-5>",
                        "<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>")

    def _on_enter(self, event):
        self.canvas.focus_set()
        # bind_all (not a per-widget bind) makes wheel scrolling reliable across platforms.
        for sequence in self._WHEEL_SEQUENCES:
            self.canvas.bind_all(sequence, self._on_wheel)

    def _on_leave(self, event):
        # v1.3.0: the cursor left the photo, so the [Crack labels] window disappears.
        clear_hover = getattr(self.app, "_clear_crack_label_hover", None)
        if clear_hover is not None:
            clear_hover()
        # Releases the global bindings once the mouse leaves this canvas.
        for sequence in self._WHEEL_SEQUENCES:
            self.canvas.unbind_all(sequence)

    @staticmethod
    def _wheel_units(event):
        """Signed scroll amount, + = up/left. Windows sends multiples of 120 per notch,
        macOS small values per trackpad step, X11 Button-4/5 events."""
        num = getattr(event, "num", None)
        if num == 4:
            return 1.0
        if num == 5:
            return -1.0
        delta = getattr(event, "delta", 0) or 0
        units = delta / 120.0 if abs(delta) >= 120 else float(delta)
        return max(-5.0, min(5.0, units))

    def _pan_zoomed_view(self, event):
        """Pans the zoomed photo by 2% of the view per wheel unit. False when not zoomed in."""
        units = self._wheel_units(event)
        if not units:
            return False
        sideways = bool(getattr(event, "state", 0) & 0x0001)  # Shift held (also macOS horizontal swipes)
        fraction = -0.02 * units
        return self.app.pan_view_by_fraction(fraction if sideways else 0.0, 0.0 if sideways else fraction)

    def _on_wheel(self, event):
        # While the on-screen [?] help guide is open, the wheel scrolls its
        # command list instead, via the same synthetic Up/Down key codes.
        if getattr(self.app.cfg, "show_help_menu", False):
            scrolling_up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
            self.app._pending_keys.append(0 if scrolling_up else 1)
            return "break"
        # Zoomed in: the wheel / two-finger trackpad pans the photo (Shift = sideways).
        if self._pan_zoomed_view(event):
            return "break"
        # Otherwise, scroll the photo view itself (a plain tk.Canvas has
        # no built-in wheel-to-scroll behavior -- this is what wires it up).
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            delta = getattr(event, "delta", 0)
            self.canvas.yview_scroll(int(-delta / 120) or (-1 if delta > 0 else 1), "units")
        return "break"


# PDF user manual panel: shows the real PDF inside the main window ([?] key).

class PdfManualPanel:
    """Renders a PDF as a scrollable, lazily-rendered stack of page images inside a Tk Frame, with both a vertical and a horizontal scrollbar.
    """

    def __init__(self, master, pdf_path):
        self.frame = ttk.Frame(master)
        self._pdf_path = pdf_path
        self._photo_refs = []  # keeps live references -- Tk drops images otherwise
        self._rendered = False
        self._build_scrollable_canvas()

    def _build_scrollable_canvas(self):
        """Builds the canvas plus BOTH scrollbars."""
        self.canvas = tk.Canvas(self.frame, bg="#3c3c3c", highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        # Same bind_all-while-hovering pattern as EmbeddedCanvas, for
        # the same reliability reason -- see its own docstring.
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _on_leave(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            delta = getattr(event, "delta", 0)
            self.canvas.yview_scroll(int(-delta / 120) or (-1 if delta > 0 else 1), "units")
        return "break"

    def _render_one_page(self, page, dpi, y, viewport_width):
        """Renders one PDF page at vertical offset y, horizontally centered within viewport_width. Returns (page_width, page_height)."""
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        pil_image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        photo = ImageTk.PhotoImage(pil_image)
        self._photo_refs.append(photo)
        x = max(0, (viewport_width - pix.width) // 2)
        self.canvas.create_image(x, y, image=photo, anchor="nw")
        return pix.width, pix.height

    def _render_all_pages(self, doc, dpi=110):
        """Stacks every page vertically, centered horizontally, and sets the scrollregion to fit the whole stack."""
        viewport_width = max(1, self.canvas.winfo_width())
        y = 0
        max_w = 0
        page_gap = 16
        for page in doc:
            w, h = self._render_one_page(page, dpi, y, viewport_width)
            y += h + page_gap
            max_w = max(max_w, w)
        # Never narrower than the viewport -- keeps a narrow page centered.
        scrollregion_w = max(viewport_width, max_w)
        self.canvas.configure(scrollregion=(0, 0, scrollregion_w, max(y, 1)))

    def _show_fallback_message(self, message):
        """Shows an error message instead of crashing when the manual can't be displayed (PyMuPDF missing, file missing/corrupt)."""
        self.canvas.create_text(20, 20, anchor="nw", fill="white", width=760,
                                 font=("TkDefaultFont", 12), text=message)
        self.canvas.configure(scrollregion=(0, 0, 800, 100))

    def _import_pymupdf(self):
        """Prefers the modern `import pymupdf` name over the deprecated `fitz` alias. Returns None if neither is available."""
        try:
            import pymupdf as fitz
            return fitz
        except ImportError:
            pass
        try:
            import fitz
            return fitz
        except ImportError:
            return None

    def _load_and_render(self):
        fitz = self._import_pymupdf()
        if fitz is None:
            self._show_fallback_message(
                "The PDF viewer needs the 'pymupdf' package.\n"
                "Install it with: pip install pymupdf"
            )
            return
        try:
            doc = fitz.open(self._pdf_path)
        except Exception as e:
            self._show_fallback_message(f"Could not open the user manual:\n{self._pdf_path}\n{e}")
            return
        try:
            self._render_all_pages(doc)
        finally:
            doc.close()

    def show(self):
        """Packs the panel, then renders the PDF on first call only. Packed before rendering so winfo_width() reflects the real on-screen width."""
        self.frame.pack(fill="both", expand=True)
        if not self._rendered:
            self.canvas.update_idletasks()
            self._load_and_render()
            self._rendered = True

    def hide(self):
        self.frame.pack_forget()


class DatasetBrowserPanel:
    """v1.0.9 [Dataset browser] (code 31): covers the window like the manual. Lists every photo of Training,
    Validation and Test set with its building group number on a white, scrollable list, and moves the selected
    photos to another set (see dataset_browser.py for the rules)."""

    NAVY, TEAL, INK, ZEBRA = "#0f3460", "#1f8a9e", "#1f2937", "#f3f7f9"
    SPLITS = ("training", "validation", "test")

    def __init__(self, master, app):
        self.app = app
        self.frame = tk.Frame(master, bg="white")
        header = tk.Frame(self.frame, bg=self.NAVY)
        header.pack(fill="x")
        tk.Label(header, text="Dataset browser", bg=self.NAVY, fg="white",
                 font=("TkDefaultFont", 15, "bold")).pack(side="left", padx=14, pady=8)
        self.summary = tk.Label(header, text="", bg=self.NAVY, fg="#cfe3ea", font=("TkDefaultFont", 10))
        self.summary.pack(side="left", padx=10)
        tk.Button(header, text="✕ Close", command=self.app._toggle_dataset_browser,
                  cursor="hand2").pack(side="right", padx=12, pady=6)

        actions = tk.Frame(self.frame, bg="#e8f1f4")
        actions.pack(fill="x")
        tk.Label(actions, text="Move the selected photos to:", bg="#e8f1f4", fg=self.INK,
                 font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(14, 6), pady=6)
        self.move_buttons = {}
        for split in self.SPLITS:
            btn = tk.Button(actions, text=self.app.cfg.DATASET_SPLIT_DIRS[split], cursor="hand2",
                            command=lambda t=split: self.move_selected(t))
            btn.pack(side="left", padx=3, pady=6)
            self.move_buttons[split] = btn
        self.whole_building = tk.BooleanVar(value=True)
        tk.Checkbutton(actions, text="Whole building", variable=self.whole_building, bg="#e8f1f4",
                       fg=self.INK).pack(side="left", padx=10)
        tk.Button(actions, text="↻ Refresh", command=self.refresh, cursor="hand2").pack(side="right", padx=12)

        body = tk.Frame(self.frame, bg="white")
        body.pack(fill="both", expand=True)
        style = ttk.Style()
        style.configure("Dataset.Treeview", background="white", fieldbackground="white", foreground=self.INK,
                        rowheight=24, font=("TkDefaultFont", 10))
        style.configure("Dataset.Treeview.Heading", background=self.NAVY, foreground="white",
                        font=("TkDefaultFont", 10, "bold"))
        style.map("Dataset.Treeview", background=[("selected", self.TEAL)], foreground=[("selected", "white")])
        self.tree = ttk.Treeview(body, columns=("building", "kind"), style="Dataset.Treeview",
                                 selectmode="extended")
        self.tree.heading("#0", text="Photo")
        self.tree.heading("building", text="Building group")
        self.tree.heading("kind", text="Notes")
        self.tree.column("#0", width=520, anchor="w")
        self.tree.column("building", width=140, anchor="center")
        self.tree.column("kind", width=160, anchor="w")
        self.tree.tag_configure("set", background="#dbe9ef", foreground=self.NAVY, font=("TkDefaultFont", 11, "bold"))
        self.tree.tag_configure("odd", background=self.ZEBRA)
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.status = tk.Label(self.frame, text="", bg="white", fg=self.INK, anchor="w")
        self.status.pack(fill="x", padx=14, pady=4)
        self._photo_of = {}    # tree item -> (split, photo)
        self._set_item = {}    # split -> section item

    def refresh(self):
        """Rebuilds the list from the folders."""
        children = self.tree.get_children()
        if children:
            self.tree.delete(*children)
        self._photo_of, self._set_item = {}, {}
        rows = self.app.dataset_inventory()
        names = self.app.cfg.DATASET_SPLIT_DIRS
        counts = []
        for split in self.SPLITS:
            mine = [r for r in rows if r["split"] == split]
            buildings = len({r["group"] for r in mine})
            section = self.tree.insert("", "end", text=f"{names[split]}  ({len(mine)} photo(s))",
                                       values=(f"{buildings} building(s)", ""), open=True, tags=("set",))
            self._set_item[split] = section
            for i, r in enumerate(mine):
                notes = ", ".join(n for n, on in (("drone", r["drone"]), ("no cracks", r["negative"])) if on)
                item = self.tree.insert(section, "end", text=r["photo"], values=(r["number"], notes),
                                        tags=("odd",) if i % 2 else ())
                self._photo_of[item] = (split, r["photo"])
            counts.append(f"{names[split]}: {len(mine)}")
        self.summary.configure(text="   ".join(counts))

    def selected_photos(self):
        """[(split, photo)] selected; a selected set row stands for all its photos."""
        picked = []
        for item in self.tree.selection():
            if item in self._photo_of:
                picked.append(self._photo_of[item])
            else:
                picked += [self._photo_of[c] for c in self.tree.get_children(item) if c in self._photo_of]
        return sorted(set(picked))

    def move_selected(self, target):
        picked = self.selected_photos()
        names = self.app.cfg.DATASET_SPLIT_DIRS
        if not picked:
            self.status.configure(text="Select one or more photos first (Ctrl/Shift + click for several).")
            return False
        whole = bool(self.whole_building.get())
        plan = self.app.plan_set_move(picked, target, whole)
        if not plan:
            self.status.configure(text=f"The selected photos are already in {names[target]}.")
            return False
        extra = len(plan) - len([p for p in picked if p[0] != target])
        lines = [f"Move {len(plan)} photo(s) to {names[target]}?"]
        if extra > 0:
            lines.append(f"{extra} of them are other photos of the same building(s), moved too so no building "
                         "is in two sets.")
        lines.append("Training set keeps photo, JSON and masks; Validation and Test set keep the photo only.")
        if target == "training" or any(s == "training" for s, _ in plan):
            lines.append("The Training set changes: press CV folds again afterwards.")
        if not messagebox.askyesno("Crack Detector - Dataset browser", "\n".join(lines), parent=self.frame):
            return False
        moved, problems = self.app.move_photos_between_sets(picked, target, whole)
        self.app._all_segmented_cache = None
        self.refresh()
        text = f"{moved} photo(s) moved to {names[target]}."
        if problems:
            text += f" {len(problems)} not moved: " + "; ".join(problems[:3])
        self.status.configure(text=text)
        print(f" [DATASET BROWSER] {text}")
        return moved > 0

    def show(self):
        self.frame.pack(fill="both", expand=True)
        self.refresh()

    def hide(self):
        self.frame.pack_forget()


# Status strip: fixed panel at the bottom showing console output continuously
# (a double-clicked .app/.exe otherwise has no visible console).

# USER REQUEST: real, clickable shortcut buttons (not just a static list) for
# the commands that aren't already toolbar buttons -- icons match the ones
# already used for the same commands in _TOOL_COMMANDS/_VIEW_COMMANDS/etc.
SIDEBAR_WIDTH = 210
# v1.2.0: flat "sidebar" icon at the top of each sidebar that hides it; the hidden one keeps only a thin strip.
SIDEBAR_COLLAPSED_WIDTH = 34
# v1.0.8: two sidebars, each split into sections of similar commands (heading, [(icon, label, key, code)]).
# Right ("More Features"): tracing, crack editing and crack inspection -- what is used while working on a photo.
_RIGHT_SIDEBAR_SECTIONS = [
    ("Tools", [
        ("\u26A1", "Crack tool", "C", ord('c')),
        ("\u25A6", "Detachment tool", "D", ord('d')),
        ("\U0001F6AB", "Ignore region", "I", ord('i')),
        # v1.1.0: on = the next cracks are traced as uncertain (red); again = normal (code 7, uncertain_trace.py).
        ("\U0001F534", "Uncertain crack", "", 7),
        # v1.3.0: after each trace, a weak crack is proposed as uncertain (code 12, warn_uncertain.py).
        ("\u26A0", "Warn uncertain", "", 12),
    ]),
    ("Edit cracks", [
        ("\u2702", "Cut crack part", "1", ord('1')),
        ("\U0001F517", "Join / re-route", "2", ord('2')),
        ("\u26D3", "Link two cracks", "3", ord('3')),
        # v1.0.8 button-only clean-up tools (codes 21-22, see crack_cleanup.py).
        ("\u232B", "Eraser", "", 21),
        ("\U0001F5D1", "Remove uncertain [?]", "", 22),
        # v1.0.7: the arrow turns left once no other route is left; [$] applies the route on screen.
        ("\u25B6", "Alternative route", "", 18),
        ("\U0001F9F2", "Snap / translate", "T", ord('t')),
        ("\u2194", "Width-edit mode", "A", ord('a')),
        ("\U0001F504", "Retrace cracks", "V", ord('v')),
    ]),
    ("Inspect cracks", [
        ("\U0001F522", "Show cracks by number", "", 19),
        ("\U0001F4CB", "Crack report", "", 16),
        # v1.3.0: hover a crack number or line to read its data (code 11, crack_labels.py).
        ("\U0001F4AC", "Crack labels", "", 11),
        ("\U0001F7E9", "Highlight viewed (HWAV)", "", 14),
    ]),
]
# Left ("View & Dataset"): display, building group / scale, and the dataset (mode, validation, sets, folds).
_LEFT_SIDEBAR_SECTIONS = [
    ("View", [
        ("\U0001F535", "Toggle blue overlay", "N", ord('n')),
        ("\U0001F4CD", "Toggle markers", "Space", ord(' ')),
        ("\u2B1A", "Zoom window", "", 15),
        ("\U0001F50E", "Reset zoom", "0", ord('0')),
        ("\u26F6", "Fullscreen", "F", ord('f')),
    ]),
    ("Building & scale", [
        ("\U0001F3F7", "Assign building", "B", ord('b')),
        ("\u2B1C", "Building portion (import)", "5", 20),
        ("\U0001F50D", "Compatibility check", "G", ord('g')),
        ("\U0001F4D0", "Set scale", "K", ord('k')),
    ]),
    ("Dataset", [
        ("\U0001F500", "Switch Mode", "", 6),  # no physical key -- button-only action
        ("\u2705", "Validation", "", 17),
        # v1.0.8 dataset split, Mode 2 only (codes 23-25, see dataset_split.py).
        ("\U0001F4DA", "Training set", "", 23),
        ("\U0001F9EA", "Test set", "", 24),
        ("\U0001F4CF", "Validation set", "", 25),
        ("\u21F6", "Auto split (all photos)", "", 28),
        ("\u21BA", "Reset pipeline training", "", 30),
        ("\U0001F4C2", "Dataset browser", "", 31),
        ("\U0001F5C2", "CV folds (Training set)", "", 26),
    ]),
]
RIGHT_SIDEBAR_TITLE = "More Features"
LEFT_SIDEBAR_TITLE = "View & Dataset"
# Every sidebar command, right panel first then left, in on-screen order.
_SIDEBAR_SHORTCUTS = [entry for sections in (_RIGHT_SIDEBAR_SECTIONS, _LEFT_SIDEBAR_SECTIONS)
                      for _, entries in sections for entry in entries]

# Sidebar toggles shown pressed while their tool is on: code -> reads the state from cfg.
_SIDEBAR_TOGGLE_STATES = {
    ord('i'): lambda cfg: cfg.current_tool == 'ignore',
    14: lambda cfg: cfg.show_viewed_windows,
    15: lambda cfg: cfg.zoom_window_state["active"],
    16: lambda cfg: cfg.crack_report_state["active"],
    18: lambda cfg: cfg.alt_route_state["active"],
    19: lambda cfg: cfg.crack_filter is not None,
    20: lambda cfg: cfg.portion_state["active"] or cfg.portion_state["rect"] is not None,
    21: lambda cfg: cfg.eraser_state["active"],
    7: lambda cfg: cfg.uncertain_trace_mode,
    11: lambda cfg: cfg.crack_labels_mode,
    12: lambda cfg: cfg.warn_uncertain_mode,
}

# Sidebar buttons whose icon follows the state: code -> reads the icon from cfg.
_SIDEBAR_DYNAMIC_ICONS = {
    18: lambda cfg: "\u25C0" if cfg.alt_route_state["active"] and cfg.alt_route_state["direction"] < 0 else "\u25B6",
}
VALIDATION_CODE = 17
# v1.3.0: Ctrl/Cmd+click on these sidebar buttons sends another code: [Crack tool] -> make a blue crack uncertain,
# [Uncertain crack] -> make a red crack normal (then click the crack or its number).
_SIDEBAR_CTRL_CLICK_CODES = {ord('c'): MARK_UNCERTAIN_CODE, 7: MARK_NORMAL_CODE}
# Sidebar buttons enabled only when this holds (greyed out otherwise): code -> reads it from cfg.
_SIDEBAR_ENABLED_STATES = {
    code: (lambda cfg: cfg.modalita_scelta == "2" and not cfg.queue_idle) for code in (23, 24, 25)
}
# Dataset split buttons: hovering one shows how many photos each set holds (code -> DATASET_SPLIT_DIRS key).
_SPLIT_SET_BUTTONS = {23: "training", 24: "test", 25: "validation"}


class HoverTip:
    """Small borderless window next to a widget while the mouse is over it; its text is built
    when the mouse enters, so it always shows the current state."""

    def __init__(self, widget, text_fn, side="left"):
        self.widget, self.text_fn, self.tip, self.side = widget, text_fn, None, side
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)   # a click moves the photo: the counts change

    def show(self, event=None):
        self.hide()
        try:
            text = self.text_fn()
        except Exception as e:
            text = f"(not available: {e})"
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        tk.Label(self.tip, text=text, justify="left", anchor="w", background="#ffffe0", foreground="#000000",
                 relief="solid", borderwidth=1, font=("TkDefaultFont", 10), padx=8, pady=5).pack()
        self.tip.update_idletasks()
        # Towards the canvas, top-aligned with the button: left of a right-sidebar button, right of a left one.
        if self.side == "left":
            x = self.widget.winfo_rootx() - self.tip.winfo_reqwidth() - 6
        else:
            x = self.widget.winfo_rootx() + self.widget.winfo_width() + 6
        self.tip.wm_geometry(f"+{max(0, x)}+{self.widget.winfo_rooty()}")

    def hide(self, event=None):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


def split_counts_text(counts, hovered):
    """Hover text of the set buttons: photos per set, the hovered set marked, and the total."""
    lines = ["Photos per set:"]
    for name, n in counts:
        mark = "\u25B6" if name == hovered else "  "
        lines.append(f"{mark} {name}: {n}")
    lines.append(f"   Total: {sum(n for _, n in counts)}")
    return "\n".join(lines)


class SidebarToggle:
    """v1.2.0 flat icon button that hides/shows a sidebar, drawn on a small Canvas (no 3D relief): a rounded
    window outline whose side panel, on the sidebar's side, is filled while the sidebar is shown and empty, with
    a small arrow pointing back in, while it is hidden. Grey at rest; darker on a light pill when hovered."""

    SIZE_W, SIZE_H = 28, 24
    INK, INK_HOVER, ACCENT = "#6b7280", "#1f2937", "#2563eb"
    HOVER_BG = "#e5e7eb"

    def __init__(self, master, side, command):
        self.side, self.command = side, command
        self.collapsed, self.hovered = False, False
        self.bg = self._parent_background(master)
        self.canvas = tk.Canvas(master, width=self.SIZE_W, height=self.SIZE_H, highlightthickness=0, bd=0,
                                bg=self.bg, cursor="hand2", takefocus=0)
        self.redraw()

    @staticmethod
    def _parent_background(master):
        try:
            return master.cget("background")
        except (tk.TclError, AttributeError, TypeError):
            return "#ececec"

    def bind_events(self):
        """add='+' keeps the HoverTip bindings made on the same canvas."""
        self.canvas.bind("<Enter>", self._on_enter, add="+")
        self.canvas.bind("<Leave>", self._on_leave, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._on_click, add="+")

    def pack(self, **kw):
        self.canvas.pack(**kw)

    def _on_enter(self, event=None):
        self.hovered = True
        self.redraw()

    def _on_leave(self, event=None):
        self.hovered = False
        self.redraw()

    def _on_click(self, event=None):
        self.command()

    def set_collapsed(self, collapsed):
        self.collapsed = collapsed
        self.redraw()

    @staticmethod
    def _rounded_rect_points(x0, y0, x1, y1, r):
        """Corner points doubled so a smooth polygon draws a rectangle with rounded corners."""
        return [x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y0 + r, x1, y1 - r,
                x1, y1 - r, x1, y1, x1 - r, y1, x1 - r, y1, x0 + r, y1, x0 + r, y1, x0, y1, x0, y1 - r,
                x0, y1 - r, x0, y0 + r, x0, y0 + r, x0, y0]

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = self.SIZE_W, self.SIZE_H
        ink = self.INK_HOVER if self.hovered else self.INK
        if self.hovered:
            c.create_polygon(self._rounded_rect_points(1, 1, w - 1, h - 1, 6), smooth=True,
                             fill=self.HOVER_BG, outline="")
        x0, y0, x1, y1 = 6, 6, w - 6, h - 6
        c.create_polygon(self._rounded_rect_points(x0, y0, x1, y1, 3), smooth=True, fill="", outline=ink, width=1.6)
        divider = x0 + 5 if self.side == "left" else x1 - 5
        c.create_line(divider, y0, divider, y1, fill=ink, width=1.6)
        if not self.collapsed:
            fx0, fx1 = (x0 + 1, divider) if self.side == "left" else (divider, x1 - 1)
            c.create_rectangle(fx0, y0 + 2, fx1, y1 - 1, fill=(self.ACCENT if self.hovered else ink), outline="")
        else:
            self._draw_arrow(c, ink, x0, x1, y0, y1, divider)

    def _draw_arrow(self, c, ink, x0, x1, y0, y1, divider):
        """Small chevron in the empty window, pointing the way the sidebar opens (> for the left, < for the right)."""
        cy = (y0 + y1) / 2.0
        if self.side == "left":
            cx = (divider + x1) / 2.0
            pts = (cx - 1.5, cy - 3, cx + 1.5, cy, cx - 1.5, cy + 3)
        else:
            cx = (x0 + divider) / 2.0
            pts = (cx + 1.5, cy - 3, cx - 1.5, cy, cx + 1.5, cy + 3)
        c.create_line(*pts, fill=(self.ACCENT if self.hovered else ink), width=1.6, capstyle="round",
                      joinstyle="round")


class ShortcutsSidebar:
    """v1.0.8: both sidebars, one on each side of the canvas, seen as one: buttons, buttons_by_code and
    refresh_states() cover the two panels (right panel first). frame / _canvas are the right panel's."""

    def __init__(self, master, app):
        self.app = app
        self.right = SidebarPanel(master, app, RIGHT_SIDEBAR_TITLE, _RIGHT_SIDEBAR_SECTIONS, side="right")
        self.left = SidebarPanel(master, app, LEFT_SIDEBAR_TITLE, _LEFT_SIDEBAR_SECTIONS, side="left")
        self.panels = (self.right, self.left)
        self.frame, self._canvas = self.right.frame, self.right._canvas
        self.buttons = self.right.buttons + self.left.buttons
        self.buttons_by_code = {**self.right.buttons_by_code, **self.left.buttons_by_code}
        self.hover_tips = {**self.right.hover_tips, **self.left.hover_tips}

    def refresh_states(self, cfg, validation_available):
        for panel in self.panels:
            panel.refresh_states(cfg, validation_available)

    def hidden_sides(self):
        """v1.2.0: the sides whose sidebar is currently hidden ('left', 'right')."""
        return tuple(panel.side for panel in self.panels if panel.collapsed)


class SidebarPanel:
    """Panel of real, clickable shortcut buttons beside the canvas, for the
    commands not already on the toolbar, grouped under section headings.
    Vertically scrollable so a button is never simply unreachable on a
    shorter window (USER REPORT -- the last one used to be clipped)."""

    def __init__(self, master, app, title, sections, side="right"):
        self.app, self.side = app, side
        self.collapsed = False
        self.frame = ttk.Frame(master, width=SIDEBAR_WIDTH)
        self.frame.pack(side=side, fill="y", padx=(4, 2) if side == "right" else (2, 4), pady=2)
        self.frame.pack_propagate(False)  # keeps a fixed width regardless of content

        self._build_header(title)

        canvas = tk.Canvas(self.frame, highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")
        self._canvas, self._v_scroll = canvas, v_scroll

        inner = ttk.Frame(canvas)
        canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Enter>", lambda e: self._bind_wheel())
        canvas.bind("<Leave>", lambda e: self._unbind_wheel())

        self.buttons = []
        self.buttons_by_code = {}
        self.headings = []
        self._labels_by_code = {code: (label, key) for _, entries in sections for _, label, key, code in entries}
        self._shown_texts = {}  # code -> text last set by _set_icon()
        for heading, entries in sections:
            self.headings.append(self._add_heading(inner, heading))
            for icon, label, key, code in entries:
                btn = self._add_row(inner, icon, label, key, code)
                self.buttons.append(btn)
                self.buttons_by_code[code] = btn
        self.hover_tips = {code: HoverTip(self.buttons_by_code[code], self._split_tip_text(split), side=side)
                           for code, split in _SPLIT_SET_BUTTONS.items() if code in self.buttons_by_code}

    def _build_header(self, title):
        """Title row with the small sidebar icon that hides/shows the panel (v1.2.0), on the canvas side."""
        header = tk.Frame(self.frame)
        header.pack(fill="x", pady=(4, 2))
        button_side = "left" if self.side == "right" else "right"
        self.collapse_button = SidebarToggle(header, self.side, self.toggle_collapsed)
        self.collapse_button.pack(side=button_side, padx=2, pady=2)
        self.title_label = tk.Label(header, text=title, font=("TkDefaultFont", 10, "bold"), anchor="w")
        self._pack_title()
        # The tip opens towards the canvas, like the button itself; made first, the toggle's hover adds to it.
        self.collapse_tip = HoverTip(self.collapse_button.canvas, self._collapse_tip_text, side=button_side)
        self.collapse_button.bind_events()

    def _pack_title(self):
        self.title_label.pack(side="left", fill="x", expand=True, padx=(8, 2), pady=(4, 2))

    def _collapse_tip_text(self):
        where = "right" if self.side == "right" else "left"
        if self.collapsed:
            return f"Show the {where} sidebar"
        return f"Hide the {where} sidebar: the photo extends to the {where}"

    def toggle_collapsed(self):
        """v1.2.0 sidebar icon: hides the panel (only a thin strip with the button is left, so it can be
        shown again) and gives its width to the photo; a second click brings the panel back."""
        self.set_collapsed(not self.collapsed)

    def set_collapsed(self, collapsed):
        if collapsed == self.collapsed:
            return
        self.collapsed = collapsed
        if collapsed:
            self._unbind_wheel()
            self.title_label.pack_forget()
            self._canvas.pack_forget()
            self._v_scroll.pack_forget()
            self.frame.configure(width=SIDEBAR_COLLAPSED_WIDTH)
        else:
            self.frame.configure(width=SIDEBAR_WIDTH)
            self._pack_title()
            self._canvas.pack(side="left", fill="both", expand=True)
            self._v_scroll.pack(side="right", fill="y")
        self.collapse_button.set_collapsed(collapsed)
        self.app._on_sidebar_layout_changed()

    @staticmethod
    def _add_heading(master, text):
        """Small grey section title above a group of similar commands."""
        label = tk.Label(master, text=text.upper(), font=("TkDefaultFont", 8, "bold"), fg="#6b7280", anchor="w")
        label.pack(fill="x", padx=6, pady=(8, 1))
        return label

    def _split_tip_text(self, split):
        return lambda: split_counts_text(self.app.dataset_split_counts(), self.app.cfg.DATASET_SPLIT_DIRS[split])

    def refresh_states(self, cfg, validation_available):
        """Toggle buttons look pressed while their tool is on; Validation is enabled once every photo is segmented."""
        for code, is_on in _SIDEBAR_TOGGLE_STATES.items():
            btn = self.buttons_by_code.get(code)
            if btn is not None:
                btn.configure(relief=("sunken" if is_on(cfg) else "raised"))
        btn = self.buttons_by_code.get(VALIDATION_CODE)
        if btn is not None:
            btn.configure(state=("normal" if validation_available else "disabled"))
        for code, enabled in _SIDEBAR_ENABLED_STATES.items():
            btn = self.buttons_by_code.get(code)
            if btn is not None:
                btn.configure(state=("normal" if enabled(cfg) else "disabled"))
        for code, icon_of in _SIDEBAR_DYNAMIC_ICONS.items():
            self._set_icon(code, icon_of(cfg))

    def _set_icon(self, code, icon):
        """Re-labels a sidebar button with a new icon, only when it actually changed."""
        btn = self.buttons_by_code.get(code)
        if btn is None:
            return
        label, key = self._labels_by_code[code]
        text = f"{icon} {label}" + (f" [{key}]" if key else "")
        if self._shown_texts.get(code) != text:
            btn.configure(text=text)
            self._shown_texts[code] = text

    def _bind_wheel(self):
        # Same bind_all-while-hovering pattern as EmbeddedCanvas/PdfManualPanel/toolbar.
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", self._on_wheel)
        self._canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        if getattr(event, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _add_row(self, master, icon, label, key, code):
        """One real, clickable shortcut button, all on a single, compact
        line: icon, label, and the [KEY] badge together -- shorter than
        a two-line button, so more fit in the same height. A plain
        tk.Button (not ttk) -- its wraplength support is unambiguous
        across themes/platforms, unlike ttk.Button's, which risks a hard
        crash on some Tk builds if the option isn't in its supported set."""
        key_part = f" [{key}]" if key else ""  # some actions (Switch Mode) have no physical key
        btn = tk.Button(
            master, takefocus=0, relief="raised", bd=1, cursor="hand2",
            text=f"{icon} {label}{key_part}",
            wraplength=SIDEBAR_WIDTH - 20, justify="left", anchor="w",
            font=("TkDefaultFont", 9),
            command=lambda c=code: self.app._send(c),
        )
        btn.pack(fill="x", padx=4, pady=2, ipady=3)
        if code in _SIDEBAR_CTRL_CLICK_CODES:
            self._bind_ctrl_click(btn, _SIDEBAR_CTRL_CLICK_CODES[code])
        return btn

    def _bind_ctrl_click(self, btn, alt_code):
        """v1.3.0: Ctrl+click (Cmd+click on macOS) on the button sends alt_code instead of its own command."""
        def on_ctrl_click(event, c=alt_code):
            self.app._send(c)
            return "break"   # the button is not pressed, so its own command does not run
        sequences = ["<Control-Button-1>"]
        if sys.platform == "darwin":
            sequences += ["<Command-Button-1>", "<Control-Button-2>", "<Control-Button-3>"]
        for sequence in sequences:
            btn.bind(sequence, on_ctrl_click)


class StatusBar:
    """A fixed strip embedded at the bottom of the main window (not a separate window), sized explicitly to fit alongside the canvas and toolbar.
    """
    MAX_LINES = 1000  # bounded scroll-back history so a long session doesn't grow forever
    # Bigger, more readable text, traded for fewer visible lines at once.
    HEIGHT_LINES = 2
    FONT_SIZE = 12

    def __init__(self, master):
        frame = ttk.Frame(master)
        frame.pack(fill="x", side="bottom")

        # v1.3.0: active tools/modes (Warn uncertain, Crack labels, Eraser ...), shown here instead of on the photo.
        self.mode_label = tk.Label(frame, text="", justify="left", anchor="w", wraplength=460,
                                   font=("TkDefaultFont", self.FONT_SIZE, "bold"), padx=6)
        self.mode_label.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self._modes = ()
        self._mode_default_bg = self.mode_label.cget("background")

        self.text = tk.Text(frame, height=self.HEIGHT_LINES, wrap="word", state="disabled",
                             font=("TkDefaultFont", self.FONT_SIZE))
        self.text.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        scrollbar.pack(side="left", fill="y", pady=4)
        self.text.configure(yscrollcommand=scrollbar.set)

        arrow_buttons = ttk.Frame(frame)
        arrow_buttons.pack(side="left", fill="y", padx=(2, 4), pady=4)
        ttk.Button(arrow_buttons, text="\u25b2", width=2, takefocus=0,
                   command=lambda: self.scroll(-1)).pack()
        ttk.Button(arrow_buttons, text="\u25bc", width=2, takefocus=0,
                   command=lambda: self.scroll(1)).pack()

        self._lines = []

    def add_message(self, line):
        self._lines.append(line)
        if len(self._lines) > self.MAX_LINES:
            self._lines = self._lines[-self.MAX_LINES:]
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")  # keeps the strip scrolled to the latest message
        self.text.configure(state="disabled")

    def scroll(self, units):
        self.text.yview_scroll(units, "units")

    def set_modes(self, texts):
        """Shows the active-mode lines (duplicates dropped, order kept); the widget is touched only when they change."""
        modes = tuple(dict.fromkeys(t for t in texts if t))
        if modes == self._modes:
            return
        self._modes = modes
        self.mode_label.configure(text="\n".join(modes), background="#fff4c2" if modes else self._mode_default_bg,
                                  foreground="#5a3a00")


class _StatusBarStream:
    """A writable stream that mirrors everything written to it into both the real console (if attached) and the status bar widget. Installed as sys.stdout/sys.stderr.
    """

    def __init__(self, real_stream, status_bar):
        self._real_stream = real_stream
        self._status_bar = status_bar
        self._buffer = ""

    def write(self, text):
        if self._real_stream is not None:
            try:
                self._real_stream.write(text)
            except Exception:
                pass
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                # Guarded: a status-bar update failure must never propagate
                # out and get swallowed by unrelated exception handlers elsewhere.
                try:
                    self._status_bar.add_message(line)
                except Exception:
                    pass

    def flush(self):
        if self._real_stream is not None:
            try:
                self._real_stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# Startup dialog: replaces the "1) new images / 2) reload segmented" prompt.

class _ModeChoiceDialog:
    """Startup dialog: choose "1" (new images) or "2" (reload already
    segmented ones). Shows the ECHO&#8209;TWIN logo and a short description
    of the app's purpose above the two options.
    """

    def __init__(self, master, logo_path=None):
        self.result = None
        self.top = tk.Toplevel(master)
        self.top.title("Crack & Detachment Segmentation")
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        self.top.configure(bg="#ffffff")

        self._build_header(logo_path)
        self._build_body()
        self._build_footer()

        self._center_on_screen()
        self.top.grab_set()
        self.top.focus_force()

    def _load_logo(self, logo_path):
        """Loads and downsizes the logo image. None if unavailable -- a
        missing/broken logo must never block the app from starting."""
        if not logo_path or not os.path.exists(logo_path):
            return None
        try:
            img = Image.open(logo_path)
            img.thumbnail((120, 120), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _build_header(self, logo_path):
        """Dark banner with the ECHO-TWIN logo (on its own white card, so
        its transparent background never risks blending oddly against the
        dark banner) and the app title."""
        header = tk.Frame(self.top, bg="#3a0ca3")
        header.pack(fill="x")
        self._logo_photo = self._load_logo(logo_path)
        if self._logo_photo is not None:
            card = tk.Frame(header, bg="#ffffff")
            card.pack(pady=(20, 6))
            tk.Label(card, image=self._logo_photo, bg="#ffffff").pack(padx=14, pady=10)
        tk.Label(header, text="ECHO\u2011TWIN", bg="#3a0ca3", fg="#ffffff",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 2))
        tk.Label(header, text="Crack & Detachment Segmentation", bg="#3a0ca3", fg="#ffffff",
                 font=("TkDefaultFont", 15, "bold")).pack(pady=(0, 20))

    def _build_description(self, body):
        """Short paragraph on the app's purpose and ECHO-TWIN/INAF context."""
        tk.Label(
            body, bg="#ffffff", fg="#374151", justify="left", wraplength=460,
            font=("TkDefaultFont", 10),
            text=("Assisted, pixel-accurate annotation of cracks and material "
                  "detachments on building facade photographs \u2014 developed within "
                  "the ECHO\u2011TWIN project at INAF to produce training data for "
                  "automated structural-defect detection."),
        ).pack(anchor="w", pady=(0, 20))

    def _make_option_button(self, parent, title, subtitle, value):
        """One data-source option, styled as a two-line button."""
        btn = tk.Button(
            parent, bg="#f3f4f6", activebackground="#e5e7eb", relief="flat",
            bd=1, anchor="w", justify="left", cursor="hand2",
            text=f"{title}\n{subtitle}", font=("TkDefaultFont", 10),
            command=lambda: self._choose(value),
        )
        btn.pack(fill="x", pady=6, ipady=10, ipadx=12)

    def _build_body(self):
        """The purpose description and the two data-source buttons."""
        body = tk.Frame(self.top, bg="#ffffff")
        body.pack(fill="both", expand=True, padx=32, pady=24)

        self._build_description(body)

        tk.Label(body, text="Choose the data source:", bg="#ffffff", fg="#111827",
                 font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 10))

        self._make_option_button(
            body, "1)  Load NEW images to process",
            "     photos in 'Images' without a JSON yet", "1",
        )
        self._make_option_button(
            body, "2)  Reload ALREADY SEGMENTED images",
            "     photos with a JSON in 'JSON files', to edit or correct", "2",
        )

    def _build_footer(self):
        tk.Label(self.top, text="ECHO\u2011TWIN Project \u2014 INAF", bg="#ffffff", fg="#9ca3af",
                 font=("TkDefaultFont", 8)).pack(pady=(0, 16))

    def _center_on_screen(self):
        self.top.update_idletasks()
        w, h = self.top.winfo_width(), self.top.winfo_height()
        x = (self.top.winfo_screenwidth() - w) // 2
        y = (self.top.winfo_screenheight() - h) // 2
        self.top.geometry(f"+{x}+{y}")

    def _choose(self, value):
        self.result = value
        self.top.destroy()

    def _on_close(self):
        # Same fallback as the original bare `except Exception: scelta = "1"`.
        self.result = "1"
        self.top.destroy()


class _IgnoreOptionsDialog:
    """[I] options: shape (rectangle / square / free polygon) and which photo edges a rectangle is extended to.
    No edge ticked = the region can sit anywhere in the photo. OK/Enter applies, Cancel/Esc/close keeps the tool unchanged."""

    SHAPES = (("rectangle", "Rectangle"), ("square", "Square"), ("polygon", "Free polygon (click corners, Y to close)"))
    EDGES = (("left", "Extend to the LEFT edge of the photo"), ("right", "Extend to the RIGHT edge of the photo"),
             ("top", "Extend to the TOP edge of the photo"), ("bottom", "Extend to the BOTTOM edge of the photo"))

    def __init__(self, master, current):
        self.result = None
        self.top = tk.Toplevel(master)
        self.top.title("Ignore region")
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self.shape_var = tk.StringVar(master=self.top, value=current.get("shape", "rectangle"))
        edges = current.get("edges") or {}
        self.edge_vars = {key: tk.BooleanVar(master=self.top, value=bool(edges.get(key))) for key, _ in self.EDGES}
        self._build()
        self.top.bind("<Return>", lambda e: self._ok())
        self.top.bind("<Escape>", lambda e: self._cancel())
        self.top.grab_set()
        self.top.focus_force()

    def _build(self):
        body = tk.Frame(self.top, padx=16, pady=12)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Shape", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        for value, label in self.SHAPES:
            tk.Radiobutton(body, text=label, value=value, variable=self.shape_var).pack(anchor="w")
        tk.Label(body, text="Rectangle / square: sides pushed onto the photo edge",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 0))
        tk.Label(body, text="(none ticked = free region anywhere in the photo)", fg="#6b7280").pack(anchor="w")
        for key, label in self.EDGES:
            tk.Checkbutton(body, text=label, variable=self.edge_vars[key]).pack(anchor="w")
        buttons = tk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(buttons, text="Cancel", command=self._cancel).pack(side="right")
        tk.Button(buttons, text="OK", command=self._ok).pack(side="right", padx=(0, 6))

    def _ok(self):
        self.result = {"shape": self.shape_var.get(), "edges": {k: bool(v.get()) for k, v in self.edge_vars.items()}}
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


class _GoToImageDialog:
    """v1.0.9 [Go to image]: First / Last, or an image picked from a list of 10 names at a time
    (paging as in go_to_image.PagedChooser). Go/Enter shows the picked image, Cancel/Esc does nothing."""

    def __init__(self, master, names, current):
        self.result = None
        self.chooser = PagedChooser(names, current)
        self.selected = current
        self.top = tk.Toplevel(master)
        self.top.title("Go to image")
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self._build(len(names), current)
        self.top.bind("<Return>", lambda e: self._go())
        self.top.bind("<Escape>", lambda e: self._cancel())
        self.top.grab_set()
        self.top.focus_force()

    def _build(self, total, current):
        body = tk.Frame(self.top, padx=16, pady=12)
        body.pack(fill="both", expand=True)
        tk.Label(body, text=f"Image {current + 1} of {total} on screen", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ends = tk.Frame(body)
        ends.pack(fill="x", pady=(8, 10))
        tk.Button(ends, text="\u23EE First image", command=lambda: self._finish(0)).pack(side="left")
        tk.Button(ends, text="Last image \u23ED", command=lambda: self._finish(total - 1)).pack(side="right")
        tk.Label(body, text="Or choose an image (10 at a time):").pack(anchor="w")
        row = tk.Frame(body)
        row.pack(fill="x", pady=(2, 0))
        tk.Button(row, text="\u25C0 10", command=self._previous_page).pack(side="left")
        self.combo = ttk.Combobox(row, state="readonly", width=46, values=self.chooser.labels())
        self.combo.pack(side="left", padx=6)
        self.combo.bind("<<ComboboxSelected>>", self._on_pick)
        tk.Button(row, text="10 \u25B6", command=self._next_page).pack(side="left")
        self.page_label = tk.Label(body, fg="#6b7280")
        self.page_label.pack(anchor="w", pady=(2, 0))
        self.combo.set(f"{current + 1}. {self.chooser.names[current]}")
        self._refresh_page_label()
        buttons = tk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(buttons, text="Cancel", command=self._cancel).pack(side="right")
        tk.Button(buttons, text="Go", command=self._go).pack(side="right", padx=(0, 6))

    def _refresh_page_label(self):
        first, last = self.chooser.page_range()
        self.page_label.configure(text=f"Images {first + 1}-{last + 1} of {len(self.chooser.names)}")
        self.combo.configure(values=self.chooser.labels())

    def _on_pick(self, event=None):
        self.selected = self.chooser.choose(self.combo.current())
        self.combo.set(f"{self.selected + 1}. {self.chooser.names[self.selected]}")
        self._refresh_page_label()

    def _next_page(self):
        self.chooser.next_page()
        self._refresh_page_label()

    def _previous_page(self):
        self.chooser.previous_page()
        self._refresh_page_label()

    def _go(self):
        self._finish(self.selected)

    def _finish(self, index):
        self.result = index
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


# Menu bar command tables -- (label, key_code) matching process_keypress()'s
# real key codes. Labels use a Unicode symbol prefix (plain text, no bitmap icons).

_FILE_COMMANDS = [
    ("\U0001F4BE Save and go to next\tEnter", 13),
    ("\U0001F4BE Save (stay on this image)\tS", ord('s')),
    ("\U0001F4E5 Import from previous session (projection)\tW", ord('w')),
    ("\U0001F517 Import - automatic alignment\tL", ord('l')),
    # USER REQUEST: browse the queue without exporting/archiving anything --
    # unlike Save/Enter/S above, neither of these ever saves.
    ("\u25C0 Previous Image (no save)", 5),
    ("\u25B6 Next Image (no save)", 4),
    ("\u2316 Go to image: first / last / choose (no save)", 29),
    # USER REQUEST: switch between "new images" and "reload segmented" data
    # sources without restarting the app (code 6, see process_keypress()).
    ("\U0001F504 Switch Mode (1 \u2194 2)", 6),
    ("\u2705 Validation: move training-ready photos (all photos segmented)", 17),
    ("\U0001F4DA Move to Training set (photo, JSON, masks - Mode 2)", 23),
    ("\U0001F9EA Move to Test set (photo only - Mode 2)", 24),
    ("\U0001F4CF Move to Validation set (photo only - Mode 2)", 25),
    ("\u21F6 Auto split: every segmented photo into the sets, by building", 28),
    ("\u21BA Reset pipeline training: every set back to Images / JSON / Binary files", 30),
    ("\U0001F4C2 Dataset browser: every photo of the sets, with its building", 31),
    ("\U0001F5C2 Cross-validation folds of the Training set (folds.csv)", 26),
    ("\u23FB Quit\tEsc", 27),
]

_EDIT_COMMANDS = [
    ("\u21B6 Undo\tU", ord('u')),
    ("\u21B7 Redo\tR", ord('r')),
    ("\U0001F5D1 Clear all\tX", ord('x')),
    ("\U0001F513 Emergency unlock\tBackspace", 8),
]

_TOOL_COMMANDS = [
    ("\u26A1 Tool: Crack\tC", ord('c')),
    ("\u25A6 Tool: Detachment\tD", ord('d')),
    ("\U0001F6AB Tool: Ignore region (occluded area)\tI", ord('i')),
    ("\U0001F534 Uncertain crack: next cracks traced as uncertain (red) / back to normal", 7),
    ("\u26A0 Warn uncertain: propose weak new cracks as uncertain (confirm) / off", 12),
    ("\U0001F534 Make a crack uncertain: click a blue crack or its number\tCtrl/Cmd+C", MARK_UNCERTAIN_CODE),
    ("\U0001F535 Make a crack normal: click a red crack or its number\tCtrl/Cmd+U", MARK_NORMAL_CODE),
    ("\u2705 Close detachment / ignore polygon\tY", ord('y')),
    ("\u270E Toggle edit mode\tE", ord('e')),
    ("\U0001F9F2 Automatic snap / translate\tT", ord('t')),
    ("\U0001F504 Retrace imported cracks onto real edge\tV", ord('v')),
    ("\u2702 Cut a crack part (2 clicks)\t1", ord('1')),
    ("\U0001F517 Join / re-route cracks (2 clicks)\t2", ord('2')),
    ("\u26D3 Link two cracks end-to-start (2 clicks)\t3", ord('3')),
    ("\u232B Eraser: drag over a crack to erase that part", 21),
    ("\U0001F5D1 Remove uncertain cracks [?]", 22),
    ("\U0001F50D Building-group compatibility check\tG", ord('g')),
    ("\U0001F3F7 Manually assign building group\tB", ord('b')),
    ("\U0001F4CB Crack report (click a crack)", 16),
    ("\U0001F4AC Crack labels: hover a crack to read its data / off", 11),
]

_WIDTH_COMMANDS = [
    ("\u2194 Toggle width-edit mode\tA", ord('a')),
    ("\u2795 Widen (both sides)\t+", ord('+')),
    ("\u2796 Narrow (both sides)\t-", ord('-')),
    ("\u25C0\u2795 Widen left side\t[", ord('[')),
    ("\u25C0\u2796 Narrow left side\t{", ord('{')),
    ("\u25B6\u2795 Widen right side\t]", ord(']')),
    ("\u25B6\u2796 Narrow right side\t}", ord('}')),
]

_VIEW_COMMANDS = [
    ("\u26F6 Toggle fullscreen\tF", ord('f')),
    ("\U0001F50D Reset zoom\t0", ord('0')),
    ("\u2B1A Zoom window (drag a rectangle)", 15),
    ("\U0001F7E9 Highlight windows already viewed (HWAV)", 14),
    ("\U0001F520 Toggle HUD size\tH", ord('h')),
    ("\U0001F4CD Toggle markers\tSpace", ord(' ')),
    ("\U0001F535 Toggle blue crack overlay\tN", ord('n')),
    ("\u2139 Toggle Info (file/length/building status)\tJ", ord('j')),
    ("\U0001F515 Hide on-screen messages\t.", ord('.')),
    ("\U0001F5BC Toggle overlay in export\tM", ord('m')),
    ("\u23EF Pause/resume timer\tP", ord('p')),
    ("\u2753 Toggle user manual\t?", ord('?')),
]

_NAV_COMMANDS = [
    ("\u2B06 Up", 0),
    ("\u2B07 Down", 1),
    ("\u2B05 Left", 2),
    ("\u27A1 Right", 3),
]

_CALIBRATION_COMMANDS = [
    ("\U0001F4D0 Set scale manually\tK", ord('k')),
]

_ALL_MENU_GROUPS = [
    ("\U0001F4C1 File", _FILE_COMMANDS),
    ("\u270F\uFE0F Edit", _EDIT_COMMANDS),
    ("\U0001F6E0 Tools", _TOOL_COMMANDS),
    ("\u2194 Crack Width", _WIDTH_COMMANDS),
    ("\U0001F441 View", _VIEW_COMMANDS),
    ("\U0001F9ED Navigate", _NAV_COMMANDS),
    ("\U0001F4CF Calibration", _CALIBRATION_COMMANDS),
]


class GuiCrackSegmentation(CrackSegmentation):
    """Subclasses CrackSegmentation, overriding only its GUI integration hooks (display, prompts, event pumping). All other logic is inherited unchanged.
    """

    def _resolve_app_resource_dir(self):
        """Folder the app's own bundled, read-only resources (the PDF manual) live in -- distinct from cfg.SCRIPT_DIR, which is a per-user data folder.
        """
        if getattr(sys, 'frozen', False):
            return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        return os.path.dirname(os.path.abspath(__file__))

    def _resolve_manual_pdf_path(self):
        """Path of the bundled PDF user manual."""
        return os.path.join(self._resolve_app_resource_dir(), "Crack_Segmentation_User_Manual.pdf")

    def _resolve_echo_twin_logo_path(self):
        """Path of the bundled ECHO-TWIN logo, shown on the startup dialog."""
        return os.path.join(self._resolve_app_resource_dir(), "echo_twin_logo.png")

    def _build_window_chrome(self):
        """Builds the menu bar, toolbar, a middle row (canvas + shortcuts
        sidebar), PDF panel, and status bar, plus a placeholder frame
        before the first real render."""
        self._build_menu_bar()
        self._build_toolbar()
        self._middle_row = ttk.Frame(self._tk_root)
        self._middle_row.pack(fill="both", expand=True)
        self._sidebar = ShortcutsSidebar(self._middle_row, self)
        self._canvas = EmbeddedCanvas(self._middle_row, self)
        self._canvas.canvas.focus_set()
        self._pdf_panel = PdfManualPanel(self._tk_root, self._resolve_manual_pdf_path())
        self._pdf_panel_visible = False
        self._dataset_panel = DatasetBrowserPanel(self._tk_root, self)
        self._dataset_panel_visible = False
        self._status_bar = StatusBar(self._tk_root)
        placeholder = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self._canvas.update_frame(placeholder)

    def _on_sidebar_layout_changed(self):
        """A sidebar was hidden or shown (v1.2.0): lays out the new canvas width now, so the next frame is drawn
        at the new size (see EmbeddedCanvas._compute_display_scale)."""
        try:
            self._tk_root.update_idletasks()
        except tk.TclError:
            return
        hidden = self._sidebar.hidden_sides() if hasattr(self, "_sidebar") else ()
        names = {(): "both sidebars shown", ("right",): "right sidebar hidden",
                 ("left",): "left sidebar hidden"}
        print(f" [VIEW] {names.get(hidden, 'both sidebars hidden: the photo takes the whole width')}.")

    def _wire_stdout_stderr_to_status_bar(self):
        """Mirrors every print()'d line into the status bar too, in addition to the real console if one is attached."""
        sys.stdout = _StatusBarStream(sys.stdout, self._status_bar)
        sys.stderr = _StatusBarStream(sys.stderr, self._status_bar)

    def _get_screen_dimensions(self):
        """Returns (screen_w, screen_h), falling back to a sane default if Tk can't report them."""
        try:
            return self._tk_root.winfo_screenwidth(), self._tk_root.winfo_screenheight()
        except tk.TclError:
            return 1280, 800

    def _compute_initial_window_geometry(self, screen_w, screen_h):
        """Computes (target_w, target_h, x) for the initial window: fits the canvas+sidebar when the screen allows, but never wider than the screen."""
        # Generous chrome allowance (OS menu bar/dock/title bar/toolbar/status bar).
        chrome_allowance = 260
        default_canvas_viewport_h = 650  # comfortable default; resize the window to see more/less
        screen_capped_w = max(900, screen_w - 40)
        canvas_and_sidebar_w = CANVAS_W + 24 + 2 * SIDEBAR_WIDTH   # v1.0.8: a sidebar on each side
        # USER REPORT: forcing the window wider than the screen (so the
        # toolbar never had to scroll) instead pushed the Shortcuts
        # sidebar's own right edge off-screen -- unlike the toolbar, the
        # sidebar has no scrollbar to reach the cut-off part. Now that the
        # toolbar itself scrolls horizontally (see _build_toolbar), it no
        # longer needs this window-width floor at all: the window is
        # capped to the screen, and the canvas (which already has its own
        # scrollbars) is what gives up space first, not the sidebar --
        # pack() always honors the sidebar's own fixed width first.
        target_w = min(max(canvas_and_sidebar_w, 900), screen_capped_w)
        target_h = min(default_canvas_viewport_h + chrome_allowance, max(400, screen_h - 100))
        x = max(0, (screen_w - target_w) // 2)
        return target_w, target_h, x

    def _apply_initial_window_geometry(self):
        """Sizes the resizable main window to a modest, likely-to-fit default height; scrollbars reveal whatever doesn't fit on a smaller screen."""
        try:
            self._tk_root.update_idletasks()
            screen_w, screen_h = self._get_screen_dimensions()
            target_w, target_h, x = self._compute_initial_window_geometry(screen_w, screen_h)

            self._tk_root.geometry(f"{target_w}x{target_h}+{x}+0")
            self._tk_root.minsize(900, 400)
            self._tk_root.resizable(True, True)
        except tk.TclError:
            # Last-resort fallback: skip explicit sizing rather than risk crashing.
            pass

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._tk_root = tk.Tk()
        self._tk_root.title("Crack & Detachment Segmentation")
        self._tk_root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._build_window_chrome()
        self._wire_stdout_stderr_to_status_bar()

        self._tk_root.bind("<KeyPress>", self._on_keypress)
        self._apply_initial_window_geometry()

    # Window chrome

    def _on_window_close(self):
        # Closing the window is equivalent to pressing ESC, not a silent kill.
        self._pending_keys.append(27)

    def _send(self, key_code):
        # The bridge between every menu/toolbar command and the segmentation
        # logic: queues the code, drained through process_keypress() like a real keystroke.
        self._pending_keys.append(key_code)
        # Returns focus to the canvas after every button click -- a focused
        # button can otherwise swallow the very next real keypress (e.g. Enter).
        try:
            self._canvas.canvas.focus_set()
        except tk.TclError:
            pass

    def _on_keypress(self, event):
        code = tk_key_event_to_code(event)
        if code is not None:
            self._pending_keys.append(code)

    def _build_menu_bar(self):
        menubar = tk.Menu(self._tk_root)
        for group_name, commands in _ALL_MENU_GROUPS:
            menu = tk.Menu(menubar, tearoff=0)
            for label, code in commands:
                menu.add_command(label=label, command=lambda c=code: self._send(c))
            menubar.add_cascade(label=group_name, menu=menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        # Deliberate alias of _VIEW_COMMANDS's own "?" entry.
        help_menu.add_command(label="\u2753 Show user manual\t?", command=lambda: self._send(ord('?')))
        menubar.add_cascade(label="\u2753 Help", menu=help_menu)

        self._tk_root.config(menu=menubar)

    def _toolbar_shortcuts(self):
        """The handful of most-used actions shown as toolbar buttons; every command is always reachable from the menu bar too."""
        # v1.1.0: short labels so the row fits the window up to Manual; full names and keys are in the menus.
        return [
            ("\u25C0 Prev", 5),
            ("\u25B6 Next", 4),
            ("\u2316 Go to", 29),
            ("\U0001F4BE Save", ord('s')),
            ("\U0001F4BE Save+next", 13),
            ("\U0001F4E5 W", ord('w')),
            ("\U0001F517 L", ord('l')),
            ("\u21B6 Undo", ord('u')),
            ("\u21B7 Redo", ord('r')),
            ("\U0001F5D1 Clear", ord('x')),
            ("\u2139 Info", ord('j')),
            ("\u2753 Manual", ord('?')),
        ]

    def _create_one_toolbar_button(self, bar, label, code):
        """Creates one toolbar button, tracking it as the Previous/Next Image reference if it is one."""
        # takefocus=0: stops these buttons from grabbing keyboard focus on click.
        # width=-1: as narrow as its text allows (no fixed minimum width).
        btn = ttk.Button(bar, text=label, command=lambda c=code: self._send(c), takefocus=0, width=-1)
        btn.pack(side="left", padx=1)
        # Greyed out at the start/end of the queue -- see _update_navigation_button_states().
        if code == 5:
            self._prev_image_button = btn
        elif code == 4:
            self._next_image_button = btn

    def _build_toolbar(self):
        # A slim, horizontally-scrollable row for the handful of most-used
        # actions; every command is also reachable from the menu bar too.
        # USER REPORT: the last button (Manual) kept being unreachable on
        # a narrower window/screen no matter how the target width was
        # computed -- scrollable makes that impossible, since every
        # button is always reachable by scrolling regardless of width.
        outer = ttk.Frame(self._tk_root)
        outer.pack(fill="x")

        canvas = tk.Canvas(outer, height=40, highlightthickness=0)
        h_scroll = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)
        canvas.pack(side="top", fill="x")
        h_scroll.pack(side="top", fill="x")
        self._toolbar_canvas = canvas

        bar = ttk.Frame(canvas)
        canvas.create_window(0, 0, window=bar, anchor="nw")
        self._toolbar_frame = bar  # measured by __init__ to size the window wide enough for it
        bar.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Enter>", lambda e: self._bind_toolbar_wheel())
        canvas.bind("<Leave>", lambda e: self._unbind_toolbar_wheel())

        self._prev_image_button = None
        self._next_image_button = None
        for label, code in self._toolbar_shortcuts():
            self._create_one_toolbar_button(bar, label, code)

    def _bind_toolbar_wheel(self):
        # Same bind_all-while-hovering pattern as EmbeddedCanvas/PdfManualPanel.
        self._toolbar_canvas.bind_all("<MouseWheel>", self._on_toolbar_wheel)
        self._toolbar_canvas.bind_all("<Button-4>", self._on_toolbar_wheel)
        self._toolbar_canvas.bind_all("<Button-5>", self._on_toolbar_wheel)

    def _unbind_toolbar_wheel(self):
        self._toolbar_canvas.unbind_all("<MouseWheel>")
        self._toolbar_canvas.unbind_all("<Button-4>")
        self._toolbar_canvas.unbind_all("<Button-5>")

    def _on_toolbar_wheel(self, event):
        """Vertical wheel/trackpad motion scrolls the toolbar horizontally -- there's nothing to scroll vertically in a single-row strip."""
        if getattr(event, "num", None) == 4:
            self._toolbar_canvas.xview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._toolbar_canvas.xview_scroll(1, "units")
        else:
            self._toolbar_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def _update_navigation_button_states(self):
        """Enables/disables the Previous/Next Image toolbar buttons based on queue position -- Previous disabled on the first image, Next on the last.
        """
        if self._prev_image_button is None or self._next_image_button is None:
            return
        try:
            queue = self.cfg.image_queue
            pos = queue.index(self.cfg.CURRENT_IMAGE_PATH)
        except (ValueError, AttributeError):
            return  # no current image yet (very first frame) -- leave buttons as they are
        self._prev_image_button.configure(state=("disabled" if pos <= 0 else "normal"))
        self._next_image_button.configure(state=("disabled" if pos >= len(queue) - 1 else "normal"))

    # Prompt overrides: same return types as the base class, asked via a dialog.

    def _prompt_mode_choice(self):
        # Hides the main window while the choice is open, so this initial
        # dialog is the only thing on screen -- restored right after.
        self._tk_root.withdraw()
        dialog = _ModeChoiceDialog(self._tk_root, self._resolve_echo_twin_logo_path())
        self._tk_root.wait_window(dialog.top)
        self._tk_root.deiconify()
        return dialog.result or "1"

    def _prompt_pixel_scale(self, current_value):
        return simpledialog.askfloat(
            "Pixel-to-cm scale",
            f"New scale (current value: {current_value:.6f} cm/pixel):",
            parent=self._tk_root,
        )

    def _prompt_crack_numbers(self, current_text):
        return simpledialog.askstring(
            "Show cracks by number",
            "Crack numbers separated by ';' (e.g. 3;7;12).\nLeave empty to show every crack:",
            initialvalue=current_text, parent=self._tk_root,
        )

    def _prompt_ignore_options(self, current):
        # [I] / sidebar button: modal Tk dialog for shape and edges; None = cancelled.
        dialog = _IgnoreOptionsDialog(self._tk_root, current)
        self._tk_root.wait_window(dialog.top)
        return dialog.result

    def _prompt_calibration_distance(self):
        return simpledialog.askfloat(
            "Calibration",
            "Enter the real-world distance in centimeters (cm):",
            parent=self._tk_root,
        )

    def _report_fatal_error(self, message):
        # A double-clicked .app/.exe has no visible console -- shows a real
        # window instead so a fatal startup error is never silently invisible.
        messagebox.showerror("Crack Detector - Error", message, parent=self._tk_root)

    def _acknowledge_save_warning(self):
        # The default draws an invisible cv2 window and blocks on an unsafe
        # cv2.waitKey(0) -- a modal Tk dialog replaces it entirely.
        messagebox.showinfo(
            "Crack Detector",
            "[S] saves and stays on this image; [Enter] saves and moves "
            "to the next one.\n\nThe JSON goes to 'JSON files' and the masks to "
            "'Binary files'; the photo stays in 'Images'. To modify it again later, use Mode 2.",
            parent=self._tk_root,
        )

    def _keep_open_when_queue_empty(self):
        # The GUI stays open with no photo left, so Validation can still be run after a restart.
        return True

    def _prompt_queue_exhausted(self, finished_mode, target_mode, skipped_count):
        # Modal Yes/No instead of the terminal prompt: Yes switches mode, No exits.
        message = self._queue_end_message(finished_mode, target_mode, skipped_count)
        return bool(messagebox.askyesno("Crack Detector - Images finished", message, parent=self._tk_root))

    def _confirm_training_validation(self, message):
        return bool(messagebox.askyesno("Crack Detector - Validation", message, parent=self._tk_root))

    def _notify_training_validation_result(self, message):
        messagebox.showinfo("Crack Detector - Validation", message, parent=self._tk_root)

    def _notify_mode_switch_unavailable(self, target_mode):
        messagebox.showinfo(
            "Crack Detector",
            f"No images available for Mode {target_mode}.",
            parent=self._tk_root,
        )

    def _confirm_same_group_split(self, message):
        return bool(messagebox.askyesno("Crack Detector - Same building group", message, parent=self._tk_root))

    def _ask_auto_split_percent(self, default):
        return simpledialog.askstring("Crack Detector - Auto split",
                                      "Training / Validation / Test percentages (e.g. 70/15/15;\n"
                                      "80/0/20 when cross-validation replaces the validation set):",
                                      initialvalue=default, parent=self._tk_root)

    def _confirm_reset_pipeline(self, message):
        return bool(messagebox.askyesno("Crack Detector - Reset pipeline training", message, parent=self._tk_root))

    def _notify_reset_pipeline(self, message):
        messagebox.showinfo("Crack Detector - Reset pipeline training", message, parent=self._tk_root)

    def _confirm_auto_split(self, message):
        return bool(messagebox.askyesno("Crack Detector - Auto split", message, parent=self._tk_root))

    def _notify_auto_split(self, message):
        messagebox.showinfo("Crack Detector - Auto split", message, parent=self._tk_root)

    def _choose_image_index(self, names, current):
        dialog = _GoToImageDialog(self._tk_root, names, current)
        self._tk_root.wait_window(dialog.top)
        return dialog.result

    def _ask_cv_fold_count(self, default):
        return simpledialog.askinteger("Crack Detector - Cross-validation", "Number of folds (each building in one fold):",
                                       initialvalue=default, minvalue=2, maxvalue=20, parent=self._tk_root)

    def _confirm_cv_folds(self, message):
        return bool(messagebox.askyesno("Crack Detector - Cross-validation", message, parent=self._tk_root))

    def _notify_cv_folds(self, message):
        messagebox.showinfo("Crack Detector - Cross-validation", message, parent=self._tk_root)

    def _ask_save_before_split(self, set_name):
        message = (f"This photo has changes not saved yet.\n\nSave them before moving it to '{set_name}'?\n\n"
                   "Yes = save, then move\nNo = move it as last saved\nCancel = do nothing")
        return messagebox.askyesnocancel("Crack Detector - Unsaved changes", message, parent=self._tk_root)

    def _confirm_warn_uncertain(self, number, reasons):
        message = (f"Crack {number} is below the limit:\n  " + "\n  ".join(reasons) + "\n\n"
                   "Save it as uncertain?\n\nYes = uncertain (red, JSON and masks updated now)\n"
                   "No = keep it as a normal crack")
        return bool(messagebox.askyesno("Crack Detector - Warn uncertain", message, parent=self._tk_root))

    def _confirm_remove_uncertain(self, count):
        message = (f"Remove the {count} uncertain crack(s) [?] shown on screen?\n\n"
                   "The JSON and the binary masks are updated right away; [U] undoes it on screen.")
        return bool(messagebox.askyesno("Crack Detector - Remove uncertain cracks", message, parent=self._tk_root))

    def _confirm_crack_filter_removal(self, bldg_tag, json_paths, incompatible):
        # Replaces the default's invisible cv2 popup with a real, modal Tk dialog.
        message = (
            f"Group {bldg_tag[2:]}: {len(json_paths)} photos compared.\n"
            f"{len(incompatible)} incompatible crack(s) found "
            f"(indices: {sorted(incompatible)}).\n\n"
            "This will remove them from the JSON, mask, and overlay of "
            "EVERY photo in this group. Full details are in the status log.\n\n"
            "Remove them now?"
        )
        return messagebox.askyesno("Crack Detector - Confirm crack filter", message, parent=self._tk_root)

    # Display + event-loop integration

    def _patch_get_window_image_rect_if_needed(self, window_name):
        """Monkey-patches cv2.getWindowImageRect (once) so callers get the fixed canvas size regardless of the real (hidden) window's state."""
        if getattr(cv2, "_crackseg_get_window_image_rect_patched", False):
            return
        real_get_window_image_rect = cv2.getWindowImageRect

        def _patched_get_window_image_rect(name):
            if name == window_name:
                return (0, 0, CANVAS_W, CANVAS_H)
            return real_get_window_image_rect(name)

        cv2.getWindowImageRect = _patched_get_window_image_rect
        cv2._crackseg_get_window_image_rect_patched = True

    def _create_display_window(self, window_name):
        # A real (hidden) OpenCV window still needs to exist for geometry
        # queries; getWindowImageRect() is patched to return a fixed size.
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._patch_get_window_image_rect_if_needed(window_name)
        cv2.resizeWindow(window_name, 1, 1)
        cv2.moveWindow(window_name, -10000, -10000)

    def _display_frame(self, window_name, frame):
        self._canvas.update_frame(frame)
        # v1.3.0: the mode lines collected while drawing this frame go to the status bar.
        modes, self._frame_mode_texts = getattr(self, "_frame_mode_texts", []), []
        if hasattr(self, "_status_bar"):
            self._status_bar.set_modes(modes)

    def _put_hud_text(self, win_out, text, org, color, scale=None, thick=None):
        """GUI: tool/mode indicators (UNCERTAIN MODE, WARN UNCERTAIN, ERASER ...) go to the status bar, not the photo."""
        if not hasattr(self, "_frame_mode_texts"):
            self._frame_mode_texts = []
        self._frame_mode_texts.append(text)

    def _visible_canvas_top(self):
        """First frame row visible in the scrollable canvas, so the [J] panel stays on screen."""
        try:
            top = int(self._canvas.canvas.canvasy(0)) - int(self._canvas._content_offset[1])
        except (tk.TclError, AttributeError, TypeError, ValueError):
            return 0
        return max(0, min(CANVAS_H - 200, top))

    def _draw_bottom_hint_bar(self, win_out, hm):
        # The GUI's menu bar, toolbar and status bar already show these commands.
        return None

    def _toggle_help_menu(self):
        """GUI override: shows/hides the real PDF user manual instead of the text overlay render_scene() draws in headless mode."""
        if self._pdf_panel_visible:
            self._pdf_panel.hide()
            self._middle_row.pack(fill="both", expand=True)
            self._pdf_panel_visible = False
        else:
            if getattr(self, "_dataset_panel_visible", False):
                self._dataset_panel.hide()
                self._dataset_panel_visible = False
            self._middle_row.pack_forget()
            self._pdf_panel.show()
            self._pdf_panel_visible = True

    def _toggle_dataset_browser(self):
        """GUI override (code 31): shows/hides the Dataset browser over the whole image, like the manual."""
        if self._dataset_panel_visible:
            self._dataset_panel.hide()
            self._middle_row.pack(fill="both", expand=True)
            self._dataset_panel_visible = False
            return
        if self._pdf_panel_visible:
            self._pdf_panel.hide()
            self._pdf_panel_visible = False
        self._middle_row.pack_forget()
        self._dataset_panel.show()
        self._dataset_panel_visible = True

    def _toggle_fullscreen(self):
        """GUI override: toggles the real Tk window between filling the
        screen and its previous size. The base class's version targets
        the real, hidden OpenCV window GUI mode never draws to (see
        _create_display_window) -- calling it here made that window pop
        up for real, fullscreen and blank (USER REPORT)."""
        self.cfg.is_fullscreen = not self.cfg.is_fullscreen
        if self.cfg.is_fullscreen:
            self._pre_fullscreen_geometry = self._tk_root.geometry()
            screen_w, screen_h = self._get_screen_dimensions()
            self._tk_root.geometry(f"{screen_w}x{screen_h}+0+0")
        else:
            geometry = getattr(self, "_pre_fullscreen_geometry", None)
            if geometry:
                self._tk_root.geometry(geometry)

    def _pump_extra_events(self):
        try:
            self._update_navigation_button_states()
            self._sidebar.refresh_states(self.cfg, self._validation_available())
        except tk.TclError:
            pass
        try:
            self._tk_root.update()
        except tk.TclError:
            # Window was destroyed (app shutting down) -- nothing left to pump.
            pass
        return None

    def _wait_key(self, delay_ms):
        # Never calls the real cv2.waitKey() -- unsafe on macOS alongside a live Tk window.
        return -1


if __name__ == "__main__":
    GuiCrackSegmentation().run()
