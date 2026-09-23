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


def tk_key_event_to_code(event):
    """Translates one Tk <KeyPress> event into the integer code process_keypress() expects, matching what cv2.waitKey() would return. None if unmapped.
    """
    if event.keysym in _SPECIAL_KEYSYM_TO_CODE:
        return _SPECIAL_KEYSYM_TO_CODE[event.keysym]
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

    def __init__(self, master, app):
        self.app = app
        self._build_canvas_and_scrollbar(master)

        self._image_id = None
        self._current_photo = None  # keeps a live reference -- Tk drops the image otherwise
        self._content_offset = (0, 0)

        self._bind_canvas_events()

    def _compute_content_offset(self):
        """Offset (x, y) to center the fixed 1200x900 photo within a larger viewport. Zero in either axis once the viewport isn't bigger."""
        viewport_w = max(1, self.canvas.winfo_width())
        viewport_h = max(1, self.canvas.winfo_height())
        x_offset = max(0, (viewport_w - CANVAS_W) // 2)
        y_offset = max(0, (viewport_h - CANVAS_H) // 2)
        return x_offset, y_offset

    def update_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_image)
        self._content_offset = self._compute_content_offset()
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
        x_offset, y_offset = getattr(self, "_content_offset", (0, 0))
        cx = int(self.canvas.canvasx(event.x)) - x_offset
        cy = int(self.canvas.canvasy(event.y)) - y_offset
        return cx, cy

    def _on_press(self, event):
        self.canvas.focus_set()
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    def _on_release(self, event):
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    def _on_motion(self, event):
        x, y = self._canvas_xy(event)
        self.app.mouse_callback(cv2.EVENT_MOUSEMOVE, x, y, 0, None)

    def _on_enter(self, event):
        self.canvas.focus_set()
        # bind_all (not a per-widget bind) makes wheel scrolling reliable across platforms.
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _on_leave(self, event):
        # Releases the global bindings once the mouse leaves this canvas.
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        # While the on-screen [?] help guide is open, the wheel scrolls its
        # command list instead, via the same synthetic Up/Down key codes.
        if getattr(self.app.cfg, "show_help_menu", False):
            scrolling_up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
            self.app._pending_keys.append(0 if scrolling_up else 1)
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


# Status strip: fixed panel at the bottom showing console output continuously
# (a double-clicked .app/.exe otherwise has no visible console).

# USER REQUEST: real, clickable shortcut buttons (not just a static list) for
# the commands that aren't already toolbar buttons -- icons match the ones
# already used for the same commands in _TOOL_COMMANDS/_VIEW_COMMANDS/etc.
SIDEBAR_WIDTH = 210
_SIDEBAR_SHORTCUTS = [
    ("\u26A1", "Crack tool", "C", ord('c')),
    ("\u25A6", "Detachment tool", "D", ord('d')),
    ("\U0001F535", "Toggle blue overlay", "N", ord('n')),
    ("\U0001F4CD", "Toggle markers", "Space", ord(' ')),
    ("\U0001F504", "Retrace cracks", "V", ord('v')),
    ("\U0001F9F2", "Snap / translate", "T", ord('t')),
    ("\u2194", "Width-edit mode", "A", ord('a')),
    ("\U0001F3F7", "Assign building", "B", ord('b')),
    ("\U0001F50D", "Compatibility check", "G", ord('g')),
    ("\U0001F4D0", "Set scale", "K", ord('k')),
    ("\u26F6", "Fullscreen", "F", ord('f')),
    ("\U0001F50E", "Reset zoom", "0", ord('0')),
    ("\U0001F500", "Switch Mode", "", 6),  # no physical key -- button-only action
]


class ShortcutsSidebar:
    """Panel of real, clickable shortcut buttons to the right of the
    canvas, for the commands not already on the toolbar. Vertically
    scrollable so a button is never simply unreachable on a shorter
    window (USER REPORT -- the last one used to be clipped)."""

    def __init__(self, master, app):
        self.app = app
        self.frame = ttk.Frame(master, width=SIDEBAR_WIDTH)
        self.frame.pack(side="right", fill="y", padx=(4, 2), pady=2)
        self.frame.pack_propagate(False)  # keeps a fixed width regardless of content

        tk.Label(self.frame, text="More Shortcuts", font=("TkDefaultFont", 10, "bold"),
                 anchor="w").pack(anchor="w", fill="x", padx=8, pady=(8, 6))

        canvas = tk.Canvas(self.frame, highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")
        self._canvas = canvas

        inner = ttk.Frame(canvas)
        canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Enter>", lambda e: self._bind_wheel())
        canvas.bind("<Leave>", lambda e: self._unbind_wheel())

        self.buttons = []
        for icon, label, key, code in _SIDEBAR_SHORTCUTS:
            self.buttons.append(self._add_row(inner, icon, label, key, code))

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
        return btn


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
            "     from the 'Images' folder", "1",
        )
        self._make_option_button(
            body, "2)  Reload ALREADY SEGMENTED images",
            "     from 'already processed images', to edit or correct", "2",
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


# Menu bar command tables -- (label, key_code) matching process_keypress()'s
# real key codes. Labels use a Unicode symbol prefix (plain text, no bitmap icons).

_FILE_COMMANDS = [
    ("\U0001F4BE Save and go to next\tEnter", 13),
    ("\U0001F4BE Save and go to next\tS", ord('s')),
    ("\U0001F4E5 Import from previous session (projection)\tW", ord('w')),
    ("\U0001F517 Import - automatic alignment\tL", ord('l')),
    # USER REQUEST: browse the queue without exporting/archiving anything --
    # unlike Save/Enter/S above, neither of these ever saves.
    ("\u25C0 Previous Image (no save)", 5),
    ("\u25B6 Next Image (no save)", 4),
    # USER REQUEST: switch between "new images" and "reload segmented" data
    # sources without restarting the app (code 6, see process_keypress()).
    ("\U0001F504 Switch Mode (1 \u2194 2)", 6),
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
    ("\u2705 Close detachment polygon\tY", ord('y')),
    ("\u270E Toggle edit mode\tE", ord('e')),
    ("\U0001F9F2 Automatic snap / translate\tT", ord('t')),
    ("\U0001F504 Retrace imported cracks onto real edge\tV", ord('v')),
    ("\U0001F50D Building-group compatibility check\tG", ord('g')),
    ("\U0001F3F7 Manually assign building group\tB", ord('b')),
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
    ("\U0001F520 Toggle HUD size\tH", ord('h')),
    ("\U0001F4CD Toggle markers\tSpace", ord(' ')),
    ("\U0001F535 Toggle blue crack overlay\tN", ord('n')),
    ("\u2139 Toggle Info (file/length/building status)\tJ", ord('j')),
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
        self._status_bar = StatusBar(self._tk_root)
        placeholder = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self._canvas.update_frame(placeholder)

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
        canvas_and_sidebar_w = CANVAS_W + 24 + SIDEBAR_WIDTH
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
        return [
            ("\u25C0 Previous Image", 5),
            ("\u25B6 Next Image", 4),
            ("\U0001F4BE Save [S]", ord('s')),
            ("\U0001F4BE Save & next [Enter]", 13),
            ("\U0001F4E5 Import W", ord('w')),
            ("\U0001F517 Import L", ord('l')),
            ("\u21B6 Undo [U]", ord('u')),
            ("\u21B7 Redo [R]", ord('r')),
            ("\U0001F5D1 Clear [X]", ord('x')),
            ("\u2139 Info [J]", ord('j')),
            ("\u2753 Manual [?]", ord('?')),
        ]

    def _create_one_toolbar_button(self, bar, label, code):
        """Creates one toolbar button, tracking it as the Previous/Next Image reference if it is one."""
        # takefocus=0: stops these buttons from grabbing keyboard focus on click.
        btn = ttk.Button(bar, text=label, command=lambda c=code: self._send(c), takefocus=0)
        btn.pack(side="left", padx=2)
        # Greyed out at the start/end of the queue -- see _update_navigation_button_states().
        if "Previous Image" in label:
            self._prev_image_button = btn
        elif "Next Image" in label:
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
            "Saving this image will automatically proceed to the next "
            "file in the queue.\n\nTo modify this image again later, "
            "restart the program in Mode 2.",
            parent=self._tk_root,
        )

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

    def _toggle_help_menu(self):
        """GUI override: shows/hides the real PDF user manual instead of the text overlay render_scene() draws in headless mode."""
        if self._pdf_panel_visible:
            self._pdf_panel.hide()
            self._middle_row.pack(fill="both", expand=True)
            self._pdf_panel_visible = False
        else:
            self._middle_row.pack_forget()
            self._pdf_panel.show()
            self._pdf_panel_visible = True

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
