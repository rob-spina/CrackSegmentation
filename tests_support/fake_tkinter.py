"""Minimal stand-in for the parts of tkinter/ttk/simpledialog/messagebox
and PIL.ImageTk that crack_segmentation_gui.py touches, used ONLY to
smoke-test the app's wiring logic in a sandbox that has no real Tk/
display available. This is not a tkinter (or PIL.ImageTk) reimplementation
-- it just records what was called, and lets tests fire synthetic
<Button-1>/<Motion>/<KeyPress>/etc. events through the exact same bind()
callbacks the real widgets would invoke.

PIL.Image itself is NOT stubbed -- it needs no real Tk interpreter, so
tests exercise the REAL cv2.cvtColor()/PIL.Image.fromarray() conversion
in EmbeddedCanvas.update_frame(). Only the final ImageTk.PhotoImage step
(which needs a genuine Tk app context) is faked.
"""
import sys
import types


class _Event:
    """A stand-in for a real tkinter event object -- only the attributes
    this codebase actually reads (x, y, keysym, char, num, delta) are
    supported."""
    def __init__(self, x=0, y=0, keysym="", char="", num=None, delta=0):
        self.x = x
        self.y = y
        self.keysym = keysym
        self.char = char
        self.num = num
        self.delta = delta


class _Widget:
    def __init__(self, master=None, **kw):
        self.master = master
        self.kw = kw
        self.children = []
        self._bindings = {}
        if master is not None and hasattr(master, "children"):
            master.children.append(self)

    def pack(self, **kw): return self
    def pack_forget(self): return self
    def pack_propagate(self, *a, **kw): return self
    def grid(self, **kw): return self
    def grid_rowconfigure(self, *a, **kw): return self
    def grid_columnconfigure(self, *a, **kw): return self
    def config(self, **kw): self.kw.update(kw); return self
    def configure(self, **kw): return self.config(**kw)

    def winfo_width(self):
        # Real Tk only knows its actual on-screen width once mapped/
        # laid out; tests set widget._test_width directly to simulate
        # a specific window size (see PdfManualPanel's centering logic).
        return getattr(self, "_test_width", 1)

    def winfo_height(self):
        return getattr(self, "_test_height", 1)

    def bind(self, sequence=None, func=None, **kw):
        if sequence is not None:
            self._bindings[sequence] = func
        return self

    def fire(self, sequence, **event_kwargs):
        """Test helper: invokes whatever handler was bind()-ed for
        `sequence` (e.g. "<Button-1>") with a synthetic event carrying
        the given x/y/keysym/char."""
        handler = self._bindings.get(sequence)
        if handler is None:
            raise AssertionError(f"nothing bound for {sequence!r}")
        return handler(_Event(**event_kwargs))

    def bind_all(self, sequence=None, func=None, **kw):
        # Simplified for testing purposes: registers into the SAME
        # per-instance _bindings dict `bind()` uses (a real Tk bind_all()
        # is process-global, but fire() only needs to find the handler on
        # this instance to simulate the event).
        if sequence is not None:
            self._bindings[sequence] = func
        return self

    def unbind_all(self, sequence=None, **kw):
        if sequence is not None:
            self._bindings.pop(sequence, None)
        return self
    def geometry(self, *a, **kw):
        if a:
            self.kw["geometry_arg"] = a[0]
            return None
        return self.kw.get("geometry_arg", "1x1+0+0")
    def minsize(self, *a, **kw): return self
    def protocol(self, name, handler):
        setattr(self, "_protocol_" + name.replace(".", "_"), handler)
        return self
    def title(self, *a, **kw): return self
    def withdraw(self): self.withdrawn = True
    def deiconify(self): self.withdrawn = False
    def destroy(self): self.destroyed = True
    def resizable(self, *a, **kw): return self
    def grab_set(self): return self
    def focus_force(self): return self
    def focus_set(self): self.has_focus = True
    def lift(self): return self
    def update(self): self.updated = getattr(self, "updated", 0) + 1
    def update_idletasks(self): return self
    def winfo_screenwidth(self): return 1440
    def winfo_screenheight(self): return 900
    def winfo_reqwidth(self): return 1200
    def winfo_reqheight(self): return 900
    def wait_window(self, w=None): return self
    def yview(self, *a, **kw): return self
    def yview_scroll(self, *a, **kw): return self
    def xview(self, *a, **kw): return self
    def xview_scroll(self, *a, **kw): return self
    def canvasx(self, x, *a, **kw): return x
    def canvasy(self, y, *a, **kw): return y
    def bbox(self, *a, **kw): return (0, 0, 0, 0)
    def create_window(self, *a, **kw): return self
    def set(self, *a, **kw): return self


class Tk(_Widget):
    pass


class Toplevel(_Widget):
    pass


class Frame(_Widget):
    pass


class LabelFrame(_Widget):
    pass


class Label(_Widget):
    pass


class Button(_Widget):
    def __init__(self, master=None, text="", command=None, **kw):
        super().__init__(master, text=text, **kw)
        self.text = text
        self.command = command
        if master is not None and hasattr(master, "buttons"):
            master.buttons.append(self)


# The real ttk::button widget accepts a materially SMALLER option set than
# classic tk.Button -- e.g. no wraplength, no bg/fg/relief/font -- passing
# one of those raises a real TclError at runtime (a silent app-launch
# crash for a double-clicked .app, with no console to show the traceback).
# Unlike the shared, unvalidated Button above, this catches that class of
# bug in tests instead of only on a real Mac.
_TTK_BUTTON_VALID_OPTIONS = {
    "master", "text", "textvariable", "command", "width", "underline",
    "image", "compound", "default", "state", "padding", "cursor",
    "takefocus", "style", "class_",
}


class TtkButton(Button):
    def __init__(self, master=None, text="", command=None, **kw):
        bad = set(kw) - _TTK_BUTTON_VALID_OPTIONS
        if bad:
            raise TclError(f"unknown option \"-{next(iter(bad))}\" (ttk::button has no such option)")
        super().__init__(master, text=text, command=command, **kw)


class Canvas(_Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.width = kw.get("width")
        self.height = kw.get("height")
        self._items = {}
        self._next_id = 1

    def create_image(self, x, y, anchor=None, image=None, **kw):
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = {"x": x, "y": y, "anchor": anchor, "image": image}
        return item_id

    def create_text(self, x, y, anchor=None, text="", **kw):
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = {"x": x, "y": y, "anchor": anchor, "text": text, **kw}
        return item_id

    def itemconfig(self, item_id, **kw):
        self._items.setdefault(item_id, {}).update(kw)

    itemconfigure = itemconfig

    def coords(self, item_id, *args):
        if not args:
            item = self._items.get(item_id, {})
            return [item.get("x", 0), item.get("y", 0)]
        x, y = args[0], args[1]
        self._items.setdefault(item_id, {})["x"] = x
        self._items[item_id]["y"] = y

    def create_window(self, x, y, anchor=None, window=None, **kw):
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = {"x": x, "y": y, "anchor": anchor, "window": window}
        return item_id

    def bbox(self, item_id_or_tag):
        return (0, 0, self.width or 0, self.height or 0)

    def xview_scroll(self, number, what): pass
    def xview(self, *a): return (0.0, 1.0)
    def yview_scroll(self, number, what): pass
    def yview(self, *a): return (0.0, 1.0)


class Text(_Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.content = ""

    def configure(self, **kw):
        return self.config(**kw)

    def insert(self, index, text):
        self.content += text

    def see(self, index):
        return self

    def get(self, start, end=None):
        return self.content


class Menu(_Widget):
    def __init__(self, master=None, tearoff=0, **kw):
        super().__init__(master, **kw)
        self.items = []  # (kind, label, command_or_submenu)

    def add_command(self, label=None, command=None, **kw):
        self.items.append(("command", label, command))

    def add_cascade(self, label=None, menu=None, **kw):
        self.items.append(("cascade", label, menu))

    def add_separator(self, **kw):
        self.items.append(("separator", None, None))


class TclError(Exception):
    pass


class Scrollbar(_Widget):
    def __init__(self, master=None, orient=None, command=None, **kw):
        super().__init__(master, **kw)
        self.orient = orient
        self.command = command


ttk = types.SimpleNamespace(Frame=Frame, LabelFrame=LabelFrame, Button=TtkButton, Label=Label, Scrollbar=Scrollbar)


class _simpledialog:
    last_call = None
    next_return = 42.0

    @staticmethod
    def askfloat(title, prompt, parent=None, **kw):
        _simpledialog.last_call = (title, prompt)
        return _simpledialog.next_return


simpledialog = _simpledialog


class _messagebox:
    last_call = None

    @staticmethod
    def showerror(title, message, parent=None, **kw):
        _messagebox.last_call = ("error", title, message)

    @staticmethod
    def showinfo(title, message, parent=None, **kw):
        _messagebox.last_call = ("info", title, message)

    @staticmethod
    def showwarning(title, message, parent=None, **kw):
        _messagebox.last_call = ("warning", title, message)

    @staticmethod
    def askyesno(title, message, parent=None, **kw):
        _messagebox.last_call = ("askyesno", title, message)
        return True


messagebox = _messagebox

# register as the 'tkinter' package (with submodules) BEFORE the real
# import happens
mod = types.ModuleType("tkinter")
mod.Tk = Tk
mod.Toplevel = Toplevel
mod.Frame = Frame
mod.LabelFrame = LabelFrame
mod.Label = Label
mod.Button = Button
mod.Canvas = Canvas
mod.Text = Text
mod.Menu = Menu
mod.TclError = TclError
mod.ttk = ttk
mod.simpledialog = simpledialog
mod.messagebox = messagebox
sys.modules["tkinter"] = mod
sys.modules["tkinter.ttk"] = ttk
sys.modules["tkinter.simpledialog"] = simpledialog
sys.modules["tkinter.messagebox"] = messagebox


# ---------------------------------------------------------------------------
# PIL.ImageTk stub -- PhotoImage() normally needs a real live Tk
# interpreter under it (it registers the image with the Tcl runtime), so
# it can't work against the widget stubs above. PIL.Image itself needs no
# such thing, so it is left completely real/untouched: tests exercise the
# genuine cv2.cvtColor()/Image.fromarray() conversion, only the final
# "hand this to Tk" step is faked here.
# ---------------------------------------------------------------------------

class _FakePhotoImage:
    _created = []  # test helper: every PIL.Image handed to PhotoImage(), in order

    def __init__(self, pil_image=None, **kw):
        self.pil_image = pil_image
        _FakePhotoImage._created.append(pil_image)


_imagetk_mod = types.ModuleType("PIL.ImageTk")
_imagetk_mod.PhotoImage = _FakePhotoImage
sys.modules["PIL.ImageTk"] = _imagetk_mod
ImageTk = _imagetk_mod
