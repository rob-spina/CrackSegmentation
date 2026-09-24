"""fake_fitz.py

A minimal stand-in for PyMuPDF's `fitz` module, used only to test
PdfManualPanel's rendering pipeline (crack_segmentation_gui.py) without
requiring the real pymupdf package to be installed in this sandbox.

Everything downstream of fitz.open()/page.get_pixmap() -- PIL.Image.
frombytes(), the (stubbed) ImageTk.PhotoImage(), canvas placement, and
scrollregion sizing -- runs for real; only the PDF decoding itself is
faked, exactly the same "fake only what the sandbox can't provide"
principle as tests_support/fake_tkinter.py.

Install into sys.modules['fitz'] BEFORE calling PdfManualPanel.show(),
since it does a local `import fitz` (deliberately not a top-of-file
import -- see its own docstring for why).
"""


class FakePixmap:
    """A tiny, real, renderable RGB image (solid gray) -- real bytes,
    so Image.frombytes() downstream produces a genuine PIL image, not
    a mock."""

    def __init__(self, width=40, height=60):
        self.width = width
        self.height = height
        self.alpha = False
        self.samples = bytes([180, 180, 180]) * (width * height)


class FakePage:
    def __init__(self, width=40, height=60):
        self._width = width
        self._height = height
        self.get_pixmap_calls = []

    def get_pixmap(self, dpi=110):
        self.get_pixmap_calls.append(dpi)
        return FakePixmap(self._width, self._height)


class FakeDocument:
    """Iterable of FakePage, mirroring fitz.Document's own page
    iteration. n_pages defaults to 2 -- enough to verify multi-page
    stacking without slowing tests down."""

    def __init__(self, n_pages=2):
        self.pages = [FakePage() for _ in range(n_pages)]
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self):
        self.closed = True


class FakeFitzModule:
    """Swap in for the real `fitz` module. open_should_fail/
    n_pages_to_open let a test control the two failure paths
    PdfManualPanel._load_and_render() guards against (missing file /
    corrupt PDF), and the multi-page case."""

    def __init__(self, n_pages=2, open_should_raise=None):
        self.n_pages = n_pages
        self.open_should_raise = open_should_raise
        self.opened_paths = []
        self.last_document = None

    def open(self, path):
        self.opened_paths.append(path)
        if self.open_should_raise is not None:
            raise self.open_should_raise
        self.last_document = FakeDocument(self.n_pages)
        return self.last_document
