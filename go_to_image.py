"""go_to_image.py
v1.0.9 [Go to image] (toolbar, File menu, code 29), mixed into CrackSegmentation (smart_segmentation.py).
Jumps straight to the first image, the last image or a chosen image of the queue, in Mode 1 and Mode 2,
without saving (like Previous / Next Image). The chosen image is picked from a list showing 10 names at a
time: choosing the 10th name of a page shows the next 10, choosing the 1st one of a page (after the first)
shows the previous 10; buttons page back and forth as well. The paging lives in PagedChooser so the GUI
dialog and the terminal fallback behave the same.
"""
import os


GO_TO_IMAGE_CODE = 29
PAGE_SIZE = 10


class PagedChooser:
    """10-at-a-time view over a list of names (0-based indices into it)."""

    def __init__(self, names, start_index=0, page_size=PAGE_SIZE):
        self.names = list(names)
        self.page_size = page_size
        self.page = self.page_of(start_index)

    @property
    def page_count(self):
        return max(1, -(-len(self.names) // self.page_size))

    def page_of(self, index):
        return min(max(0, index), max(0, len(self.names) - 1)) // self.page_size

    def page_range(self):
        """(first, last) 0-based indices shown on the current page."""
        first = self.page * self.page_size
        return first, min(len(self.names), first + self.page_size) - 1

    def labels(self):
        """'n. name' entries of the current page (n counted from 1, as on the HUD)."""
        first, last = self.page_range()
        return [f"{i + 1}. {self.names[i]}" for i in range(first, last + 1)]

    def next_page(self):
        self.page = min(self.page_count - 1, self.page + 1)

    def previous_page(self):
        self.page = max(0, self.page - 1)

    def choose(self, position_on_page):
        """0-based queue index of the entry chosen on the page; choosing the last entry of a full page moves
        the list to the next 10, choosing the first entry of a later page moves it to the previous 10."""
        first, last = self.page_range()
        index = min(last, first + max(0, position_on_page))
        if index == last and last - first + 1 == self.page_size and self.page < self.page_count - 1:
            self.next_page()
        elif index == first and self.page > 0:
            self.previous_page()
        return index

    @staticmethod
    def index_from_label(label):
        """'12. DJI_0349.JPG' -> 11; None when the text does not start with a number."""
        head = str(label).split(".", 1)[0].strip()
        return int(head) - 1 if head.isdigit() else None


class GoToImageMixin:
    """[Go to image]. Relies on CrackSegmentation's cfg and queue bookkeeping."""

    def _queue_names(self):
        return [os.path.basename(p) for p in self.cfg.image_queue]

    def _current_queue_index(self):
        try:
            return self.cfg.image_queue.index(self.cfg.CURRENT_IMAGE_PATH)
        except ValueError:
            return 0

    def _choose_image_index(self, names, current):
        """0-based index of the image to show, or None if cancelled. A GUI wrapper overrides this with a
        dialog (first / last / paged list); this terminal version takes 'f', 'l' or a number."""
        chooser = PagedChooser(names, current)
        print("\n".join(chooser.labels()))
        try:
            answer = input(f"Go to image: f = first, l = last, or a number 1-{len(names)} [ENTER = cancel]: ")
        except EOFError:
            return None
        answer = answer.strip().lower()
        if answer in ("f", "first"):
            return 0
        if answer in ("l", "last"):
            return len(names) - 1
        return int(answer) - 1 if answer.isdigit() and 1 <= int(answer) <= len(names) else None

    def _handle_go_to_image_key(self):
        """Code 29 [Go to image]. True when another image was chosen: the image loop exits without saving
        and run()'s queue walk continues from the chosen position."""
        names = self._queue_names()
        if self.cfg.queue_idle or not names:
            print(" [GO TO IMAGE] No image in the queue.")
            return False
        current = self._current_queue_index()
        target = self._choose_image_index(names, current)
        if target is None or not (0 <= target < len(names)) or target == current:
            return False
        print(f"\n[GO TO IMAGE] Image {target + 1}/{len(names)} ({names[target]}), WITHOUT saving the current one.")
        self.cfg.translate_state["active"] = False
        self.cfg.translate_state["idx"] = None
        self.cfg.translate_state["start"] = None
        self.cfg.navigate_target = target
        return True
