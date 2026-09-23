# Contributing to CrackSegmentation

Thanks for your interest! Bug reports, ideas and pull requests are all welcome.

## Reporting bugs and requesting features

Use the issue templates. Please **never attach private or third-party photos**
(facade photos and LabelMe JSON files embed the full image). Describe the problem,
or reproduce it with a synthetic image or one you are allowed to share.

## Development setup

```bash
git clone https://github.com/<your-username>/CrackSegmentation.git
cd CrackSegmentation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: PDF manual panel
```

Run the test suite (it is fully headless, Tkinter and PyMuPDF are replaced by the
doubles in `tests_support/`):

```bash
python <your_test_runner>.py      # TODO: replace with the actual test entry point
```

## Code conventions

- **Comments and docstrings in English**, at most two lines each.
- **Small, single-responsibility functions.** A function longer than about 15 lines
  should be split into helpers; 10 to 15 lines is acceptable. The only documented
  exception is the hot search loop of `a_star_pathfinding`, kept inline on purpose to
  avoid call overhead on a performance-critical path.
- **Refactoring is mechanical**: restructure, never rewrite logic in the same change.
- **Performance changes must be behavior-identical**, and verified with tests
  (for example randomized, bit-exact comparisons against the previous implementation).
- **Run the full test suite after every change**, not only at the end.
- **New tests follow the existing patterns.** For example `tests_support/fake_fitz.py`
  mirrors `fake_tkinter.py`. Every fixed bug gets a regression test.
- **Bugs found along the way** are fixed and documented in the same pull request,
  ideally in a separate commit.
- **Tunable values go in `config.py`**, with a short comment. Note that modules using
  `from config import *` hold their own copies of the names: code that changes a constant
  at runtime must update both `config.X` and the bare name in the consuming module.
- **User-facing text** (on-screen HUD and console messages) is currently in Italian.
  Please open an issue before changing or translating it, so the interface stays consistent.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep the pull request focused: one fix or feature at a time.
3. Fill in the pull request template. CI (syntax check and test suite) must pass.
4. Do not commit photos, masks, JSON annotations or other data (the `.gitignore` covers
   the common cases, but double-check with `git status`).

## License

By contributing you agree that your contributions are released under the
[MIT License](LICENSE) of this project.
