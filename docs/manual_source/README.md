# Manual source

`manual.html` is the source of truth for `Crack_Segmentation_User_Manual.pdf`
(the PDF shipped in the repo root and embedded in the app's PDF manual panel).
The PDF is rendered from this HTML via headless Chromium (Playwright) — it is
not written or exported from a word processor.

Edit `manual.html` (and the images in `logos/` if needed), then regenerate
the PDF as described below. **Never edit the PDF directly** — changes will be
lost the next time it's regenerated from this source.

## Regenerating the PDF

Requires Python 3 and Playwright with the Chromium browser:

```bash
pip install playwright
playwright install chromium
```

Then, from this folder (`docs/manual_source/`):

```bash
python3 - <<'EOF'
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HTML_PATH = Path("manual.html").resolve()
OUTPUT_PATH = Path("../../Crack_Segmentation_User_Manual.pdf").resolve()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{HTML_PATH}")
        await page.pdf(
            path=str(OUTPUT_PATH),
            print_background=True,
            format="A4",
            scale=297 / 364,   # see "Why the scale factor" below — do not drop this
        )
        await browser.close()

asyncio.run(main())
EOF
```

This writes the PDF to the repository root, overwriting the existing file.

## Why the scale factor

The cover page is authored at **364 mm** of content height (to fit the full
gradient background, logos and funding-partner strip), while `page.pdf()`
prints onto standard **A4 (297 mm)** paper. Without compensating for the
difference, the bottom of the cover page — including the funding logos strip
— gets silently clipped off the printed page with no error or warning.

The fix is to pass `scale=297/364` (≈ 0.816) to `page.pdf()`. This shrinks the
whole page uniformly so the full 364 mm of content fits inside the 297 mm A4
page, instead of letting Chromium clip it. If the cover page's CSS height
(`.cover` or equivalent, near the top of `manual.html`) is ever changed, this
ratio must be recalculated (`297 / <new height in mm>`) and updated in the
render script above.

## After regenerating

1. Open the new PDF and check the page count and every page visually — a
   change in the credits box, the affiliations line, or any padding can push
   content onto an extra (near-blank) trailing page. If that happens, tighten
   padding/margins in the `<style>` block (search for `.credits-box`,
   `.affiliations`, `.ai-box`, `.footer-note`) rather than the `scale` factor.
2. Commit `manual.html` (and `logos/` if changed) together with the
   regenerated `Crack_Segmentation_User_Manual.pdf` in the same commit, so
   the two never drift apart.

## Files

```
manual_source/
├── manual.html      # full HTML + inline CSS source for the 4-page manual
├── logos/           # cover page images (INAF, ECHO-TWIN, funding strip, gradient, rule)
└── README.md        # this file
```
