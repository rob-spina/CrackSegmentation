# CrackSegmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22948009.svg)](https://doi.org/10.5281/zenodo.22948009)

Interactive Python / OpenCV tool for **manually segmenting and annotating structural cracks and detachments** on photos of building facades (e.g. drone survey imagery).

It produces [LabelMe](https://github.com/wkentaro/labelme)-compatible annotations and binary masks that can be used directly to train or fine-tune segmentation models.

<!-- Add a screenshot here: ![CrackSegmentation screenshot](docs/screenshot.png) -->

## Features

- **Two tracing tools**: cracks as polylines, detachments as closed polygons. Detachment contours are traced with an A* path search guided by the real image edges.
- **Two working modes**: new photos traced from scratch, or already-segmented photos reopened for correction.
- **Building groups**: photos of the same building are grouped automatically (SIFT features + RANSAC homography), so the same cracks can be tracked across different shots.
- **Crack import between photos**: project cracks from a previously saved photo of the same group onto the current one, with plausibility guards that silently drop implausible projections.
- **Snap and re-trace helpers**: elastic translation with automatic snap to the real fracture edge, and batch re-tracing of imported cracks.
- **Cross-photo consistency check**: cracks whose shape is incompatible between photos of the same group can be discarded (always after operator confirmation).
- **Training-mask generation**: the exported crack mask follows the two real edges of each crack, with a configurable band width. Per-stretch manual widening is available, and the band can be derived automatically from the image GSD.
- **Measurements**: crack length and detachment area once the image scale is calibrated.
- **Embedded PDF user manual** inside the main window (optional, requires PyMuPDF).

## Installation

Requires Python 3.10+ and Tkinter (bundled with most Python distributions; on Debian/Ubuntu: `sudo apt install python3-tk`).

```bash
git clone https://github.com/<your-username>/CrackSegmentation.git
cd CrackSegmentation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: PDF manual panel
```

## Tested Configurations

The application has been verified to launch and run correctly on the following real-world machines:

| OS | Device | Processor | RAM | Notes |
|----|--------|-----------|-----|-------|
| macOS Tahoe | MacBook Pro (MacBookPro17,1) | Apple M1 (8-core: 4P+4E) | 8 GB | Model no. MYD92T/A |
| Windows 11 Home | HP Laptop 15s-fq5xxx | Intel Core i7-1255U @ 1.70 GHz | 16 GB | 64-bit, x64 |
| Ubuntu 24.04.4 LTS | Victus by HP Gaming Laptop 16-r1xxx | Intel Core i7-14700HX | 16 GB | NVIDIA GeForce RTX 4050 Max-Q (AD107M) |

## Quick start

No sample photos are included in this repository (facade photos are typically private / third-party data). Before running the tool:

1. Create an `Images/` folder in the project root (the value of `IMAGE_FOLDER` in `config.py`) and place the facade photos you want to segment inside it.
2. Run the tool:

   ```bash
   python smart_segmentation.py
   ```

3. Trace cracks and detachments, then save with **S**, **Q** or **Enter**. The tool moves on to the next photo.

The complete list of shortcuts is available in the in-app help menu. The most used ones:

| Key | Action |
|-----|--------|
| `S` / `Q` / `Enter` | Save the current photo and move to the next one |
| `W` / `L` | Import cracks from a previously saved photo of the same building group |
| `T` | Elastic translation with automatic snap to the real fracture edge |
| `V` | Batch re-trace of imported cracks onto the real edges (with plausibility check) |
| `G` | Compare cracks across photos of the same group and discard incompatible ones (also runs automatically after each save) |
| `B` | Manually assign a building group number |
| `A` | Widen the training mask on a single crack stretch; refine with `+` / `-` and `[` `]` `{` `}` |
| `K` | Enter the image scale manually |
| `R` | Redo |

## Output files

For every saved photo `<name>`:

| File | Content |
|------|---------|
| `<name>.json` | LabelMe annotation (`linestrip` shapes for cracks, polygons for detachments), with the image embedded as base64 |
| `<name>-crack_mask.png` | Binary crack mask, single channel, same resolution as the photo (0 = background, 255 = crack) |
| `<name>-detachment_mask.png` | Binary filled-area mask of the detachments (same format) |
| `<name>-seg.jpg` | Colored overlay for visual inspection |

Mask files are written only when the photo contains at least one active element. Saved photos are archived in `already processed images/`.

## Configuration

Main tunables live in `config.py`:

| Constant | Meaning |
|----------|---------|
| `IMAGE_FOLDER` | Folder containing the photos to segment |
| `CRACK_MASK_AUTO_EDGE_ENABLED` | Enable automatic detection of the two crack edges in the exported mask |
| `CRACK_AUTO_EDGE_MAX_OFFSET_PX` | Maximum offset per side (pixels) of the automatic crack band |
| `CRACK_MASK_DILATION_PX` | Uniform dilation of the exported crack mask |
| `IMAGE_GSD_MM_PER_PX` | Ground sample distance in mm/pixel. `0.0` = not set (the two pixel constants above are used as-is) |
| `CRACK_TARGET_PHYSICAL_WIDTH_MM` | Physical crack width the exported band should represent once the GSD is known |

When `IMAGE_GSD_MM_PER_PX` is set, the band width in pixels is derived automatically from it, with a safety ceiling against input errors.

## Companion tools

**`compare_and_filter_cracks.py`** compares cracks across two or more photos of the same building group and removes those whose shape deviates beyond a configurable threshold. It updates the JSON (a `.bak` backup is created), the binary mask and, if the original photo can be found, the overlay. Use `--dry-run` to preview the changes and `--images-dir` to point to the original photos.

```bash
python compare_and_filter_cracks.py --help
```

**`crack_segmentation_evaluator.py`** evaluates the quality of segmentation outputs, on a single image pair or in batch mode, with JSON / CSV reports.

```bash
python crack_segmentation_evaluator.py --help
```

## Project layout

```
CrackSegmentation/
├── Images/                            # your input photos — not included, create it locally (see Quick start)
├── smart_segmentation.py              # main interactive tool
├── config.py                          # tunable constants and shared state
├── crack_segmentation_gui.py          # GUI layer (Tkinter window, PDF manual panel)
├── compare_and_filter_cracks.py       # cross-photo crack consistency filter
├── crack_segmentation_evaluator.py    # segmentation quality evaluator
├── tests_support/                     # test doubles (fake_tkinter.py, fake_fitz.py, ...)
├── .github/                           # CI workflow, issue and pull request templates
├── CONTRIBUTING.md
├── requirements.txt
└── requirements-optional.txt
```

## Notes

- Code comments, docstrings, on-screen (HUD) text, and console messages are all in English.
- The PDF manual panel degrades gracefully: without PyMuPDF (or without the PDF file) the rest of the application works normally.

## Contributing

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the code conventions used in this project.

## Authors & Credits

This software was developed within the framework of the **ECHO-TWIN** project by the INAF (Istituto Nazionale di Astrofisica) Research Group.

| Name | Contribution |
|------|--------------|
| **Roberto Spina** | Code Architecture, Software Design & Lead Development |
| **Fabio Vitello** | Project Coordination & Supervision |
| **Eva Sciacca** | Project Coordination & Supervision |
| **Leonardo Pelonero** | General Support |
| **Salvatore Scavo** | General Support |

**AI Disclosure.** This application utilizes artificial intelligence models for certain stages of the workflow, specifically for analysis, technical content generation, and development support. The AI components do not replace scientific validation nor do they influence results in a deterministic manner: they are used as assistance tools, not as sources of truth.

## License

Released under the [MIT License](LICENSE).

The optional dependency [PyMuPDF](https://github.com/pymupdf/PyMuPDF) is distributed by Artifex under AGPL-3.0 or a commercial license. It is not included in this repository and is not required to run the tool. If you redistribute a build that bundles it, check that the AGPL terms are met.
