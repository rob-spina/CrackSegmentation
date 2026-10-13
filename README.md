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

## Downloads

Pre-built, self-contained packages are available on the [Releases page](https://github.com/rob-spina/CrackSegmentation/releases) — no separate Python installation required on the target machine.

**macOS** — `CrackSegmentation.dmg`
Open it in Finder, drag `CrackSegmentation` onto the Applications shortcut, eject, then launch from Applications/Launchpad. Don't run the `.dmg` itself as a script. It's unsigned (no paid Apple Developer certificate): if Gatekeeper blocks the first launch, allow it via **System Settings › Privacy & Security › "Open Anyway"**, or right-click the app and choose **Open**.

**Windows** — `CrackSegmentation-Setup.exe`
Run the installer and follow the wizard. It's unsigned: if SmartScreen flags it, click **More info › Run anyway** — expected for independently built software, not a sign of a problem.

**Linux (Debian/Ubuntu, amd64)** — `cracksegmentation_*_amd64.deb`
```bash
sudo apt install ./cracksegmentation_*_amd64.deb
# or: sudo dpkg -i ./cracksegmentation_*_amd64.deb && sudo apt-get install -f
```
Then launch via the Applications menu or by typing `cracksegmentation` in a terminal. Unsigned: your system may warn about an unknown source, which is expected for a locally built package.

**Linux (other distributions, amd64)** — `CrackSegmentation-linux-x86_64.tar.gz`
Portable build, no package manager required — works on Fedora, openSUSE, Arch, and others.
```bash
tar -xzvf CrackSegmentation-linux-x86_64.tar.gz
./CrackSegmentation/CrackSegmentation
```

Prefer running from source, or need another platform? See Installation below — each platform can also be built directly from source using the scripts in `packaging/` (`build_mac.sh`, `build_linux.sh`, `build_windows.bat`).

**Where to put your photos (pre-built packages only).** Unlike running from source (where `Images/` lives next to the script — see Quick start below), a pre-built package creates its data folder automatically on first launch, normally at:
```
Documents/CrackSegmentation/Images        (macOS/Linux)
Documents\CrackSegmentation\Images        (Windows)
```
Just drop your photos in there and (re)launch the app. If `Documents` isn't writable on your system (e.g. a broken OneDrive/cloud-sync redirect on Windows), the app automatically falls back to a per-user app-data folder instead (`%LOCALAPPDATA%\CrackSegmentation` on Windows) — the exact path it's using is always shown in full in the "No valid file found in: ..." message if the folder is still empty.

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

3. Trace cracks and detachments, then save with **S** (stays on the photo), **Q** or **Enter** (moves on to the next photo). The JSON goes to `JSON files/`, the masks to `Binary files/`; the photo stays in `Images/`.

**Go to image** (toolbar or File menu, since 1.0.9), in Mode 1 and Mode 2: jumps to the **First** or **Last** image of the queue, or to one picked from a drop-down list showing 10 names at a time. Picking the 10th name shows the next 10, picking the 1st one shows the previous 10, and the ◀ 10 / 10 ▶ buttons page as well. Like Previous / Next Image, the photo on screen is not saved.

**Sidebars** (since 1.0.8): the buttons for the commands not on the toolbar are split between two sidebars, one on each side of the canvas, grouped by purpose. Right, *More Features*: **Tools** (Crack, Detachment, Ignore region), **Edit cracks** (Cut, Join, Link, Eraser, Remove uncertain, Alternative route, Snap, Width-edit, Retrace), **Inspect cracks** (Show cracks by number, Crack report, HWAV). Left, *View & Dataset*: **View** (blue overlay, markers, Zoom window, Reset zoom, Fullscreen), **Building & scale** (Assign building, Building portion, Compatibility check, Set scale), **Dataset** (Switch Mode, Validation, Training / Test / Validation set, Auto split, Reset pipeline training, Dataset browser, CV folds). When the canvas is narrower than the 1200 px frame (e.g. on a laptop screen), the frame is shown slightly smaller so that no part of the photo is cut off; clicks still land on the exact pixel.

The complete list of shortcuts is available in the in-app help menu. The most used ones:

| Key | Action |
|-----|--------|
| Click, click | Crack tool (`C`): click the start and the end of a crack, the route is traced along the edge. **Shift+click** in between (since 1.0.7) adds as many points as needed that the route must pass through, when it would otherwise follow a shadow line: through them the route follows the thin dark line of the crack on the photo (`TRACE_GUIDE_*` in `config.py`); `U` removes the last one |
| `S` / `Q` / `Enter` | Save the current photo and move to the next one |
| `C` / `D` / `I` | Crack tool / Detachment tool / Ignore region tool (occluded area) |
| `Y` | Close the detachment or ignore polygon |
| `W` / `L` | Import cracks from a previously saved photo of the same building group |
| `T` | Elastic translation with automatic snap to the real fracture edge |
| `V` | Batch re-trace of imported cracks onto the real edges (with plausibility check) |
| `G` | Compare cracks across photos of the same group and discard incompatible ones (also runs automatically after each save) |
| `B` | Manually assign a building group number |
| `A` | Widen the training mask on a single crack stretch; refine with `+` / `-` and `[` `]` `{` `}` |
| `K` | Enter the image scale manually |
| `1` / `2` | Cut a crack part / Join (re-route or reconnect) — two clicks |
| Eraser (right sidebar) | (since 1.0.8) A red window follows the mouse: press and drag it over a crack to erase the part it touches; a crack crossed in the middle is split in two. `[` / `]` make the window smaller / larger. The JSON and binary masks are updated on release; `U` undoes |
| Remove uncertain [?] (sidebar) | (since 1.0.8) After a confirmation, deletes every crack on screen marked `[?]` (uncertain; cracks hidden by **Show cracks by number** are kept), and updates the JSON and binary masks right away; `U` undoes |
| `3` | Link: click one crack and then another, at an end or anywhere on the blue line, to merge them into one; clicked inside a crack, its shorter leftover stays a separate crack. The JSON and binary masks are updated right away |
| `$` | Use the route shown by the **Alternative route** sidebar button (see below) |
| `R` | Redo |

**Show cracks by number** (sidebar, since 1.0.7): type one or more crack numbers separated by `;` (e.g. `3;7;12`) to show only those cracks — blue fill, markers and the `J` list; hidden cracks cannot be clicked by the tools. Leave the field empty to show every crack again. Display only: the JSON, the binary masks and the `-seg` preview always contain every crack.

**Import between different viewpoints** (since 1.0.7): when `W` / `L` find only a rough alignment between two photos of the same building taken from far apart (a close-up and a wide oblique view), the current photo is re-seen in memory from the source photo's viewpoint and matched again, which makes the alignment precise; the photo itself is never changed (`WARP_PREWARP_*` in `config.py`). Cracks running out of the current photo's view are cut at its edge.

**Building portion** (sidebar or `5`, since 1.0.7): for a photo that matches the other photos of its group only in part. Turning it on tints **orange** the part of the photo shown by the other already-segmented photos of the group, and pre-sets the window around it (`COMPATIBLE_AREA_MAX_DIM` in `config.py`); adjust it or drag a window over the shared part of the building (drag inside it to move it, click outside it to remove it, press `5` again when done). While the window is set, `W` / `L` align on that part only and import only the shapes lying mostly inside it (`IMPORT_PORTION_MIN_INSIDE` in `config.py`).

**Alternative route** (sidebar, since 1.0.7): each press shows another edge route between the start and the end of the crack traced last, in its own color (red, green, orange, ...). When no other route is left, the button's arrow turns from ▶ to ◀ and further presses step back through the routes already shown. Press `$` to make the route on screen the crack's new path (blue) and close the tool; a click on the photo closes it keeping the old route, and `U` undoes the change.

## Output files

For every saved photo `<name>`:

| File | Content |
|------|---------|
| `JSON files/<name>.json` | LabelMe annotation (`linestrip` shapes for cracks, polygons for detachments and ignore regions), with the image embedded as base64. A crack merged with **Link** carries a `links` list (`from`/`to` points of each bridged gap, `method` `edge` or `straight`) |
| `Binary files/<name>-crack_mask.png` | Binary crack mask, single channel, same resolution as the photo (0 = background, 255 = crack). Since 1.0.8 a photo saved with no crack gets an all-black one (a negative example; `CRACK_MASK_WRITE_EMPTY` in `config.py`) |
| `Binary files/<name>-detachment_mask.png` | Binary filled-area mask of the detachments (same format) |
| `Binary files/<name>-crack_uncertain_mask.png` | Only the cracks marked uncertain (same band as the crack mask), e.g. as an ignore region in training. Written when `CRACK_UNCERTAIN_EXPORT_MASK` is on and at least one crack is uncertain |
| `Binary files/<name>-ignore_mask.png` | (since 1.0.7) Filled mask of the ignore regions (255 = exclude from loss and metrics). Written when `IGNORE_EXPORT_MASK` is on and the photo has at least one ignore region |
| `Binary files/<name>-seg.jpg` | Colored overlay for visual inspection |

Mask files are written only when the photo contains at least one active element.

Data folders (since 1.0.5):

| Folder | Content |
|--------|---------|
| `Images/` | Every original photo. Photos are never moved: Mode 1 lists the ones without a JSON yet, Mode 2 the ones with a JSON |
| `Binary files/` | Every exported PNG/overlay (crack, detachment, uncertain and ignore masks, `-seg` overlay) |
| `JSON files/` | Every LabelMe JSON |
| `Training set/` | (since 1.0.8) Photos sent there with the **Training set** button (Mode 2), with their JSON and exports, in the same `Images/`, `JSON files/`, `Binary files/` layout |
| `Test set/`, `Validation set/` | (since 1.0.8) Photos sent there with the **Test set** / **Validation set** buttons (Mode 2), in `Images/`; their JSON and masks stay in `JSON files/` and `Binary files/` |
| `Suitable for training/` | (since 1.0.6) Photos picked by **Validation**, with their JSON and exports, in the same `Images/`, `Binary files/`, `JSON files/` layout, plus a `validation_report_<date>.csv` per run |

Photos are moved out of `Images/` only by the dataset buttons and by **Validation**. **Training set**, **Test set** and **Validation set** (sidebar, Mode 2, since 1.0.8) move the photo on screen to that folder, creating it when missing, and show the next photo; Training set also moves its JSON and `Binary files/` exports, Test and Validation set only the photo. Existing files are never overwritten (`_1`, `_2`, ... is appended), and unsaved changes are offered for saving first. When that set already holds photos of the same building group (`__BLDGnnn`), or the group is already in another set (the same building in two sets), those photos are listed and the move waits for a confirmation. Hovering any of the three buttons shows how many photos each set already holds, and the total.

**Auto split** (left sidebar or File menu, since 1.0.9) splits **every** segmented photo of `Images/` (photo with its JSON) into the three sets in one go, instead of one photo at a time: it asks the Training / Validation / Test percentages (default `AUTO_SPLIT_PERCENT = "70/15/15"` in `config.py`; `80/0/20` when cross-validation replaces the validation set), keeps every building group (`__BLDGnnn`) in one set, leaves a group that already has photos in a set (moved by hand) in that set, and deals the groups so that every set gets a similar share of drone photos and of photos without cracks. It shows the plan (photos, buildings, drone photos, photos without cracks per set) and moves the files only after a confirmation, as the single buttons do. It works in either mode, also on the "No photo left" screen.

**Reset pipeline training** (left sidebar or File menu, since 1.0.9) undoes the split: after a confirmation showing how many photos each set holds, every file of `Training set/`, `Validation set/` and `Test set/` goes back to `Images/`, `JSON files/` and `Binary files/`, so Auto split or the single buttons can start again. The cross-validation output (`folds.csv` and `folds/`) describes a split that no longer exists and is removed. A file whose name already exists in the working folder is never overwritten: it stays in its set and is listed. Emptied set folders are removed. It works in either mode, also on the "No photo left" screen, and the image queue is reloaded.

**Dataset browser** (left sidebar or File menu, since 1.0.9) covers the photo, like the manual, with a white scrollable list of every photo in the three sets, grouped by set, with its building group number (`__BLDGnnn`) and a note for drone photos and photos without cracks. Select one or more photos (Ctrl/Shift + click; a set's title row selects all its photos) and press **Training set**, **Validation set** or **Test set** to move them, after a confirmation. With **Whole building** ticked (the default) every photo of the same buildings moves too, so no building ends up in two sets. The layout of the sets is kept: to Training set the photo goes with its JSON and masks, from Training set to Validation or Test set the JSON and masks go back to `JSON files/` and `Binary files/`, between Validation and Test set only the photo moves. Nothing is overwritten. Press **CV folds** again after changing the Training set. **✕ Close** (or the button again) brings the photo back.

**CV folds** (sidebar or File menu, since 1.0.8) prepares cross-validation on the photos of `Training set/`: it asks the number of folds (default `CV_FOLDS = 5` in `config.py`), splits the photos so that every building group (`__BLDGnnn`) falls in a single fold (a photo without a group is a group of its own), shows the photos and buildings per fold, and after a confirmation writes `Training set/folds.csv`. Each row holds the photo, its group, its fold and the paths of its JSON, crack, uncertain and ignore masks (relative to `Training set/`, empty when missing). It also copies each fold into `Training set/folds/fold_1` .. `fold_K`, each with its photos, JSON and exports in the usual `Images/`, `JSON files/`, `Binary files/` layout, ready for a training script that wants one folder per fold (round k: `fold_k` for validation, the other folders for training). Pressed again, `folds.csv` and the `folds/` folder are replaced; `Training set/` itself is never changed. A photo saved with **no crack** traced (a wall with downpipes, cables, railings or joints but no crack) is a *negative example*: its JSON has no crack and its crack mask is all black (older saves have none), and `folds.csv` marks it with `negative = 1` so that training uses it with an all-background label, even when the mask file is missing. The **Validation** button (sidebar, available as soon as every photo in `Images/` has its JSON, whichever photo is on screen) is a different tool: a photo qualifies when its cracks reach the `VALIDATION_*` thresholds in `config.py` (mean reliability, mean confidence, share of uncertain cracks), and of several near-identical photos only the best-scoring one is kept.

Folders from earlier versions (`segmentated images/`, `already processed images/`) are migrated automatically at startup: each file is moved to the matching new folder, nothing is overwritten or deleted, and an old folder is removed only once it is empty.

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
| `CRACK_UNCERTAIN_RELIABILITY_PCT` | Cracks below this reliability (%) are saved as uncertain. `None` disables the rule |
| `CRACK_UNCERTAIN_CONFIDENCE` | Cracks below this confidence (0-1) are saved as uncertain. `None` (default) disables the rule |
| `CRACK_UNCERTAIN_LABEL` | Label of uncertain cracks (default `crack_incerta`, photo name appended) |
| `CRACK_UNCERTAIN_EXPORT_MASK` | Also write `<name>-crack_uncertain_mask.png` |
| `IGNORE_LABEL` | Label of ignore regions (default `ignore`, photo name appended) |
| `IGNORE_EXPORT_MASK` | Also write `<name>-ignore_mask.png` |
| `CRACK_MULTIVIEW_ENABLED` | Check every crack in the other photos of its building group at save |
| `CRACK_MULTIVIEW_MAX_VIEWS` | Maximum number of other photos checked |
| `CRACK_MULTIVIEW_CONFIRM_CONFIDENCE` | Confidence a crack must reach in another photo to count as confirmed there |
| `CRACK_UNCERTAIN_MULTIVIEW_RATIO` | Cracks confirmed in fewer than this share of the checked photos are saved as uncertain. `None` (default) disables the rule |

When `IMAGE_GSD_MM_PER_PX` is set, the band width in pixels is derived automatically from it, with a safety ceiling against input errors.

### Crack quality scores

Every crack shape in the JSON carries these scores, also shown per crack in the **J** panel (`Crack N: 85% c0.97 v3/4`, `[?]` = uncertain):

- `reliability_pct` (0-100): share of the traced line backed by dark texture in the adaptive-threshold mask. Textured plaster, joints and shadows can score high; faint real cracks can score low.
- `confidence` (0-1): mean percentile rank of a dark-ridge (Sato) filter response along the line, against the wall around each stretch. About 0.5 = no more line-like than the surroundings; close to 1 = clearly the most line-like structure in the area. A heuristic computed from the photo alone, not a calibrated probability.

- `multiview` (`views_checked`, `views_confirmed`, `confidence`): at save, the crack is projected into the other photos of its building group (SIFT + homography, then a local texture alignment) and its confidence is measured there. A real crack stays in place on the wall, while shadows, reflections and dirt move or vanish with light and viewpoint. A photo confirms the crack when its confidence there reaches `CRACK_MULTIVIEW_CONFIRM_CONFIDENCE`; photos that cannot be aligned, or where the crack is out of frame, are not counted. It measures the photos, not the annotations, so a crack imported with **W**/**L** is not confirmed just because it was copied. Permanent lines such as joints are confirmed too. `null` when the photo has no building group or no other photo of the group is available; computed at save, so the panel shows `v` only after saving.

An uncertain crack gets the `CRACK_UNCERTAIN_LABEL` label and `"flags": {"uncertain": true}`; the regular crack mask still contains every crack. Scores and labels are recomputed at every save.

### Ignore regions

Some parts of a facade cannot be annotated reliably: a safety net, scaffolding, cables or vegetation in front of the wall. Cracks there are only partly visible, so marking those pixels as background would teach a model the wrong thing, and a correct prediction there would be scored as a false positive.

Press **I** (or the **Ignore region** sidebar button): a small dialog asks for the shape and, for rectangles, which sides to extend to the photo edges.

- **Rectangle** / **Square**: click two opposite corners (a live preview follows the cursor). Every side ticked in the dialog (*left*, *right*, *top*, *bottom*) is pushed exactly onto that photo edge, so a safety net across the bottom of the photo is one rectangle with left, right and bottom ticked: only its top edge needs placing. With nothing ticked the region can sit anywhere in the photo. A square uses the longer side of the two clicks.
- **Free polygon**: click the corners (straight edges, no edge snapping) and close with **Y**; corners near a photo edge snap onto it.

The dialog remembers the last choice; Cancel leaves the current tool unchanged. **U** cancels the corner(s) in progress, or the last saved region; with nothing in progress, a click inside a saved region deletes it (the most recent one where regions overlap). To draw a region overlapping another, place its first corner outside the existing one. Regions are drawn as magenta-hatched glass.

Each region is saved in the JSON as a `polygon` shape with label `ignore_<photo>` and `"flags": {"ignore": true}`, and exported as `<name>-ignore_mask.png`. In training, exclude those pixels from the loss; in evaluation, exclude them from the metrics. They play no role at inference. Ignore regions belong to their own photo: **W**/**L** imports never project them onto another photo, and Validation does not count them as detachments.

Stop crack traces at the edge of the occluded area rather than tracing through it. `I` used to be an alias of `+` (zoom in); since 1.0.7 zoom in is `+` or `=`.

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
├── Images/                            # your original photos — not included, create it locally (see Quick start)
├── Binary files/                      # exported masks and overlays (created on first save)
├── JSON files/                        # LabelMe annotations (created on first save)
├── smart_segmentation.py              # main interactive tool
├── config.py                          # tunable constants and shared state
├── crack_segmentation_gui.py          # GUI layer (Tkinter window, PDF manual panel)
├── compare_and_filter_cracks.py       # cross-photo crack consistency filter
├── crack_segmentation_evaluator.py    # segmentation quality evaluator
├── tests_support/                     # test doubles (fake_tkinter.py, fake_fitz.py, ...)
├── docs/
│   └── manual_source/                 # HTML source of the PDF manual, see below
├── .github/                           # CI workflow, issue and pull request templates
├── CONTRIBUTING.md
├── requirements.txt
└── requirements-optional.txt
```

The PDF manual (`Crack_Segmentation_User_Manual.pdf`, in the repo root) is generated from `docs/manual_source/manual.html` — see [`docs/manual_source/README.md`](docs/manual_source/README.md) for how to edit it and regenerate the PDF.

## Notes

- Code comments, docstrings, on-screen (HUD) text, and console messages are all in English.
- The PDF manual panel degrades gracefully: without PyMuPDF (or without the PDF file) the rest of the application works normally.

## Contributing

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the code conventions used in this project.

## Authors & Credits

This software was developed within the framework of the **ECHO-TWIN** project by the INAF (Istituto Nazionale di Astrofisica) Research Group.

| Name | Contribution |
|------|--------------|
| **Roberto Spina** <sup>1,2</sup> | Code Architecture, Software Design & Lead Development |
| **Fabio Vitello** <sup>1</sup> | Project Coordination & Supervision |
| **Eva Sciacca** <sup>1</sup> | Project Coordination & Supervision |
| **Leonardo Pelonero** <sup>1</sup> | General Support |
| **Salvatore Scavo** <sup>1</sup> | General Support |

<sup>1</sup> INAF – Catania Astrophysical Observatory &nbsp;&nbsp; <sup>2</sup> Order of Geologists of Sicily (National Register)

**AI Disclosure.** This application utilizes artificial intelligence models for certain stages of the workflow, specifically for analysis, technical content generation, and development support. The AI components do not replace scientific validation nor do they influence results in a deterministic manner: they are used as assistance tools, not as sources of truth.

## License

Released under the [MIT License](LICENSE).

The optional dependency [PyMuPDF](https://github.com/pymupdf/PyMuPDF) is distributed by Artifex under AGPL-3.0 or a commercial license. It is not included in this repository and is not required to run the tool. If you redistribute a build that bundles it, check that the AGPL terms are met.

