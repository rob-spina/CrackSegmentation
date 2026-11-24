## CrackSegmentation 2.4.0

### What's new in 2.4.0

#### Issue log
A new **Issue log** button (left sidebar, Dataset section; also in the new **Issues** menu, code 164) keeps a list of the problems found in the segmentation while you label or review. While it is on (button pressed), every save in Mode 1 or Mode 2 opens a window to tick what was wrong:

- **A** masonry joints traced as cracks
- **B** edges, cornices, roads or streaks traced
- **C** rubble or collapse traced (area to ignore)
- **D** crazing traced as cracks
- **E** real cracks not traced
- **F** cracks traced badly (path, width or gaps)
- **X** other, with a free description

plus a note (zone, what was done). The rows go to `segmentation_issues.csv` in the data folder, one per photo and category, with the columns `n, immagine, categoria, problema, modalità` (1 or 2) and `destinazione`.

The same window asks the **destination set** of the photo: Training set, Validation set, Test set or *Not decided yet* (written as `nessuna`). It is only written down, the photo is not moved. The window proposes the set already chosen for the photo, otherwise the set that already holds photos of the same building (`__BLDGnnn`), and warns when the building is already in a set, so a building does not end up in two sets.

Saving the photo again shows its categories already ticked and replaces its rows; with nothing ticked the photo leaves the list; **Skip** leaves the list as it is.

#### Show segmentation issues
**Issues > Show segmentation issues** (menu bar, code 165) shows the list as a table over the image, like the user manual. The selected rows can be deleted, e.g. once a photo has been fixed.

### Model weights

The crack weights are still `cracksam_vit_h_lora_r4.pth` from the v2.0.0 release (SAM ViT-H + LoRA rank 4, validation IoU 0.72 on Facade390), downloaded by the app the first time. See the [model card](https://github.com/rob-spina/CrackSegmentation/blob/main/docs/sam_model/MODEL_CARD.md).

Authors: Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo (INAF).

### Requirements

As in 2.0.0 to 2.3.1, the app does not include PyTorch: the SAM model runs in a separate Python with `torch`, `torchvision` and `segment-anything`.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-2.4.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-2.4.0-windows-Setup.exe`, or the portable `CrackSegmentation-2.4.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_2.4.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_2.4.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-2.4.0-linux-x86_64.tar.gz`. Extract it and run `./CrackSegmentation/CrackSegmentation`.
