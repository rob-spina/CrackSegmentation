## CrackSegmentation 2.1.0

### What's new in 2.1.0

#### SAM click to accept: only what you click is saved
- When the mode opens, every crack the model sees is **coloured by how sure the model is**: **orange** (mean probability ≥ 0.65), **dark yellow** (≥ 0.45), **light yellow** below. Nothing is saved until you accept it (2.0.0 saved the confident cracks automatically; a photo that still has them asks once whether to remove them).
- **H / G / A** show or hide all the colours, the yellow ones, the orange ones; **L / D** discard every light / dark-yellow crack; **1 / 2 / 3 / 4** accept every orange / dark-yellow / light-yellow / coloured crack (Cmd/Ctrl+click on the SAM button = orange).
- A **dragged cyan window** (10 to 12000 screen px) accepts only the coloured parts inside it.
- **Exclude crack (X)**: a click removes a crack, accepted ones too, from JSON and PNG at once; a **dragged red window** removes everything inside it.
- **Threshold dial** (round, **[ / ]**): re-cuts the crack map live at a lower threshold (fewer false negatives) or a higher one (fewer false positives), with the counts per colour; accepted cracks never change. It also holds the orange / dark-yellow thresholds and the **maximum crack width** (40 px, **0** turns it on / off): wider parts (open lesions, missing plaster, stone joints) are shown as a 5 px centre line or hidden; accepting the line takes the real-width patch.
- **Width under the mouse**: the crack width in photo pixels is shown next to the cursor, in red above the width limit with the hint to cover the area with an ignore region instead.
- **Cut map (K)**: a line dragged across the map splits it, so a crack joined to tree branches, a cable or a stain is accepted alone.
- **Recompute crack map here**: computes the photo's map again on this computer (e.g. a map from the HPC that looks wrong).
- **Delete by number**: with the numbers shown (Space) a click on a crack's number deletes it.
- Faster drawing (cached overlay), and accepted / deleted cracks are saved with the photo (Enter / S).

#### Dataset and review
- **To be reviewed**: marks a photo to go over again (note, red frame); Mode 2 can load only the marked photos; Validation and Auto split leave them where they are. **To be reviewed by date** marks at once every photo saved up to a date.
- **Delete image**: moves a photo and its files to `Deleted images`.
- `sam_batch.py`: `--images-subdir` (a set's photos), `--skip-labelled` (only the photos still to label).

#### Other changes
- **Eraser** window up to 1200 px, with larger [ / ] steps for larger windows.
- Sidebar: locked buttons keep their label (padlock, with the reason on click); section headings readable on the dark sidebar.
- Progress window for the SAM weights download.

### Model weights

The crack weights are still `cracksam_vit_h_lora_r4.pth` from the v2.0.0 release (SAM ViT-H + LoRA rank 4, validation IoU 0.72 on Facade390), downloaded by the app the first time. See the [model card](https://github.com/rob-spina/CrackSegmentation/blob/main/docs/sam_model/MODEL_CARD.md).

Authors: Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo (INAF).

### Requirements

As in 2.0.0, the app does not include PyTorch: the SAM model runs in a separate Python with `torch`, `torchvision` and `segment-anything`.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-2.1.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-2.1.0-windows-Setup.exe`, or the portable `CrackSegmentation-2.1.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_2.1.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_2.1.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-2.1.0-linux-x86_64.tar.gz`. Extract it and run `./CrackSegmentation/CrackSegmentation`.
