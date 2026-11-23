## CrackSegmentation 2.3.0

### What's new in 2.3.0

#### Filter by building
A new **Filter by building** button (left sidebar, Dataset section; also in the Tools menu, code 163) loads only the photos of one building group, i.e. the photos whose name carries the same `__BLDGnnn` tag. A window lists the groups found in `Images`, each with its number of labelled photos and photos to label, and offers three choices:

1. **Photos already labelled**: opened as in Mode 2, with their JSON;
2. **Photos to label**: Mode 1;
3. **All** (default): each photo opens in the mode it needs (Mode 2 when it has a JSON, Mode 1 otherwise), and a photo labelled in Mode 1 stays in the queue, so a whole building can be walked back and forth.

While the filter is on the status bar shows `BUILDING BLDGnnn`; Switch Mode, Validation and the other commands that rebuild the queue turn it off. Photos without a tag, or already moved to Training / Validation / Test set, are not listed.

### Model weights

### Fixed

- The "changes not saved yet" window no longer appears for a photo that was already saved: the app now compares the cracks, detachments and ignore regions on screen with the last save (the undo history alone kept a saved photo looking modified). This also applies to Test/Validation/Training set.
- [Filter by building] asks with its own wording ("Save them before loading another building?"): it never moves the photo.

The crack weights are still `cracksam_vit_h_lora_r4.pth` from the v2.0.0 release (SAM ViT-H + LoRA rank 4, validation IoU 0.72 on Facade390), downloaded by the app the first time. See the [model card](https://github.com/rob-spina/CrackSegmentation/blob/main/docs/sam_model/MODEL_CARD.md).

Authors: Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo (INAF).

### Requirements

As in 2.0.0, 2.1.0 and 2.2.0, the app does not include PyTorch: the SAM model runs in a separate Python with `torch`, `torchvision` and `segment-anything`.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-2.3.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-2.3.0-windows-Setup.exe`, or the portable `CrackSegmentation-2.3.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_2.3.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_2.3.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-2.3.0-linux-x86_64.tar.gz`. Extract it and run `./CrackSegmentation/CrackSegmentation`.
