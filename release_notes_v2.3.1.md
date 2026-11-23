## CrackSegmentation 2.3.1

Bug-fix release for 2.3.0.

### Fixed

- **"This photo has changes not saved yet" after a save.** The window appeared even for a photo that had just been saved (sometimes several times), when pressing Filter by building or Test / Validation / Training set. The app looked only at the undo history, which a save does not empty. It now compares the cracks, detachments and ignore regions on screen with the last save or load, and asks only when they differ.
- **Filter by building: its own question.** The window used the dataset-split text ("Save them before moving it to 'filter by building'?"), but the filter never moves the photo. It now asks "Save them before loading another building?" (Yes = save, then load; No = don't save, the changes are lost; Cancel = stay on this photo).

Nothing else changes from 2.3.0: see the [2.3.0 notes](https://github.com/rob-spina/CrackSegmentation/releases/tag/v2.3.0) for Filter by building.

### Model weights

The crack weights are still `cracksam_vit_h_lora_r4.pth` from the v2.0.0 release (SAM ViT-H + LoRA rank 4, validation IoU 0.72 on Facade390), downloaded by the app the first time. See the [model card](https://github.com/rob-spina/CrackSegmentation/blob/main/docs/sam_model/MODEL_CARD.md).

Authors: Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo (INAF).

### Requirements

As in 2.0.0 to 2.3.0, the app does not include PyTorch: the SAM model runs in a separate Python with `torch`, `torchvision` and `segment-anything`.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-2.3.1-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-2.3.1-windows-Setup.exe`, or the portable `CrackSegmentation-2.3.1-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_2.3.1_amd64.deb`. Install with `sudo apt install ./cracksegmentation_2.3.1_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-2.3.1-linux-x86_64.tar.gz`. Extract it and run `./CrackSegmentation/CrackSegmentation`.
