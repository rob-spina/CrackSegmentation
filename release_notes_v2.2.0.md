## CrackSegmentation 2.2.0

### What's new in 2.2.0

#### Active learning and Leonardo switched on
Both commands were already in the 2.1.0 code but hidden; from 2.2.0 they have their buttons in the left sidebar (Dataset section) and their entries in the Tools menu.

- **Active learning (next round)** (code 153): takes the newest `SAM model/rounds/round_NN/best_lora.pth` as the SAM model (none = the model in use, round 0), ranks the photos still to label by how much the model hesitates (share of pixels with 0.3 ≤ p ≤ 0.7) and loads the 50 most uncertain (`ACTIVE_LEARNING_BATCH`) in Mode 1 with **SAM click to accept** on. The ranking goes to `SAM model/rounds/ranking_<tag>.csv`; when the Test set has labels and crack maps, the model's IoU / F1 there is added to `SAM model/rounds/history.csv` and compared with the previous round. Missing crack maps can be computed here (slow) or on the HPC.
- **Leonardo (train the next round)** (code 154): a window that runs a round on the Leonardo cluster (CINECA) from the app: **Connect** (`step ssh login` when the SSH certificate is missing or expired), **Send data + start round** (rsync of the dataset, `git pull`, `sbatch docs/sam_model/active_learning_round_leonardo.sbatch`), **Job status**, **Fetch results** (round weights and crack maps back). No password is asked or stored; it needs `step`, `ssh` and `rsync`.
- The round script exports the Training + Validation set (`active_learning_prep.py`; uncertain cracks, ignore regions and detachments become a grey band the training leaves out of the loss), fine-tunes from the previous round and computes the new crack maps of the photos still to label and of the Test set.

Group similar photos and Auto ignore stay switched off in this release (`CRACKSEG_ALL_FEATURES=1` turns them on).

### Model weights

The crack weights are still `cracksam_vit_h_lora_r4.pth` from the v2.0.0 release (SAM ViT-H + LoRA rank 4, validation IoU 0.72 on Facade390), downloaded by the app the first time. See the [model card](https://github.com/rob-spina/CrackSegmentation/blob/main/docs/sam_model/MODEL_CARD.md).

Authors: Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo (INAF).

### Requirements

As in 2.0.0 and 2.1.0, the app does not include PyTorch: the SAM model runs in a separate Python with `torch`, `torchvision` and `segment-anything`.

### Downloads

- **macOS (Apple Silicon)**: `CrackSegmentation-2.2.0-macos-arm64.dmg`. Not notarized by Apple: on first launch, open **System Settings → Privacy & Security** and click **Open Anyway** (only needed once).
- **Windows**: `CrackSegmentation-2.2.0-windows-Setup.exe`, or the portable `CrackSegmentation-2.2.0-windows-portable.zip`. Unsigned: if SmartScreen flags it, click **More info → Run anyway**.
- **Linux (Debian/Ubuntu, amd64)**: `cracksegmentation_2.2.0_amd64.deb`. Install with `sudo apt install ./cracksegmentation_2.2.0_amd64.deb`.
- **Linux (other distributions, amd64)**: `CrackSegmentation-2.2.0-linux-x86_64.tar.gz`. Extract it and run `./CrackSegmentation/CrackSegmentation`.
