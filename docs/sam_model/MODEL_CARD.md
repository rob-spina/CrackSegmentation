# CrackSegmentation SAM crack model — `cracksam_vit_h_lora_r4.pth`

Crack-segmentation weights used by the **SAM auto-segment** button of CrackSegmentation (since 2.0.0).
The app downloads this file automatically the first time the button is pressed, together with Meta's base SAM weights.

## Authors

Roberto Spina, Fabio Vitello, Eva Sciacca, Leonardo Pelonero, Salvatore Scavo — INAF, Istituto Nazionale di Astrofisica (ECHO-TWIN project).

## Model

- **Base model:** Segment Anything (SAM) ViT-H, `sam_vit_h_4b8939.pth` (Meta AI, Apache 2.0). Not included in this file: it is downloaded from Meta.
- **Fine-tuning:** LoRA adapters of rank 4 on the Q and V projections of every attention block of the image encoder, plus the fully fine-tuned mask decoder. The original SAM weights and the prompt encoder are frozen.
- **No prompts:** SAM is used as a plain binary segmentation model. The prompt encoder is called with no points, boxes or masks, giving a fixed "empty prompt" embedding.
- **Input:** native-resolution RGB tiles of 512 × 512 px. Large photos (e.g. 4K drone images) are tiled with overlap, and the tile probabilities are stitched back with a soft blending.
- **Output:** one crack-probability map per photo.
- **This file:** only the trainable tensors (LoRA `linear_a_*` / `linear_b_*` and `mask_decoder.*`), as saved by `train.py` (`best_lora.pth`, best validation IoU).
- **Code:** `cracksam/model.py`, `cracksam/sam_lora.py` and `cracksam/tiling.py` in this repository. Inference is in `sam_infer.py`.

## Training data

Trained on Facade390 (CC BY 4.0, Ge et al. 2024) and CrackSeg9K (CC0 1.0, Kulkarni et al. 2022). Some source images of CrackSeg9K (e.g. GAPS) carry their own terms; intended for research use.

- **Facade390:** facade crack images released with CrackSAM, CC BY 4.0.
- **CrackSeg9K:** 9,159 images combined from Masonry, CFD, CrackTree200, Volker, DeepCrack, Ceramic, SDNET2018, Rissbilder, Crack500 and GAPS384. The compilation is released under CC0 1.0 on Harvard Dataverse. The original images keep their own terms: GAPS, for example, is distributed for academic use only.
- Training was run on the Leonardo supercomputer (CINECA, NVIDIA A100 64 GB).

## Intended use and limits

- For research use: automatic first-pass crack segmentation of building facade photos, to be checked and corrected by an operator in CrackSegmentation.
- Every crack it finds is editable, and cracks with a low mean probability are saved as uncertain (red).
- The training photos are mostly close-up, ground-level images. On oblique drone photos taken from a different distance or with a different ground sample distance, accuracy may drop until the model is adapted to that domain (`cv_domain_adapt.py`).
- Not a substitute for a structural assessment by a qualified engineer or geologist.

## Requirements

A Python with `torch`, `torchvision` and `segment-anything`, chosen in **SAM model settings**: for example, the virtual environment used for training. A CUDA GPU or Apple Silicon (MPS) is used when present, otherwise the CPU.

## Citation

If you use these weights, please cite CrackSegmentation and the works it builds on:

- R. Spina, F. Vitello, E. Sciacca, L. Pelonero, S. Scavo (INAF), *CrackSegmentation*, software. [doi:10.5281/zenodo.22948009](https://doi.org/10.5281/zenodo.22948009)
- A. Kirillov et al., *Segment Anything*, ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- K. Ge et al., *Fine-tuning vision foundation model for crack segmentation in civil infrastructures* (CrackSAM, Facade390), 2024. [arXiv:2312.04233](https://arxiv.org/abs/2312.04233)
- S. Kulkarni et al., *CrackSeg9k: A Collection and Benchmark for Crack Segmentation Datasets and Frameworks*, ECCV Workshops 2022. [arXiv:2208.13054](https://arxiv.org/abs/2208.13054); data [doi:10.7910/DVN/EGIEBY](https://doi.org/10.7910/DVN/EGIEBY)

---

## Publishing a new version of the weights (maintainer notes)

1. Rename the trained `best_lora.pth` to `cracksam_vit_h_lora_r4.pth`.
2. Compute its checksum on the Mac:
   ```bash
   shasum -a 256 cracksam_vit_h_lora_r4.pth
   ```
3. Attach the file to the app's GitHub Release (since 2.0.0: `v2.0.0`), next to the installers: `gh release upload v2.0.0 cracksam_vit_h_lora_r4.pth`.
4. In `config.py`, `SAM_CRACK_WEIGHTS`, set `sha256` to the value from step 2. Without it, the app checks only that the file was downloaded completely.
5. Later app releases keep downloading the weights from the `v2.0.0` release, so its asset must not be deleted. For new weights, use a new file name (e.g. `cracksam_vit_h_lora_r4_v2.pth`), attach it to that app release and update `url`, `file` and `sha256`. Users who already downloaded the old weights keep them until they delete `sam_model.json` from the data folder.
