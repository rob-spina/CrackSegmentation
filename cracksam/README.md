# cracksam — SAM + LoRA crack model code

`model.py`, `sam_lora.py` and `tiling.py` of the cracksam_lora training project (the code trained on
Leonardo, CINECA), copied here so that the **SAM auto-segment** button runs exactly the same model
code as `infer.py` did. They are never imported by the app itself: `sam_infer.py` imports them in the separate
Python chosen in SAM model settings, which needs `torch`, `torchvision` and `segment-anything`.

If the training code changes, copy the new versions here.

2.0.1: `forward_batch(..., upscale=1.0)` enlarges each tile before SAM (train.py / infer.py `--upscale`); the same
`model.py` is in echo-twin and must be copied to `cracksam_lora` on Leonardo.
