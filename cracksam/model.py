"""
model.py
--------
Builds SAM + LoRA and wraps the forward pass needed to train it as a
plain binary segmentation model (no user-provided points/boxes -- SAM's
prompt encoder is called with everything set to None, giving a fixed
"empty prompt" embedding, exactly the trick used by SAM-LoRA semantic
segmentation reimplementations such as SAMed).

A note on batching: SAM's official mask_decoder was written for the
"1 image -> N prompts" case (e.g. many boxes on the same image) and
internally does `torch.repeat_interleave(image_embeddings, tokens.shape[0])`.
That assumption breaks if you naively feed it B different images with
1 prompt each in a single batched call -- it will NOT do what you want.
To stay correct and avoid a subtle silent bug, this wrapper batches the
(heavy) image encoder normally, but loops over the (tiny, cheap) prompt
encoder + mask decoder one sample at a time. This is slightly less
parallel but is unambiguously correct and is fast in practice since the
decoder is a lightweight module.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from sam_lora import (
    apply_lora_to_image_encoder,
    enable_gradient_checkpointing,
    freeze_prompt_encoder,
    print_trainable_summary,
)


def build_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    lora_r: int = 4,
    device: str = "cuda",
    grad_checkpoint: bool = False,
) -> nn.Module:
    """
    checkpoint_path: path to the base SAM .pth checkpoint (ViT-B/L/H),
                      downloaded from facebookresearch/segment-anything.
    model_type: must match the checkpoint ("vit_b", "vit_l", or "vit_h").
    grad_checkpoint: trade compute for activation memory (see sam_lora.py
                      enable_gradient_checkpointing). Recommended on
                      small-VRAM GPUs (e.g. 6 GB laptop cards).
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    apply_lora_to_image_encoder(sam, r=lora_r)
    freeze_prompt_encoder(sam)
    # Mask decoder is small (~4M params) -- fine-tune it fully.
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    if grad_checkpoint:
        enable_gradient_checkpointing(sam)

    print_trainable_summary(sam)
    return sam.to(device)


def _empty_prompt_embeddings(sam: nn.Module, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse/dense embeddings for a 'no prompt given' call (batch size 1)."""
    sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
    return sparse.to(device), dense.to(device)


def forward_batch(sam: nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    """
    images: (B, 3, H, W) float tensor in [0, 1] (as produced by dataset.py)
    returns: (B, 1, H, W) raw logits, resized back to the input resolution
    """
    images = images.to(device)
    B, _, H, W = images.shape

    # sam.preprocess expects pixel values roughly in [0, 255] and handles
    # ImageNet normalization + padding to a square img_size (1024) itself.
    pixel_values = images * 255.0
    input_images = torch.stack([sam.preprocess(img) for img in pixel_values], dim=0)

    image_embeddings = sam.image_encoder(input_images)  # (B, C, h, w), batched -- fine
    image_pe = sam.prompt_encoder.get_dense_pe()  # (1, C, h, w)

    logits_list = []
    for i in range(B):
        sparse_emb, dense_emb = _empty_prompt_embeddings(sam, device)
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings[i : i + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,  # single binary "crack" mask
        )
        # low_res_masks: (1, 1, h_low, w_low) -> back to original (H, W)
        masks = sam.postprocess_masks(
            low_res_masks, input_size=(H, W), original_size=(H, W)
        )
        logits_list.append(masks)

    return torch.cat(logits_list, dim=0)  # (B, 1, H, W)
