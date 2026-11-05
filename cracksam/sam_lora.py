"""
sam_lora.py
-----------
Injects LoRA (Low-Rank Adaptation) adapters into the Q and V projections
of every attention block in SAM's image encoder (ViT backbone), while
freezing all the original SAM weights.

This follows the same pattern used by published SAM-LoRA fine-tuning
projects (e.g. SAMed for medical imaging): only the Q and V projections
inside `attn.qkv` get a low-rank residual; K is left untouched.

NOTE: this is an independent reimplementation of the general SAM-LoRA
recipe described in the CrackSAM paper (Ge et al., 2024). The authors'
official repo (github.com/KG-TSI-Civil/CrackSAM) ships only training
data + a pointer to base SAM weights, not the fine-tuning code itself,
so this script is not a copy of their implementation -- it is a
from-scratch implementation of the same idea, for you to train your
own checkpoint on the Facade390 (or any other) dataset.
"""

import math
from typing import List

import torch
import torch.nn as nn


class _LoRA_qkv(nn.Module):
    """
    Wraps an existing `qkv` linear layer (as found in SAM's ViT attention
    blocks) and adds a low-rank residual to the Q and V slices only.

    Original qkv: x -> Linear(dim, 3*dim) -> [q | k | v]
    New:          q' = q + B_q(A_q(x))
                  v' = v + B_v(A_v(x))
                  k' = k   (unchanged)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)  # (..., 3 * dim)
        q, k, v = (
            qkv[..., : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )
        q = q + self.linear_b_q(self.linear_a_q(x))
        v = v + self.linear_b_v(self.linear_a_v(x))
        return torch.cat([q, k, v], dim=-1)


def apply_lora_to_image_encoder(
    sam_model: nn.Module, r: int = 4, verbose: bool = True
) -> List[nn.Module]:
    """
    Freezes SAM's image encoder and injects LoRA adapters (rank r) into
    the Q/V projections of every transformer block.

    Args:
        sam_model: a SAM model loaded via segment_anything.sam_model_registry
        r: LoRA rank (4-8 is a reasonable range to start with)
        verbose: print a short summary of trainable parameters

    Returns:
        list of the new _LoRA_qkv modules (kept mostly for inspection/debug)
    """
    image_encoder = sam_model.image_encoder

    # Freeze all original image-encoder weights first.
    for param in image_encoder.parameters():
        param.requires_grad = False

    lora_layers = []
    for blk in image_encoder.blocks:
        attn = blk.attn
        qkv = attn.qkv
        dim = qkv.in_features

        linear_a_q = nn.Linear(dim, r, bias=False)
        linear_b_q = nn.Linear(r, dim, bias=False)
        linear_a_v = nn.Linear(dim, r, bias=False)
        linear_b_v = nn.Linear(r, dim, bias=False)

        # Standard LoRA init: A ~ Kaiming, B = 0 so the adapter starts
        # as a no-op (delta = B(A(x)) = 0 at step 0).
        nn.init.kaiming_uniform_(linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(linear_b_q.weight)
        nn.init.zeros_(linear_b_v.weight)

        attn.qkv = _LoRA_qkv(qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v)
        lora_layers.append(attn.qkv)

    if verbose:
        trainable = sum(
            p.numel() for p in image_encoder.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in image_encoder.parameters())
        print(
            f"[sam_lora] image encoder: {trainable:,} / {total:,} "
            f"params trainable ({100 * trainable / total:.3f}%)"
        )

    return lora_layers


def freeze_prompt_encoder(sam_model: nn.Module) -> None:
    """We use SAM with *no* user prompts (points/boxes/masks all None),
    so the prompt encoder just produces a fixed 'no prompt' embedding.
    There is nothing meaningful to learn there, so we freeze it."""
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False


def print_trainable_summary(sam_model: nn.Module) -> None:
    trainable = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam_model.parameters())
    print(
        f"[sam_lora] full model: {trainable:,} / {total:,} "
        f"params trainable ({100 * trainable / total:.3f}%)"
    )


def enable_gradient_checkpointing(sam_model: nn.Module, verbose: bool = True) -> None:
    """
    Wraps every image-encoder block's forward pass in
    torch.utils.checkpoint, trading extra compute (roughly one extra
    forward pass per block) for much lower activation memory during
    backward. This matters even with LoRA: freezing a block's weights
    avoids storing gradients/optimizer state for THOSE weights, but the
    backward pass still has to flow *through* every block to reach the
    LoRA params in earlier blocks, so forward activations for all
    blocks still have to be kept in memory unless you checkpoint them.

    On a small-VRAM GPU (e.g. 6 GB laptop cards) this is often the
    difference between fitting a 1024x1024 tile in memory or not.

    NOTE: this assumes segment_anything's ViT `Block.forward(self, x)`
    signature (single positional tensor in, single tensor out), which
    is the case in the current public segment_anything release. If a
    future version changes that signature, this will raise a clear
    error rather than silently doing the wrong thing.
    """
    import torch.utils.checkpoint as ckpt

    n_wrapped = 0
    for blk in sam_model.image_encoder.blocks:
        orig_forward = blk.forward

        def make_ckpt_forward(fwd):
            def ckpt_forward(x):
                return ckpt.checkpoint(fwd, x, use_reentrant=False)
            return ckpt_forward

        blk.forward = make_ckpt_forward(orig_forward)
        n_wrapped += 1

    if verbose:
        print(f"[sam_lora] gradient checkpointing enabled on {n_wrapped} image-encoder blocks")
