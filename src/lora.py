"""
lora.py

LoRA (Low-Rank Adaptation) for CLIP's visual encoder.

WHY LoRA?
  Fine-tuning all of CLIP (150M params) on 80K VQA samples causes
  catastrophic forgetting and severe overfitting. LoRA instead freezes
  CLIP completely and injects tiny trainable matrices (A, B) alongside
  frozen weight matrices. The adaptation is:

      output = x @ W.T + x @ A.T @ B.T
             = x @ (W + BA).T

  Where W is frozen (512×512), A is (rank×512), B is (512×rank).
  With rank=8, each injected pair adds only 8192 params vs 262K for W.

WHY last 4 blocks only?
  Early transformer blocks learn low-level universal features
  (edges, textures, shapes). Last blocks learn high-level semantic
  features — these need task-specific adaptation for VQA.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen Linear layer and adds a trainable low-rank delta.
    
    Forward pass computes:
        output = x @ W.T  (frozen)  +  x @ A.T @ B.T  (trainable)
    
    This is equivalent to W + BA without ever modifying W.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()

        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_features  = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * (1.0 / math.sqrt(rank))
        )
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        for param in self.linear.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        # WHY? PyTorch's MultiheadAttention accesses out_proj.weight
        # directly internally. We expose the frozen weight so nothing breaks.
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        frozen_out = self.linear(x)
        lora_out   = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return frozen_out + lora_out


def inject_lora_into_clip(clip_model, rank: int = 8, alpha: float = 16.0,
                           num_blocks: int = 4):
    """Freeze all of CLIP and inject LoRA into the last num_blocks
    transformer blocks of the visual encoder.
    
    WHY this function?
    We want a clean, reusable way to apply LoRA to CLIP without
    manually touching each layer. This function finds the right layers
    automatically and replaces them with LoRALinear wrappers.
    """

    # Step 1: freeze everything in CLIP
    for param in clip_model.parameters():
        param.requires_grad = False

    visual        = clip_model.visual
    transformer   = visual.transformer
    all_blocks    = transformer.resblocks
    total_blocks  = len(all_blocks)

    # Step 2: inject LoRA only into last num_blocks blocks
    target_blocks = list(all_blocks)[-num_blocks:]

    injected = 0
    for block in target_blocks:
        # each transformer block has:
        #   block.attn  — multi-head self-attention
        #   block.mlp   — feedforward network

        # inject into attention output projection
        if hasattr(block.attn, "out_proj"):
            block.attn.out_proj = LoRALinear(
                block.attn.out_proj, rank=rank, alpha=alpha
            )
            injected += 1

        # inject into MLP projections
        if hasattr(block.mlp, "c_fc"):
            block.mlp.c_fc = LoRALinear(
                block.mlp.c_fc, rank=rank, alpha=alpha
            )
            injected += 1

        if hasattr(block.mlp, "c_proj"):
            block.mlp.c_proj = LoRALinear(
                block.mlp.c_proj, rank=rank, alpha=alpha
            )
            injected += 1

    # count trainable params for verification
    trainable = sum(p.numel() for p in clip_model.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in clip_model.parameters())

    print(f"LoRA injected into {injected} layers across last {num_blocks} blocks")
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return clip_model


def get_lora_params(clip_model):
    """Return only the trainable LoRA parameters.
    
    WHY a separate function?
    When building the optimizer, we need to separate LoRA params
    from fusion head params — they get different learning rates.
    """
    return [p for p in clip_model.parameters() if p.requires_grad]