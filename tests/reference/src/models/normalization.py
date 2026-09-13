"""Normalization layers shared by EEG encoder and decoder blocks."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned feature scale."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, tokens):
        dtype = tokens.dtype
        values = tokens.float()
        values = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (values * self.weight.float()).to(dtype=dtype)


def build_norm(dim, norm_type="rms_norm", norm_epsilon=1e-5):
    """Build canonical RMSNorm or the historical reproduction LayerNorm."""
    if norm_type == "rms_norm":
        return RMSNorm(dim, eps=float(norm_epsilon))
    if norm_type == "layer_norm":
        return nn.LayerNorm(int(dim), eps=float(norm_epsilon))
    raise ValueError(f"unsupported normalization: {norm_type}")
