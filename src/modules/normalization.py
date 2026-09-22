"""마지막 feature 축을 FP32로 RMS 정규화한다."""

import torch
from torch import nn


class RMSNorm(nn.Module):
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
