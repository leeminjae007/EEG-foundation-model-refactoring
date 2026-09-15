"""Reuse the current MJDE blocks and expose a one-stage variant."""

import torch
from torch import nn


class MJDELite(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("s2t_spatial", "s2t_temporal", "t2s_temporal", "t2s_spatial"):
            setattr(self, name, nn.ModuleList([getattr(original, name)[0]]))
        self.fusion_gates = nn.Parameter(original.fusion_gates[:1].detach().clone())
        self.output_norm = original.output_norm

    def forward(self, tokens, visible):
        mask = visible.unsqueeze(-1)
        fused = tokens * mask
        s2t = self.s2t_temporal[0](self.s2t_spatial[0](fused, visible), visible)
        t2s = self.t2s_spatial[0](self.t2s_temporal[0](fused, visible), visible)
        gate = self.fusion_gates[0].sigmoid().view(1, 1, 1, -1)
        fused = (gate * s2t + (1.0 - gate) * t2s) * mask
        return self.output_norm(fused) * mask


class MaskProtocol(nn.Module):
    def __init__(self, core, mode):
        super().__init__()
        self.core = core
        self.mode = mode

    def forward(self, tokens, visible):
        tokens = tokens.masked_fill(~visible.unsqueeze(-1), 0)
        attention_visible = torch.ones_like(visible) if self.mode == "dense_zero" else visible
        result = self.core(tokens, attention_visible)
        return result.masked_fill(~visible.unsqueeze(-1), 0)
