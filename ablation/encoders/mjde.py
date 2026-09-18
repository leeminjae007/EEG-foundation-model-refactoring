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


class SinglePathMJDE(nn.Module):
    """Parameter-matched six-stage S2T-only or T2S-only encoder.

    The original encoder contains six spatial and six temporal blocks split
    across its two three-stage paths. Reusing all of them in one path keeps
    the 12 attention/MLP blocks while changing only their execution order.
    """

    def __init__(self, original, order):
        super().__init__()
        if order not in ("s2t", "t2s"):
            raise ValueError("Single-path order must be s2t or t2s")
        self.order = order
        self.spatial = nn.ModuleList(
            list(original.s2t_spatial) + list(original.t2s_spatial)
        )
        self.temporal = nn.ModuleList(
            list(original.s2t_temporal) + list(original.t2s_temporal)
        )
        self.output_norm = original.output_norm

    def forward(self, tokens, visible):
        mask = visible.unsqueeze(-1)
        fused = tokens * mask
        for spatial, temporal in zip(self.spatial, self.temporal):
            if self.order == "s2t":
                fused = temporal(spatial(fused, visible), visible)
            else:
                fused = spatial(temporal(fused, visible), visible)
        return self.output_norm(fused) * mask


class AverageMJDE(nn.Module):
    """Original dual paths with a fixed element-wise 1:1 average."""

    def __init__(self, original):
        super().__init__()
        for name in ("s2t_spatial", "s2t_temporal", "t2s_temporal", "t2s_spatial"):
            setattr(self, name, getattr(original, name))
        self.output_norm = original.output_norm

    def forward(self, tokens, visible):
        mask = visible.unsqueeze(-1)
        fused = tokens * mask
        for stage in range(len(self.s2t_spatial)):
            s2t = self.s2t_temporal[stage](
                self.s2t_spatial[stage](fused, visible), visible
            )
            t2s = self.t2s_spatial[stage](
                self.t2s_temporal[stage](fused, visible), visible
            )
            fused = (s2t + t2s) * 0.5 * mask
        return self.output_norm(fused) * mask


class Mix1OnlyMJDE(nn.Module):
    """Keep both three-stage paths separate; average only their final outputs."""

    def __init__(self, original):
        super().__init__()
        for name in ("s2t_spatial", "s2t_temporal", "t2s_temporal", "t2s_spatial"):
            setattr(self, name, getattr(original, name))
        self.output_norm = original.output_norm

    def forward(self, tokens, visible):
        mask = visible.unsqueeze(-1)
        s2t = tokens * mask
        t2s = tokens * mask
        for stage in range(len(self.s2t_spatial)):
            s2t = self.s2t_temporal[stage](
                self.s2t_spatial[stage](s2t, visible), visible
            ) * mask
            t2s = self.t2s_spatial[stage](
                self.t2s_temporal[stage](t2s, visible), visible
            ) * mask
        return self.output_norm((s2t + t2s) * 0.5 * mask) * mask


class MaskProtocol(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, tokens, visible):
        tokens = tokens.masked_fill(~visible.unsqueeze(-1), 0)
        result = self.core(tokens, visible)
        return result.masked_fill(~visible.unsqueeze(-1), 0)
