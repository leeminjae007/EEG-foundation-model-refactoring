"""Learnable and parameter-free fusion of complementary feature branches."""

import torch
import torch.nn as nn


class IdentityLinear(nn.Linear):
    """Bias-free square projection initialized as the identity map."""

    preserve_identity_initialization = True

    def __init__(self, dim):
        super().__init__(int(dim), int(dim), bias=False)
        nn.init.eye_(self.weight)


class TwoBranchFusion(nn.Module):
    """Fuse two last-dimension feature tensors into ``output_dim``."""

    MODES = {"concat", "concat_mixer", "full_add"}

    def __init__(self, first_dim, second_dim, output_dim, mode):
        super().__init__()
        self.first_dim = int(first_dim)
        self.second_dim = int(second_dim)
        self.output_dim = int(output_dim)
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported two-branch fusion: {self.mode}")
        if self.mode in {"concat", "concat_mixer"}:
            if self.first_dim + self.second_dim != self.output_dim:
                raise ValueError("concatenated branch dimensions must sum to output_dim")
        elif self.first_dim != self.output_dim or self.second_dim != self.output_dim:
            raise ValueError("full_add requires two full-width branches")
        self.mixer = (
            IdentityLinear(self.output_dim)
            if self.mode == "concat_mixer" else nn.Identity()
        )

    def forward(self, first, second):
        shape = torch.broadcast_shapes(first.shape[:-1], second.shape[:-1])
        first = first.expand(*shape, first.shape[-1])
        second = second.expand(*shape, second.shape[-1])
        if self.mode == "full_add":
            return first + second
        return self.mixer(torch.cat((first, second), dim=-1))
