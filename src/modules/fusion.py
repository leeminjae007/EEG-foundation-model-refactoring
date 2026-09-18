"""Input-conditioned MJDE route mixing, shared across electrode/time positions."""
import torch
from torch import nn
from torch.nn import functional as F


FUSION_GATE_MODES = ("static_feature", "patch_scalar", "patch_feature")


def fusion_gate_mode(config):
    mode = config.get("fusion_gate", "static_feature")
    if mode not in FUSION_GATE_MODES:
        raise ValueError("Unknown encoder.fusion_gate: " + str(mode))
    return mode


class PatchFusionGate(nn.Module):
    """sigmoid(Linear([S2T, T2S])); one or D mixing coefficients per patch.

    Explicit zero Parameters avoid consuming RNG or being overwritten by the
    model-wide Linear initializer. Shared blocks therefore initialize exactly
    like the static baseline, and every route initially receives weight 0.5.
    """
    def __init__(self, dim, output_dim):
        super().__init__()
        if output_dim not in (1, dim):
            raise ValueError("A patch gate must output one or D coefficients")
        self.weight = nn.Parameter(torch.zeros(output_dim, 2 * dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.capture = False
        self.diagnostics = {}

    def forward(self, s2t, t2s, visible):
        gate = F.linear(torch.cat((s2t, t2s), dim=-1), self.weight, self.bias).sigmoid()
        if self.capture:
            with torch.no_grad():
                values = gate.detach()[visible].float()
                self.diagnostics = {"gate/visible_patches": values.shape[0]}
                if values.numel():
                    self.diagnostics.update({
                        "gate/mean": values.mean(),
                        "gate/std": values.std(unbiased=False),
                        "gate/min": values.min(), "gate/max": values.max(),
                        "gate/patch_std_mean": values.std(dim=0, unbiased=False).mean(),
                        "gate/feature_std_mean": values.std(dim=-1, unbiased=False).mean(),
                        "gate/saturated_fraction": ((values < .01) | (values > .99)).float().mean(),
                    })
        return gate
