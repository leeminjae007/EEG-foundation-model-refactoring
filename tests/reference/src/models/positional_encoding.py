"""Learnable modules for factorized spherical-harmonic positional encoding."""

import torch
import torch.nn as nn

from src.models.feature_fusion import TwoBranchFusion
from src.models.normalization import RMSNorm

from src.utils.positional import (
    real_spherical_harmonic_features,
    sinusoidal_positions,
    unit_rms,
)


class SphericalHarmonicExpansion(nn.Module):
    """Project the fixed real SH basis into the model position width."""

    def __init__(self, output_dim, max_degree=4, init_std=None):
        super().__init__()
        self.max_degree = int(max_degree)
        self.sh_dim = (self.max_degree + 1) ** 2
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("SH output dimension must be positive")
        self.projection = nn.Linear(self.sh_dim, self.output_dim, bias=False)
        if init_std is not None:
            nn.init.trunc_normal_(self.projection.weight, std=float(init_std))

    def basis(self, coordinates):
        return real_spherical_harmonic_features(coordinates, self.max_degree)

    def forward(self, coordinates):
        return self.projection(self.basis(coordinates))


class FactorizedChannelSphericalPositionalEncoder(nn.Module):
    """Add concatenated SH-spatial and sinusoidal-temporal PE to a token grid."""

    def __init__(
        self,
        embed_dim,
        spatial_dim,
        temporal_dim,
        temporal_max_period,
        max_degree=4,
        fusion="concat",
        component_normalization="none",
        component_scaling="none",
        rms_epsilon=1e-6,
        post_fusion_activation="none",
        post_fusion_normalization="none",
        init_std=None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.injection = 'additive'
        self.capture_diagnostics = False
        self.last_model_diagnostics = {}
        self.last_positions = None
        self.spatial_dim = int(spatial_dim)
        self.temporal_dim = int(temporal_dim)
        self.temporal_max_period = float(temporal_max_period)
        self.fusion = str(fusion)
        self.component_normalization = str(component_normalization)
        self.component_scaling = str(component_scaling)
        self.rms_epsilon = float(rms_epsilon)
        self.post_fusion_activation = str(post_fusion_activation)
        self.post_fusion_normalization = str(post_fusion_normalization)
        self.feature_fusion = TwoBranchFusion(
            self.spatial_dim, self.temporal_dim, self.embed_dim, self.fusion
        )
        if self.component_normalization not in {"none", "rms"}:
            raise ValueError("position component normalization must be none or rms")
        if self.component_scaling not in {"none", "learnable"}:
            raise ValueError("position component scaling must be none or learnable")
        if self.post_fusion_activation not in {"none", "gelu"}:
            raise ValueError("post-fusion position activation must be none or gelu")
        if self.post_fusion_normalization not in {"none", "rms_norm"}:
            raise ValueError("post-fusion position normalization must be none or rms_norm")
        if self.component_scaling == "learnable":
            self.spatial_alpha = nn.Parameter(torch.ones(()))
            self.temporal_beta = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("spatial_alpha", None)
            self.register_parameter("temporal_beta", None)
        self.spatial = SphericalHarmonicExpansion(
            self.spatial_dim,
            max_degree=max_degree,
            init_std=init_std,
        )
        self.post_fusion_activation_layer = (
            nn.GELU() if self.post_fusion_activation == "gelu" else nn.Identity()
        )
        self.post_fusion_norm = (
            RMSNorm(self.embed_dim, eps=self.rms_epsilon)
            if self.post_fusion_normalization == "rms_norm" else nn.Identity()
        )

    def forward(self, tokens, coordinates, return_diagnostics=False):
        batch, _, patches, _ = tokens.shape
        spatial = self.spatial(coordinates).to(dtype=tokens.dtype)
        if spatial.ndim == 2:
            spatial = spatial[None]
        temporal = sinusoidal_positions(
            patches,
            self.temporal_dim,
            self.temporal_max_period,
            tokens.device,
            tokens.dtype,
        )
        spatial = spatial[:, :, None]
        temporal = temporal[None, None]
        effective_spatial = spatial
        effective_temporal = temporal
        if self.component_normalization == "rms":
            effective_spatial = unit_rms(spatial, self.rms_epsilon)
            effective_temporal = unit_rms(temporal, self.rms_epsilon)
        if self.component_scaling == "learnable":
            effective_spatial = effective_spatial * self.spatial_alpha
            effective_temporal = effective_temporal * self.temporal_beta
        positions = self.feature_fusion(effective_spatial, effective_temporal)
        positions = self.post_fusion_norm(self.post_fusion_activation_layer(positions))
        positions = positions.expand(batch, -1, -1, -1)
        self.last_positions = positions if self.injection == 'attention_bias' else None
        output = tokens if self.injection == 'attention_bias' else tokens + positions
        if self.capture_diagnostics:
            post = positions.detach().float()
            self.last_model_diagnostics = {
                'pe/post_transform_total_rms': post.square().mean().sqrt(),
                'pe/additive_injection': post.new_tensor(float(self.injection == 'additive')),
            }
            if self.fusion == 'concat':
                self.last_model_diagnostics.update({
                    'pe/post_transform_spatial_rms': post[..., :self.spatial_dim].square().mean().sqrt(),
                    'pe/post_transform_temporal_rms': post[..., self.spatial_dim:].square().mean().sqrt(),
                })
        if not return_diagnostics:
            return output
        diagnostics = {
            "position_spatial_rms": spatial.float().square().mean().sqrt().detach(),
            "position_temporal_rms": temporal.float().square().mean().sqrt().detach(),
            "position_effective_spatial_rms": (
                effective_spatial.float().square().mean().sqrt().detach()
            ),
            "position_effective_temporal_rms": (
                effective_temporal.float().square().mean().sqrt().detach()
            ),
            "position_fused_rms": positions.float().square().mean().sqrt().detach(),
        }
        if self.component_scaling == "learnable":
            diagnostics.update({
                "position_spatial_alpha": self.spatial_alpha.detach(),
                "position_temporal_beta": self.temporal_beta.detach(),
            })
        diagnostics["position_temporal_to_spatial_rms"] = (
            diagnostics["position_temporal_rms"]
            / diagnostics["position_spatial_rms"].clamp_min(self.rms_epsilon)
        )
        diagnostics["position_effective_temporal_to_spatial_rms"] = (
            diagnostics["position_effective_temporal_rms"]
            / diagnostics["position_effective_spatial_rms"].clamp_min(
                self.rms_epsilon
            )
        )
        return output, diagnostics
