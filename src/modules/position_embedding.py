"""Pure functions for factorized spherical-harmonic positional encoding."""

import math

import torch


def sinusoidal_values(values, dim, max_period, dtype):
    """Encode arbitrary temporal coordinates with the canonical SinCos basis."""
    half = (dim + 1) // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.to(dtype=torch.float32).unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[..., :dim].to(dtype)


def sinusoidal_positions(length, dim, max_period, device, dtype):
    """Return deterministic temporal sinusoidal positions ``[T, D_t]``."""
    return sinusoidal_values(
        torch.arange(length, device=device), dim, max_period, dtype
    )


def _associated_legendre(degree, order, cosine_theta):
    value = torch.ones_like(cosine_theta)
    if order:
        sine_theta = torch.sqrt(
            ((1.0 - cosine_theta) * (1.0 + cosine_theta)).clamp_min(0.0)
        )
        factor = 1.0
        for _ in range(order):
            value = -factor * sine_theta * value
            factor += 2.0
    if degree == order:
        return value
    current = (2 * order + 1) * cosine_theta * value
    if degree == order + 1:
        return current
    previous = value
    for current_degree in range(order + 2, degree + 1):
        following = (
            (2 * current_degree - 1) * cosine_theta * current
            - (current_degree + order - 1) * previous
        ) / (current_degree - order)
        previous, current = current, following
    return current


def real_spherical_harmonic_features(coordinates, max_degree=4):
    """Evaluate the fixed real SH basis for coordinates ``[..., C, 3]``."""
    xyz = coordinates.detach().float()
    xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = xyz.unbind(dim=-1)
    cosine_theta = z.clamp(-1.0, 1.0)
    phi = torch.atan2(y, x)
    features = []
    for degree in range(int(max_degree) + 1):
        for order in range(-degree, degree + 1):
            absolute_order = abs(order)
            polynomial = _associated_legendre(
                degree, absolute_order, cosine_theta
            )
            scale = math.sqrt(
                (2 * degree + 1)
                / (4.0 * math.pi)
                * math.factorial(degree - absolute_order)
                / math.factorial(degree + absolute_order)
            )
            if order < 0:
                basis = (
                    math.sqrt(2.0) * scale * polynomial
                    * torch.sin(absolute_order * phi)
                )
            elif order == 0:
                basis = scale * polynomial
            else:
                basis = (
                    math.sqrt(2.0) * scale * polynomial
                    * torch.cos(order * phi)
                )
            features.append(basis)
    return torch.stack(features, dim=-1)



from torch import nn
from src.modules.normalization import RMSNorm


class PositionEmbedding(nn.Module):
    """좌표의 SH와 시점의 SinCos를 연결한 뒤 GELU → RMSNorm을 적용한다."""

    def __init__(self, spatial_dim, temporal_dim, max_period, epsilon):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.max_period = max_period
        self.projection = nn.Linear(25, spatial_dim, bias=False)
        self.activation = nn.GELU()
        self.norm = RMSNorm(spatial_dim + temporal_dim, epsilon)
        self.spatial_dim = spatial_dim
        self.capture = False
        self.diagnostics = {}

    def forward(self, coordinates, batch, patches, dtype):
        # 좌표 [C,3] 또는 [B,C,3] → SH [*,C,25] → [*,C,D_s].
        basis = real_spherical_harmonic_features(coordinates, max_degree=4)
        spatial = self.projection(basis).to(dtype=dtype)
        if spatial.ndim == 2:
            spatial = spatial[None]
        temporal = sinusoidal_positions(patches, self.temporal_dim,
                                        self.max_period, spatial.device, dtype)
        spatial = spatial[:, :, None]
        temporal = temporal[None, None]
        shape = torch.broadcast_shapes(spatial.shape[:-1], temporal.shape[:-1])
        spatial = spatial.expand(*shape, spatial.shape[-1])
        temporal = temporal.expand(*shape, temporal.shape[-1])
        positions = torch.cat((spatial, temporal), dim=-1)
        positions = self.activation(positions)
        positions = self.norm(positions)
        if self.capture:
            values = positions.detach().float()
            self.diagnostics = {
                "post_transform_total_rms": values.square().mean().sqrt(),
                "post_transform_spatial_rms": values[..., :self.spatial_dim].square().mean().sqrt(),
                "post_transform_temporal_rms": values[..., self.spatial_dim:].square().mean().sqrt(),
            }
        return positions.expand(batch, -1, -1, -1)
