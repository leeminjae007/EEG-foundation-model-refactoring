"""Canonical three-stage feature-gated spatial/temporal EEG encoder."""

import torch
import torch.nn as nn

from src.models.normalization import build_norm
from src.utils.model_diagnostics import attention_diagnostics
from src.utils.positional import (
    real_spherical_harmonic_features,
    sinusoidal_values,
)


class LearnableSphericalHarmonicAttentionBias(nn.Module):
    """Fixed SH geometry with learnable per-head degree weights."""

    def __init__(self, num_heads, max_degree):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_degree = int(max_degree)
        if self.max_degree < 1:
            raise ValueError("SH attention bias requires max_degree >= 1")
        # Zero initialization makes GR9-2 exactly GR9-1 at step zero.
        self.degree_weights = nn.Parameter(
            torch.zeros(self.num_heads, self.max_degree)
        )

    def forward(self, coordinates):
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("SH attention bias expects coordinates [B,C,3]")
        basis = real_spherical_harmonic_features(
            coordinates, self.max_degree
        ).to(dtype=self.degree_weights.dtype)
        correlations = []
        offset = 1  # Degree zero is a spatially constant term and is omitted.
        for degree in range(1, self.max_degree + 1):
            width = 2 * degree + 1
            degree_basis = basis[..., offset:offset + width]
            correlations.append(degree_basis @ degree_basis.transpose(-2, -1))
            offset += width
        rho = torch.stack(correlations, dim=1)  # [B,L,C,C]
        return torch.einsum("hl,blij->bhij", self.degree_weights, rho)


class LearnableSinusoidalTemporalAttentionBias(nn.Module):
    """Fixed signed-Δt SinCos relations with learned head-wise logit bias."""

    def __init__(self, num_heads, temporal_dim, temporal_max_period):
        super().__init__()
        self.num_heads = int(num_heads)
        self.temporal_dim = int(temporal_dim)
        self.temporal_max_period = float(temporal_max_period)
        if self.temporal_dim < 2:
            raise ValueError("temporal attention bias requires temporal_dim >= 2")
        self.projection = nn.Linear(self.temporal_dim, self.num_heads, bias=False)
        # A zero scale preserves the GR9-2 temporal-attention function at step 0.
        self.temporal_rel_scale = nn.Parameter(torch.zeros(self.num_heads))
        self._basis_cache = {}

    def _basis(self, length, device):
        key = (int(length), device.type, device.index)
        basis = self._basis_cache.get(key)
        if basis is None:
            positions = torch.arange(length, device=device)
            delta = positions[:, None] - positions[None, :]
            basis = sinusoidal_values(
                delta, self.temporal_dim, self.temporal_max_period,
                torch.float32,
            )
            self._basis_cache[key] = basis
        return basis

    def forward(self, length, device, dtype):
        basis = self._basis(length, device).to(dtype=self.projection.weight.dtype)
        bias = self.projection(basis)
        bias = bias * self.temporal_rel_scale.view(1, 1, -1)
        return bias.permute(2, 0, 1).to(dtype=dtype)


class LearnableAbsolutePEAttentionBias(nn.Module):
    """Head-wise learned diagonal kernels of the actual concatenated absolute PE.

    Both axial kernels use the full post-GELU/RMSNorm PE. Independent zero
    gates preserve a zero initial logit bias; unit diagonal weights allow
    nonzero gate gradients without consuming the common initialization RNG.
    """

    def __init__(self, dim, spatial_heads, temporal_heads):
        super().__init__()
        self.spatial_weights = nn.Parameter(torch.ones(spatial_heads, dim))
        self.temporal_weights = nn.Parameter(torch.ones(temporal_heads, dim))
        self.spatial_gate = nn.Parameter(torch.zeros(spatial_heads))
        self.temporal_gate = nn.Parameter(torch.zeros(temporal_heads))
        self.scale = int(dim) ** -0.5

    def forward(self, positions):
        batch, channels, patches, _ = positions.shape
        spatial = torch.einsum('bctd,hd,bktd->bthck',
            positions, self.spatial_weights.to(positions.dtype), positions)
        temporal = torch.einsum('bctd,hd,bcud->bchtu',
            positions, self.temporal_weights.to(positions.dtype), positions)
        spatial = spatial * self.spatial_gate[None, None, :, None, None] * self.scale
        temporal = temporal * self.temporal_gate[None, None, :, None, None] * self.scale
        return spatial.reshape(batch * patches, -1, channels, channels), temporal.reshape(
            batch * channels, -1, patches, patches)


class MaskedSelfAttention(nn.Module):
    """Self-attention that excludes invalid keys and zeros invalid queries."""

    def __init__(self, dim, num_heads, attention_dropout):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dimension must be divisible by heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(int(dim), 3 * int(dim))
        self.output = nn.Linear(int(dim), int(dim))
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.capture_diagnostics = False
        self.last_model_diagnostics = {}

    def forward(self, tokens, valid_mask, attention_bias=None):
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        qk_diagnostic = scores.detach() if self.capture_diagnostics else None
        if attention_bias is not None:
            scores = scores + attention_bias.to(dtype=scores.dtype)
        key_mask = valid_mask[:, None, None]
        scores = scores.masked_fill(~key_mask, -torch.inf)
        maximum = scores.amax(dim=-1, keepdim=True)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        weights = torch.exp(scores - maximum)
        weights = torch.where(key_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.capture_diagnostics:
            self.last_model_diagnostics = attention_diagnostics(
                qk_diagnostic, attention_bias, weights, valid_mask)
        weights = self.attention_dropout(weights)
        output = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.output(output) * valid_mask.unsqueeze(-1)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return self.layers(tokens)


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, tokens):
        if not self.training or self.probability == 0.0:
            return tokens
        keep = 1.0 - self.probability
        shape = (tokens.shape[0],) + (1,) * (tokens.ndim - 1)
        return tokens * tokens.new_empty(shape).bernoulli_(keep) / keep


class MaskedTransformerBlock(nn.Module):
    """Pre-norm masked attention and GELU feed-forward residual block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        attention_dropout,
        drop_path=0.0,
        norm_epsilon=1e-5,
        norm_type="rms_norm",
    ):
        super().__init__()
        self.attention_norm = build_norm(dim, norm_type, norm_epsilon)
        self.attention = MaskedSelfAttention(dim, num_heads, attention_dropout)
        self.mlp_norm = build_norm(dim, norm_type, norm_epsilon)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens, valid_mask, attention_bias=None):
        mask = valid_mask.unsqueeze(-1)
        tokens = tokens * mask
        tokens = tokens + self.drop_path(
            self.attention(
                self.attention_norm(tokens), valid_mask, attention_bias
            )
        )
        tokens = tokens + self.drop_path(self.mlp(self.mlp_norm(tokens)))
        return tokens * mask


class AxisTransformerBlock(nn.Module):
    """Apply one Transformer block independently along a grid axis."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        attention_dropout,
        drop_path=0.0,
        norm_epsilon=1e-5,
        norm_type="rms_norm",
    ):
        super().__init__()
        self.block = MaskedTransformerBlock(
            dim,
            num_heads,
            mlp_ratio,
            dropout,
            attention_dropout,
            drop_path,
            norm_epsilon,
            norm_type,
        )

    def forward(self, grid, valid_mask, axis, attention_bias=None, bias_per_axis_row=False):
        batch, channels, patches, dim = grid.shape
        if axis == "spatial":
            tokens = grid.permute(0, 2, 1, 3).reshape(
                batch * patches, channels, dim
            )
            mask = valid_mask.permute(0, 2, 1).reshape(
                batch * patches, channels
            )
            bias = None
            if attention_bias is not None:
                bias = (attention_bias if bias_per_axis_row else
                        attention_bias.repeat_interleave(patches, dim=0))
            output = self.block(tokens, mask, attention_bias=bias)
            return output.reshape(batch, patches, channels, dim).permute(
                0, 2, 1, 3
            )
        if axis != "temporal":
            raise ValueError("axis must be spatial or temporal")
        tokens = grid.reshape(batch * channels, patches, dim)
        mask = valid_mask.reshape(batch * channels, patches)
        return self.block(tokens, mask, attention_bias=attention_bias).reshape(
            batch, channels, patches, dim
        )


class DualPathLatentEncoder(nn.Module):
    """Dual-3 encoder with stage-wise feature-gated S2T/T2S branches."""

    def __init__(
        self,
        embed_dim,
        depth,
        spatial_heads,
        temporal_heads,
        mlp_ratio,
        dropout,
        attention_dropout,
        fusion,
        fusion_schedule="stagewise",
        drop_path_rate=0.0,
        norm_epsilon=1e-5,
        norm_type="rms_norm",
        spatial_attention_sh_bias=False,
        spherical_harmonic_max_degree=4,
        temporal_attention_sincos_bias=False,
        temporal_dim=100,
        temporal_max_period=10000.0,
        absolute_pe_attention_bias=False,
    ):
        super().__init__()
        if depth % 4 or fusion != "gated_sum" or fusion_schedule != "stagewise":
            raise ValueError("canonical Dual-3 requires gated stage-wise depth/4")
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        stages = depth // 4
        rates = torch.linspace(0.0, drop_path_rate, stages).tolist()

        def make(heads, rate):
            return AxisTransformerBlock(
                embed_dim,
                heads,
                mlp_ratio,
                dropout,
                attention_dropout,
                drop_path=rate,
                norm_epsilon=norm_epsilon,
                norm_type=norm_type,
            )

        self.s2t_spatial = nn.ModuleList([
            make(spatial_heads, rate) for rate in rates
        ])
        self.s2t_temporal = nn.ModuleList([
            make(temporal_heads, rate) for rate in rates
        ])
        self.t2s_temporal = nn.ModuleList([
            make(temporal_heads, rate) for rate in rates
        ])
        self.t2s_spatial = nn.ModuleList([
            make(spatial_heads, rate) for rate in rates
        ])
        self.fusion_gates = nn.Parameter(torch.zeros(stages, embed_dim))
        self.output_norm = build_norm(embed_dim, norm_type, norm_epsilon)
        self.spatial_attention_bias = (
            LearnableSphericalHarmonicAttentionBias(
                spatial_heads, spherical_harmonic_max_degree
            ) if spatial_attention_sh_bias else None
        )
        self.temporal_attention_bias = (
            LearnableSinusoidalTemporalAttentionBias(
                temporal_heads, temporal_dim, temporal_max_period
            ) if temporal_attention_sincos_bias else None
        )
        self._gate_records = []
        self.absolute_attention_bias = (
            LearnableAbsolutePEAttentionBias(embed_dim, spatial_heads, temporal_heads)
            if absolute_pe_attention_bias else None
        )

    def forward(
        self, tokens, visible_mask, channel_coordinates=None,
        return_branch_outputs=False,
        absolute_positions=None,
    ):
        mask = visible_mask.unsqueeze(-1)
        fused = tokens * mask
        attention_bias = None
        if self.spatial_attention_bias is not None:
            if channel_coordinates is None:
                raise ValueError("SH attention bias requires channel coordinates")
            attention_bias = self.spatial_attention_bias(channel_coordinates)
        temporal_attention_bias = None
        if self.temporal_attention_bias is not None:
            temporal_attention_bias = self.temporal_attention_bias(
                tokens.shape[2], tokens.device, tokens.dtype
            )
        per_axis_bias = self.absolute_attention_bias is not None
        if per_axis_bias:
            if absolute_positions is None:
                raise ValueError('Absolute PE attention bias requires the original position grid')
            attention_bias, temporal_attention_bias = self.absolute_attention_bias(absolute_positions)
        self._gate_records = []
        stages = zip(
            self.s2t_spatial,
            self.s2t_temporal,
            self.t2s_temporal,
            self.t2s_spatial,
        )
        final_s2t = final_t2s = None
        for index, (ss, st, tt, ts) in enumerate(stages):
            s2t = st(
                ss(fused, visible_mask, "spatial", attention_bias, per_axis_bias),
                visible_mask, "temporal", temporal_attention_bias
            )
            t2s = ts(
                tt(fused, visible_mask, "temporal", temporal_attention_bias),
                visible_mask, "spatial",
                attention_bias, per_axis_bias
            )
            gate = self.fusion_gates[index].sigmoid().view(1, 1, 1, -1)
            fused = (gate * s2t + (1.0 - gate) * t2s) * mask
            if index == len(self.s2t_spatial) - 1:
                # These are deliberately pre-gate.  The *same* output norm is
                # applied below; no new parameters are introduced.
                final_s2t, final_t2s = s2t, t2s
            self._gate_records.append(gate)
        main = self.output_norm(fused) * mask
        if not return_branch_outputs:
            return main
        return {
            "main": main,
            "s2t": self.output_norm(final_s2t) * mask,
            "t2s": self.output_norm(final_t2s) * mask,
        }

    def fusion_diagnostics(self):
        if not self._gate_records:
            return {}
        gates = torch.stack(self._gate_records)
        diagnostics = {
            "fusion_s2t_weight": gates.mean().detach(),
            "fusion_weight_std": gates.std(unbiased=False).detach(),
        }
        if self.spatial_attention_bias is not None:
            diagnostics["sh_attention_bias_weight_abs_mean"] = (
                self.spatial_attention_bias.degree_weights.abs().mean().detach()
            )
        if self.temporal_attention_bias is not None:
            diagnostics["temporal_attention_bias_scale_abs_mean"] = (
                self.temporal_attention_bias.temporal_rel_scale.abs().mean().detach()
            )
        return diagnostics
