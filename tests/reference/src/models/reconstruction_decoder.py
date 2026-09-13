"""Lightweight decoder for masked raw EEG patch reconstruction."""

import torch
import torch.nn as nn

from src.models.normalization import build_norm
from src.models.feature_fusion import TwoBranchFusion
from src.models.normalization import RMSNorm
from src.models.positional_encoding import SphericalHarmonicExpansion
from src.models.transformer_blocks import (
    TransformerBlock,
)
from src.utils.positional import (
    sinusoidal_positions,
    unit_rms,
)
from src.utils.reconstruction import gather_target_blocks


class MaskedPatchDecoder(nn.Module):
    """Decode a target patch set from the shared visible token grid."""

    def __init__(
        self,
        embed_dim,
        decoder_dim,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias,
        dropout,
        attention_dropout,
        norm_epsilon,
        temporal_max_period,
        default_channel_coordinates,
        position_type,
        spherical_harmonic_max_degree,
        spatial_position_dim,
        temporal_position_dim,
        position_fusion="concat",
        position_component_normalization="none",
        position_component_scaling="none",
        position_rms_epsilon=1e-6,
        position_post_fusion_activation="none",
        position_post_fusion_normalization="none",
        output_dim=200,
        norm_type="rms_norm",
        init_std=None,
    ):
        super().__init__()
        self.capture_diagnostics = False
        self.last_model_diagnostics = {}
        if position_type != "spherical_harmonic_factorized":
            raise ValueError("only factorized SH decoder PE is retained")
        self.embed_dim = int(embed_dim)
        self.decoder_dim = int(decoder_dim)
        self.output_dim = int(output_dim)
        self.temporal_max_period = float(temporal_max_period)
        self.spatial_position_dim = int(spatial_position_dim)
        self.temporal_position_dim = int(temporal_position_dim)
        self.position_fusion = str(position_fusion)
        self.position_component_normalization = str(
            position_component_normalization
        )
        self.position_component_scaling = str(position_component_scaling)
        self.position_rms_epsilon = float(position_rms_epsilon)
        self.position_post_fusion_activation = str(position_post_fusion_activation)
        self.position_post_fusion_normalization = str(position_post_fusion_normalization)
        self.position_feature_fusion = TwoBranchFusion(
            self.spatial_position_dim,
            self.temporal_position_dim,
            self.decoder_dim,
            self.position_fusion,
        )
        if self.position_component_normalization not in {"none", "rms"}:
            raise ValueError("position component normalization must be none or rms")
        if self.position_component_scaling not in {"none", "learnable"}:
            raise ValueError("position component scaling must be none or learnable")
        if self.position_post_fusion_activation not in {"none", "gelu"}:
            raise ValueError("decoder post-fusion position activation must be none or gelu")
        if self.position_post_fusion_normalization not in {"none", "rms_norm"}:
            raise ValueError("decoder post-fusion position normalization must be none or rms_norm")
        if self.position_component_scaling == "learnable":
            self.position_spatial_alpha = nn.Parameter(torch.ones(()))
            self.position_temporal_beta = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("position_spatial_alpha", None)
            self.register_parameter("position_temporal_beta", None)
        self.spherical_expansion = SphericalHarmonicExpansion(
            self.spatial_position_dim,
            max_degree=spherical_harmonic_max_degree,
            init_std=init_std,
        )
        self.position_post_fusion_activation_layer = (
            nn.GELU() if self.position_post_fusion_activation == "gelu" else nn.Identity()
        )
        self.position_post_fusion_norm = (
            RMSNorm(self.decoder_dim, eps=self.position_rms_epsilon)
            if self.position_post_fusion_normalization == "rms_norm" else nn.Identity()
        )
        self.context_projection = nn.Linear(self.embed_dim, self.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_epsilon=norm_epsilon,
                norm_type=norm_type,
            )
            for _ in range(int(depth))
        ])
        self.norm = build_norm(self.decoder_dim, norm_type, norm_epsilon)
        self.output_projection = nn.Linear(self.decoder_dim, self.output_dim)
        self.register_buffer(
            "default_channel_coordinates",
            default_channel_coordinates.detach().float().clone(),
        )

    def _positions(self, coordinates, batch, patches, dtype):
        spatial = self.spherical_expansion(coordinates).to(dtype=dtype)
        if spatial.ndim == 2:
            spatial = spatial[None]
        temporal = sinusoidal_positions(
            patches,
            self.temporal_position_dim,
            self.temporal_max_period,
            spatial.device,
            dtype,
        )
        spatial = spatial[:, :, None]
        temporal = temporal[None, None]
        if self.position_component_normalization == "rms":
            spatial = unit_rms(spatial, self.position_rms_epsilon)
            temporal = unit_rms(temporal, self.position_rms_epsilon)
        if self.position_component_scaling == "learnable":
            spatial = spatial * self.position_spatial_alpha
            temporal = temporal * self.position_temporal_beta
        positions = self.position_feature_fusion(spatial, temporal)
        positions = self.position_post_fusion_norm(
            self.position_post_fusion_activation_layer(positions)
        )
        if self.capture_diagnostics:
            post = positions.detach().float()
            self.last_model_diagnostics = {'pe/post_transform_total_rms': post.square().mean().sqrt()}
            if self.position_fusion == 'concat':
                self.last_model_diagnostics.update({
                    'pe/post_transform_spatial_rms': post[..., :self.spatial_position_dim].square().mean().sqrt(),
                    'pe/post_transform_temporal_rms': post[..., self.spatial_position_dim:].square().mean().sqrt(),
                })
        return positions.expand(batch, -1, -1, -1)

    def forward(
        self,
        context_grid,
        context_mask,
        target_mask,
        target_blocks=None,
        target_block_valid=None,
        target_token_valid=None,
        channel_coordinates=None,
    ):
        if target_blocks is None:
            target_blocks = target_mask[:, None]
        if (context_mask & target_mask).any():
            raise ValueError("context and target masks overlap")
        if not torch.equal(target_blocks.any(dim=1), target_mask):
            raise ValueError("target_mask must be the target-block union")

        batch, _, patches, _ = context_grid.shape
        context_counts = context_mask.flatten(1).sum(dim=1)
        block_counts = target_blocks.flatten(2).sum(dim=2)
        target_count = int(block_counts.max())
        num_blocks = target_blocks.shape[1]
        expected_block_valid = block_counts.gt(0)
        expected_token_valid = (
            torch.arange(target_count, device=context_grid.device)[None, None]
            < block_counts[:, :, None]
        )
        if target_block_valid is not None and not torch.equal(
            target_block_valid, expected_block_valid
        ):
            raise ValueError("target_block_valid does not match target blocks")
        if target_token_valid is not None and not torch.equal(
            target_token_valid, expected_token_valid
        ):
            raise ValueError("target_token_valid does not match target blocks")

        coordinates = (
            self.default_channel_coordinates
            if channel_coordinates is None else channel_coordinates
        )
        positions = self._positions(coordinates, batch, patches, context_grid.dtype)
        projected = self.context_projection(context_grid) + positions

        equal_context_counts = torch.equal(
            context_counts, context_counts[:1].expand_as(context_counts)
        )
        equal_target_counts = torch.equal(
            block_counts, block_counts[:1, :1].expand_as(block_counts)
        )
        if equal_context_counts and equal_target_counts:
            # Preserve the exact historical decoder path for every existing
            # fixed-count masking policy and checkpoint lineage.
            fixed_context_count = int(context_counts[0])
            fixed_target_count = int(block_counts[0, 0])
            context = projected.flatten(1, 2)[context_mask.flatten(1)].reshape(
                batch, fixed_context_count, self.decoder_dim
            )
            target_positions = gather_target_blocks(positions, target_blocks)
            targets = target_positions.reshape(
                batch * num_blocks, fixed_target_count, self.decoder_dim
            ) + self.mask_token
            context = context[:, None].expand(
                -1, num_blocks, -1, -1
            ).reshape(
                batch * num_blocks, fixed_context_count, self.decoder_dim
            )
            sequence = torch.cat((context, targets), dim=1)
            for block in self.blocks:
                sequence = block(sequence)
            decoded = self.output_projection(
                self.norm(sequence)[:, fixed_context_count:]
            )
            return decoded.reshape(
                batch, num_blocks * fixed_target_count, self.output_dim
            )

        # Variable-count policies retain the original [C,T] grid. Tokens that
        # belong to neither context nor target (possible for I-JEPA-style
        # policies) are excluded from attention rather than treated as masks.
        masked = self.mask_token.view(1, 1, 1, -1) + positions
        active_mask = context_mask | target_mask
        sequence = torch.where(
            context_mask.unsqueeze(-1), projected, masked
        ).flatten(1, 2)
        for block in self.blocks:
            sequence = block(sequence, token_mask=active_mask.flatten(1))
        decoded_grid = self.output_projection(self.norm(sequence)).reshape(
            batch, context_grid.shape[1], patches, self.output_dim
        )
        decoded, gathered_target_valid = gather_target_blocks(
            decoded_grid, target_blocks, return_token_valid=True
        )
        if not torch.equal(gathered_target_valid, expected_token_valid):
            raise RuntimeError("decoder target validity mismatch")
        return decoded
