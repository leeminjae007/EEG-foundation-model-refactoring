"""Composition of the canonical EEG tokenizer, SHPE, and Dual-3 encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dual_path_encoder import DualPathLatentEncoder
from src.models.positional_encoding import FactorizedChannelSphericalPositionalEncoder
from src.models.time_frequency_tokenizer import (
    CBraModTimeFrequencyPatchEncoder,
    SharedTimeFrequencyPatchEncoder,
)
from src.utils.electrode_geometry import resolve_channel_coordinates


class ChannelEEGEncoder(nn.Module):
    """Encode ``[B,C,L]`` EEG as coordinate-aware ``[B,C,T,D]`` tokens."""

    def __init__(self, patch_encoder, positional_encoder, context_encoder, default_channel_names):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.positional_encoder = positional_encoder
        self.context_encoder = context_encoder
        coordinates, valid = resolve_channel_coordinates(tuple(default_channel_names))
        if not valid.all():
            missing = [
                name for name, available in zip(default_channel_names, valid)
                if not available
            ]
            raise ValueError(f"missing electrode coordinates: {missing}")
        self.register_buffer("default_channel_coordinates", coordinates)

    @property
    def embed_dim(self):
        return self.patch_encoder.embed_dim

    @property
    def num_context_tokens(self):
        return self.default_channel_coordinates.shape[0]

    def expand_region_mask(self, mask):
        return mask

    def _metadata(self, x, channel_coordinates, channel_validity):
        batch, channels, _ = x.shape
        coordinates = (
            self.default_channel_coordinates
            if channel_coordinates is None else channel_coordinates
        )
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1)
        if coordinates.shape != (batch, channels, 3):
            raise ValueError("channel_coordinates must have shape [B,C,3]")
        validity = channel_validity
        if validity is None:
            validity = torch.ones(batch, channels, dtype=torch.bool, device=x.device)
        if validity.shape != (batch, channels) or validity.dtype != torch.bool:
            raise ValueError("channel_validity must be boolean [B,C]")
        return coordinates, validity

    def valid_token_mask(
        self,
        batch_size,
        num_patches,
        channel_region_ids=None,
        channel_validity=None,
    ):
        del channel_region_ids
        if channel_validity is None:
            channel_validity = torch.ones(
                int(batch_size),
                self.num_context_tokens,
                dtype=torch.bool,
                device=self.default_channel_coordinates.device,
            )
        elif channel_validity.ndim == 1:
            channel_validity = channel_validity.unsqueeze(0).expand(int(batch_size), -1)
        return channel_validity[:, :, None].expand(-1, -1, int(num_patches))

    def encode_tokens(
        self,
        x,
        channel_coordinates=None,
        channel_region_ids=None,
        channel_validity=None,
        visible_mask=None,
        return_latent_validity=False,
        return_auxiliary=False,
    ):
        del channel_region_ids
        coordinates, validity = self._metadata(x, channel_coordinates, channel_validity)
        safe_x = x.masked_fill(~validity[:, :, None], 0.0)
        patch_kwargs = {}
        if isinstance(self.patch_encoder, CBraModTimeFrequencyPatchEncoder):
            patch_kwargs["visible_mask"] = visible_mask
        if return_auxiliary:
            tokens, diagnostics = self.patch_encoder(
                safe_x, return_diagnostics=True, **patch_kwargs
            )
            normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
            channel_vectors = normalized.permute(0, 2, 1, 3)
            cosine = torch.matmul(channel_vectors, channel_vectors.transpose(-1, -2))
            pair_mask = validity[:, None, :, None] & validity[:, None, None, :]
            diagonal = torch.eye(
                validity.shape[1], dtype=torch.bool, device=x.device
            )[None, None]
            values = cosine[(pair_mask & ~diagonal).expand_as(cosine)]
            diagnostics["mean_absolute_cosine_similarity"] = (
                values.abs().mean() if values.numel() else tokens.new_zeros(())
            ).detach()
        else:
            tokens = self.patch_encoder(safe_x, **patch_kwargs)
        if return_auxiliary:
            tokens, position_diagnostics = self.positional_encoder(
                tokens, coordinates, return_diagnostics=True
            )
            diagnostics.update(position_diagnostics)
        else:
            tokens = self.positional_encoder(tokens, coordinates)
        tokens = tokens * validity[:, :, None, None]
        if return_auxiliary:
            return tokens, validity, diagnostics
        if return_latent_validity:
            return tokens, validity
        return tokens

    def forward(
        self,
        x,
        channel_coordinates=None,
        channel_region_ids=None,
        channel_validity=None,
        visible_mask=None,
        return_valid_token_mask=False,
        return_auxiliary=False,
        return_branch_outputs=False,
    ):
        result = self.encode_tokens(
            x,
            channel_coordinates,
            channel_region_ids,
            channel_validity,
            visible_mask,
            return_latent_validity=True,
            return_auxiliary=return_auxiliary,
        )
        if return_auxiliary:
            tokens, validity, diagnostics = result
        else:
            tokens, validity = result
        valid_token_mask = validity[:, :, None].expand(-1, -1, tokens.shape[2])
        effective_mask = valid_token_mask
        if visible_mask is not None:
            effective_mask = visible_mask & valid_token_mask
        coordinates, _ = self._metadata(x, channel_coordinates, channel_validity)
        encoded = self.context_encoder(
            tokens,
            visible_mask=effective_mask,
            channel_coordinates=coordinates,
            return_branch_outputs=return_branch_outputs,
            absolute_positions=self.positional_encoder.last_positions,
        )
        if return_branch_outputs:
            branch_outputs = encoded
            encoded = branch_outputs["main"]
        if return_auxiliary:
            diagnostics.update(self.context_encoder.fusion_diagnostics())
            if return_branch_outputs:
                return encoded, valid_token_mask, diagnostics, branch_outputs
            return encoded, valid_token_mask, diagnostics
        if return_branch_outputs:
            return encoded, valid_token_mask, branch_outputs
        if return_valid_token_mask:
            return encoded, valid_token_mask
        return encoded


def build_eeg_encoder(config):
    """Build the single retained D200 Dual-3 EEG encoder."""
    patch = config["patch_encoder"]
    encoder = config["encoder"]
    position = config["position"]
    if config["latent_tokenizer"].get("mode") != "none":
        raise ValueError("only channel-token mode is retained")
    if encoder.get("architecture", "s2t_t2s") != "s2t_t2s":
        raise ValueError("only the canonical s2t_t2s architecture is retained")
    tokenizer_architecture = patch.get("architecture", "project_dual_concat")
    if tokenizer_architecture == "cbramod":
        patch_encoder = CBraModTimeFrequencyPatchEncoder(
            patch_samples=patch["patch_samples"],
            embed_dim=patch["embed_dim"],
            dropout=patch["dropout"],
        )
    else:
        patch_encoder = SharedTimeFrequencyPatchEncoder(
            patch_samples=patch["patch_samples"],
            embed_dim=patch["embed_dim"],
            time_conv_channels=patch["time_conv_channels"],
            time_kernel_sizes=patch["time_kernel_sizes"],
            time_strides=patch["time_strides"],
            time_group_norm_groups=patch["time_group_norm_groups"],
            frequency_bins=patch["frequency_bins"],
            frequency_hidden_dim=patch["frequency_hidden_dim"],
            fusion=patch["fusion"],
            dropout=patch["dropout"],
            time_feature_dim=patch["time_feature_dim"],
            frequency_feature_dim=patch["frequency_feature_dim"],
            frequency_input_transform=patch.get(
                "frequency_input_transform", "log1p_magnitude"
            ),
            frequency_projection=patch.get("frequency_projection", "mlp"),
        )
    positional_encoder = FactorizedChannelSphericalPositionalEncoder(
        embed_dim=encoder["embed_dim"],
        spatial_dim=position["encoder_spatial_dim"],
        temporal_dim=position["encoder_temporal_dim"],
        temporal_max_period=position["temporal_max_period"],
        max_degree=position["spherical_harmonic_max_degree"],
        fusion=position.get("fusion", "concat"),
        component_normalization=position.get("component_normalization", "none"),
        component_scaling=position.get("component_scaling", "none"),
        rms_epsilon=position.get("rms_epsilon", 1e-6),
        post_fusion_activation=position.get("post_fusion_activation", "none"),
        post_fusion_normalization=position.get("post_fusion_normalization", "none"),
        init_std=(
            encoder.get("init_std", 0.02)
            if config["experiment"].get("profile")
            == "historical_d192_reproduction"
            else None
        ),
    )
    positional_encoder.injection = position.get('encoder_injection', 'additive')
    context_encoder = DualPathLatentEncoder(
        embed_dim=encoder["embed_dim"],
        absolute_pe_attention_bias=position.get('encoder_injection', 'additive') == 'attention_bias',
        depth=encoder["depth"],
        spatial_heads=encoder["spatial_heads"],
        temporal_heads=encoder["temporal_heads"],
        mlp_ratio=encoder["mlp_ratio"],
        dropout=encoder["dropout"],
        attention_dropout=encoder["attention_dropout"],
        fusion=encoder["fusion"],
        fusion_schedule=encoder["fusion_schedule"],
        drop_path_rate=encoder["drop_path_rate"],
        norm_epsilon=encoder["norm_epsilon"],
        norm_type=encoder.get("norm_type", "rms_norm"),
        spatial_attention_sh_bias=position.get("spatial_attention_sh_bias", False),
        spherical_harmonic_max_degree=position.get(
            "spherical_harmonic_max_degree", 4
        ),
        temporal_attention_sincos_bias=position.get(
            "temporal_attention_sincos_bias", False
        ),
        temporal_dim=position["encoder_temporal_dim"],
        temporal_max_period=position["temporal_max_period"],
    )
    return ChannelEEGEncoder(
        patch_encoder,
        positional_encoder,
        context_encoder,
        default_channel_names=config["data"]["channel_names"],
    )
