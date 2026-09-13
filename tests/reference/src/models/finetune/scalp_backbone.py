"""Raw-EEG encoder composition and downstream transfer classifier."""

import torch
import torch.nn as nn

from src.models.eeg_encoder import build_eeg_encoder
from src.utils.downstream_diagnostics import representation_geometry


def build_raw_eeg_encoder(config):
    return build_eeg_encoder(config)


def load_pretrained_encoder(encoder, checkpoint_path):
    """Strictly load the complete raw-EEG context encoder state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["context_encoder"], strict=True)
    return checkpoint


class RepresentationFlatten(nn.Module):
    """Flatten a [B,C,T,D] representation in CSBrain/CBraMod order."""

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                "representation flatten expects features with shape [B,C,T,D]"
            )
        # CSBrain explicitly makes the representation contiguous before view.
        # Doing the same makes the C -> T -> D flattening order unambiguous even
        # when an encoder returns a non-contiguous tensor.
        return features.contiguous().view(features.shape[0], -1)


def projection_activation_layers(input_dim, output_dim, activation):
    """Build the retained linear-plus-GELU downstream projection."""
    if activation != "gelu":
        raise ValueError("only GELU is retained")
    return [nn.Linear(input_dim, output_dim), nn.GELU()]


class DownstreamClassifier(nn.Module):
    """Fine-tune the EEG encoder with a configured downstream head."""

    def __init__(
        self,
        encoder,
        embed_dim,
        num_outputs,
        dropout,
        pooling,
        num_latents,
        num_patches,
        head_hidden_tokens=None,
        head_activation="gelu",
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.diagnostics_enabled = False
        self.last_diagnostics = {}
        if pooling != "all_patch_reps":
            raise ValueError("only all_patch_reps pooling is retained")
        input_dim = num_latents * num_patches * embed_dim
        default_hidden_tokens = num_patches
        hidden_tokens = (
            default_hidden_tokens if head_hidden_tokens is None else int(head_hidden_tokens)
        )
        if hidden_tokens <= 0:
            raise ValueError('head_hidden_tokens must be positive')
        temporal_dim = hidden_tokens * embed_dim
        self.head = nn.Sequential(
            RepresentationFlatten(),
            *projection_activation_layers(input_dim, temporal_dim, head_activation),
            nn.Dropout(dropout),
            *projection_activation_layers(temporal_dim, embed_dim, head_activation),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_outputs),
        )

    def forward(
        self,
        x,
        channel_coordinates,
        channel_region_ids,
        channel_validity,
    ):
        encoder_auxiliary = {}
        if self.diagnostics_enabled:
            features, valid_token_mask, encoder_auxiliary = self.encoder(
                x,
                channel_coordinates=channel_coordinates,
                channel_region_ids=channel_region_ids,
                channel_validity=channel_validity,
                return_auxiliary=True,
            )
        else:
            features, valid_token_mask = self.encoder(
                x,
                channel_coordinates=channel_coordinates,
                channel_region_ids=channel_region_ids,
                channel_validity=channel_validity,
                return_valid_token_mask=True,
            )
        if features.ndim != 4:
            raise ValueError("encoder output must have shape [B,K,T,D]")
        weights = valid_token_mask.unsqueeze(-1)
        features = features * weights
        mean_pooled = features.sum(dim=(1, 2))
        mean_pooled = mean_pooled / weights.sum(
            dim=(1, 2)
        ).clamp_min(1)
        head_input = features
        summary_embedding = mean_pooled
        logits = self.head(head_input)
        # Retain the representation used by downstream diagnostics.
        self.last_summary_embedding = summary_embedding
        if self.diagnostics_enabled:
            with torch.no_grad():
                values = features.float()
                expanded_weights = weights.expand_as(features)
                selected = values[expanded_weights]
                geometry = representation_geometry(summary_embedding)
                self.last_diagnostics = {
                    'encoder_feature_mean': selected.mean().detach(),
                    'encoder_feature_std': selected.std(
                        unbiased=False).detach(),
                    'sample_embedding_variance': summary_embedding.float().var(
                        dim=0, unbiased=False).mean().detach(),
                    'head_input_std': head_input.float().std(
                        unbiased=False).detach(),
                    'head_input_mean': head_input.float().mean().detach(),
                    'head_input_max_abs': head_input.float().abs().max().detach(),
                    'logit_mean': logits.float().mean().detach(),
                    'logit_std': logits.float().std(
                        unbiased=False).detach(),
                    'logit_max_abs': logits.float().abs().max().detach(),
                    'encoder_effective_rank': geometry[
                        'effective_rank'].detach(),
                    'encoder_mean_pairwise_cosine': geometry[
                        'mean_pairwise_cosine'].detach(),
                    **{
                        f'encoder_auxiliary/{key}': value.detach()
                        for key, value in encoder_auxiliary.items()
                        if torch.is_tensor(value) and value.ndim == 0
                    },
                }
        return logits
