"""ISRUC sequence-to-sequence full-fine-tuning model."""

import torch
import torch.nn as nn

from src.models.finetune.scalp_backbone import RepresentationFlatten
from src.models.finetune.scalp_backbone import projection_activation_layers


class Model(nn.Module):
    """Classify 20 consecutive sleep epochs with shared temporal context."""

    def __init__(
        self,
        encoder,
        embed_dim,
        num_latents,
        num_patches,
        dropout,
        pooling,
        head_activation="gelu",
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        if pooling != "all_patch_reps":
            raise ValueError("only all_patch_reps pooling is retained")
        epoch_input_dim = num_latents * num_patches * embed_dim
        representation_flatten = RepresentationFlatten()
        modules = {
            "epoch_projection": nn.Sequential(
                *projection_activation_layers(
                    epoch_input_dim, 512, head_activation
                ),
            ),
            "sequence_encoder": nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=4,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                ),
                num_layers=1,
                enable_nested_tensor=False,
            ),
            "classifier": nn.Linear(512, 5),
        }
        modules["representation_flatten"] = representation_flatten
        self.head = nn.ModuleDict(modules)

    @staticmethod
    def _repeat_metadata(value, sequence_length):
        return value[:, None].expand(
            value.shape[0], sequence_length, *value.shape[1:]
        ).reshape(value.shape[0] * sequence_length, *value.shape[1:])

    def forward(
        self,
        x,
        channel_coordinates,
        channel_region_ids,
        channel_validity,
    ):
        if x.ndim != 4 or x.shape[1] != 20:
            raise ValueError("ISRUC input must have shape [B,20,C,L]")
        batch_size, sequence_length, channels, signal_length = x.shape
        x = x.reshape(batch_size * sequence_length, channels, signal_length)
        features, valid_token_mask = self.encoder(
            x,
            channel_coordinates=self._repeat_metadata(
                channel_coordinates, sequence_length
            ),
            channel_region_ids=self._repeat_metadata(
                channel_region_ids, sequence_length
            ),
            channel_validity=self._repeat_metadata(
                channel_validity, sequence_length
            ),
            return_valid_token_mask=True,
        )
        features = features * valid_token_mask.unsqueeze(-1)
        epoch_input = self.head["representation_flatten"](features)
        epoch_features = self.head["epoch_projection"](epoch_input)
        epoch_features = epoch_features.reshape(
            batch_size, sequence_length, 512
        )
        sequence_features = self.head["sequence_encoder"](epoch_features)
        # Expose the exact per-epoch representation seen by the classifier so
        # the shared downstream diagnostics can compute CKA and subject/class
        # probes without changing the ISRUC sequence protocol.
        self.last_summary_embedding = sequence_features
        return self.head["classifier"](sequence_features)
