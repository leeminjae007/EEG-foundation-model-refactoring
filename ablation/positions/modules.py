"""Tiny baselines plus the original CBraMod ACPE and REVE 4D PE."""

import mne
import numpy as np
import torch
from torch import nn

from ablation.montage import canonical
from ablation.sources import upstream, reve_position_source
from src.modules.normalization import RMSNorm
from src.modules.position_embedding import sinusoidal_positions


def dimensions(config, location):
    position = config["position"]
    return position[location + "_spatial_dim"], position[location + "_temporal_dim"]


class NoPosition(nn.Module):
    def __init__(self, config, location):
        super().__init__()
        self.dim = sum(dimensions(config, location))

    def forward(self, coordinates, batch, patches, dtype):
        return torch.zeros(batch, coordinates.shape[-2], patches, self.dim,
                           device=coordinates.device, dtype=dtype)


class ChannelID(nn.Module):
    """Named-channel lookup + the baseline's unchanged temporal SinCos/transform.

    This is an elementary control, not attributed to a separate paper. IDs are
    fixed across datasets; they never mean 'column number in this batch'.
    """
    def __init__(self, config, location):
        super().__init__()
        self.spatial_dim, self.temporal_dim = dimensions(config, location)
        self.max_period = config["position"]["temporal_max_period"]
        self.vocabulary = tuple(config["ablation"]["channel_vocabulary"])
        self.embedding = nn.Embedding(len(self.vocabulary), self.spatial_dim)
        self.activation = nn.GELU()
        self.norm = RMSNorm(self.spatial_dim + self.temporal_dim, config["position"]["rms_epsilon"])
        self.register_buffer("indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.set_channels(config["data"]["channel_names"])

    def set_channels(self, names):
        lookup = {name: index for index, name in enumerate(self.vocabulary)}
        missing = [name for name in names if canonical(name) not in lookup]
        if missing:
            raise ValueError("Channel ID vocabulary is missing " + str(missing))
        self.indices = torch.tensor([lookup[canonical(name)] for name in names],
                                    dtype=torch.long, device=self.indices.device)

    def forward(self, coordinates, batch, patches, dtype):
        if coordinates.shape[-2] != len(self.indices):
            raise ValueError("Call set_channels with this dataset's channel names before forwarding")
        spatial = self.embedding(self.indices).to(dtype)[None, :, None]
        temporal = sinusoidal_positions(patches, self.temporal_dim, self.max_period,
                                        coordinates.device, dtype)[None, None]
        spatial = spatial.expand(batch, -1, patches, -1)
        temporal = temporal.expand(batch, len(self.indices), -1, -1)
        return self.norm(self.activation(torch.cat((spatial, temporal), dim=-1)))


class ACPE(nn.Module):
    uses_tokens = True

    def __init__(self, config, location):
        super().__init__()
        dim = sum(dimensions(config, location))
        module = upstream("cbramod", "models/cbramod.py")
        # Extract the original depthwise Conv2d(19,7); no convolution is rewritten.
        self.positional_encoding = module.PatchEmbedding(200, 200, dim, 30).positional_encoding

    def forward(self, tokens, coordinates, visible):
        # ACPE mixes neighbours. Remove hidden content BEFORE the convolution.
        tokens = tokens.masked_fill(~visible.unsqueeze(-1), 0)
        return self.positional_encoding(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class REVE4D(nn.Module):
    def __init__(self, config, location):
        super().__init__()
        dim = sum(dimensions(config, location))
        settings = config["ablation"]
        freqs = settings.get("reve_freqs", 4)
        # The upstream truncation has a :-0 edge case. Reject it, don't patch it.
        if dim % 2 or (dim != 512 and freqs ** 4 <= dim // 2) or (dim == 512 and freqs != 4):
            raise ValueError("REVE source requires even width < 2*freqs**4, or width 512 with freqs=4")
        module = reve_position_source()
        self.fourier4d = module.FourierEmb4D(dim, freqs=freqs)
        self.mlp4d = module.mlp_pos_embedding(dim)
        self.ln = nn.LayerNorm(dim)
        self.noise_ratio = settings.get("reve_noise_ratio", 0.0)
        positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        # src coordinates divide every position by this radius; REVE uses metres.
        self.coordinate_scale = max(float(np.linalg.norm(value)) for value in positions.values())

    def forward(self, coordinates, batch, patches, dtype):
        if coordinates.ndim == 2:
            coordinates = coordinates[None].expand(batch, -1, -1)
        positions = coordinates.float() * self.coordinate_scale
        if self.training and self.noise_ratio:
            noise = np.random.normal(loc=0, scale=self.noise_ratio, size=(positions.shape[1], 3))
            positions = positions + torch.from_numpy(noise).to(positions)
        positions = self.fourier4d.add_time_patch(positions, patches)
        embedding = self.ln(self.fourier4d(positions) + self.mlp4d(positions))
        return embedding.reshape(batch, coordinates.shape[-2], patches, -1).to(dtype)
