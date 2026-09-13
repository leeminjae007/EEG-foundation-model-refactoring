"""Two-branch time/frequency EEG patch encoder."""

import torch
import torch.nn as nn

from src.models.feature_fusion import TwoBranchFusion


class CBraModTimeFrequencyPatchEncoder(nn.Module):
    """CBraMod time/frequency patch stem followed by this project's SHPE.

    The two feature branches mirror the official CBraMod ``PatchEmbedding``.
    Its depthwise positional convolution is intentionally not part of this
    tokenizer because positional encoding is supplied by the following SHPE.
    """

    def __init__(self, patch_samples=200, embed_dim=200, dropout=0.1):
        super().__init__()
        if int(patch_samples) != 200 or int(embed_dim) != 200:
            raise ValueError("the official CBraMod tokenizer requires P=D=200")
        self.patch_samples = int(patch_samples)
        self.embed_dim = int(embed_dim)
        self.time_feature_dim = self.embed_dim
        self.frequency_feature_dim = self.embed_dim
        # Match the official CBraMod pretraining path: hidden raw patches are
        # replaced before either the temporal stem or the spectral transform.
        # The fixed zero vector is deliberately present in the state dict so
        # downstream construction has the same strict-loading contract.
        self.mask_encoding = nn.Parameter(
            torch.zeros(self.patch_samples), requires_grad=False
        )
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, self.embed_dim),
            nn.Dropout(float(dropout)),
        )

    def _patchify(self, x):
        if x.ndim != 3 or x.shape[-1] % self.patch_samples:
            raise ValueError("expected [B,C,L] with L divisible by patch size")
        batch, channels, signal_length = x.shape
        patch_count = signal_length // self.patch_samples
        return (
            x.reshape(batch, channels, patch_count, self.patch_samples),
            batch,
            channels,
            patch_count,
        )

    def _branches(self, patches):
        batch, channels, patch_count, patch_size = patches.shape
        flattened = patches.reshape(batch, 1, channels * patch_count, patch_size)
        temporal = self.proj_in(flattened)
        temporal = temporal.permute(0, 2, 1, 3).contiguous().reshape(
            batch, channels, patch_count, self.embed_dim
        )
        spectral = torch.fft.rfft(
            patches.reshape(batch * channels * patch_count, patch_size),
            dim=-1,
            norm="forward",
        )
        spectral = spectral.abs().reshape(batch, channels, patch_count, 101)
        frequency = self.spectral_proj(spectral)
        return temporal, frequency

    def _replace_hidden_patches(self, patches, visible_mask):
        if visible_mask is None:
            return patches
        if visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be boolean")
        if visible_mask.shape != patches.shape[:-1]:
            raise ValueError(
                "visible_mask must have shape [B,C,T] matching the patch grid"
            )
        mask_encoding = self.mask_encoding.to(
            device=patches.device, dtype=patches.dtype
        ).view(1, 1, 1, self.patch_samples)
        return torch.where(visible_mask.unsqueeze(-1), patches, mask_encoding)

    def temporal_parameters(self):
        return self.proj_in.parameters()

    def frequency_parameters(self):
        return self.spectral_proj.parameters()

    def forward(self, x, visible_mask=None, return_diagnostics=False):
        patches, _, _, _ = self._patchify(x)
        patches = self._replace_hidden_patches(patches, visible_mask)
        temporal, frequency = self._branches(patches)
        tokens = temporal + frequency
        if not return_diagnostics:
            return tokens
        return tokens, {
            "temporal_feature_norm": temporal.float().norm(
                dim=-1
            ).mean().detach(),
            "frequency_feature_norm": frequency.float().norm(
                dim=-1
            ).mean().detach(),
        }


class SharedTimeFrequencyPatchEncoder(nn.Module):
    """Convert ``[B,C,L]`` into non-overlapping ``[B,C,T,D]`` tokens."""

    def __init__(
        self,
        patch_samples,
        embed_dim,
        time_conv_channels,
        time_kernel_sizes,
        time_strides,
        time_group_norm_groups,
        frequency_bins,
        frequency_hidden_dim,
        fusion,
        dropout,
        time_feature_dim,
        frequency_feature_dim,
        frequency_input_transform="log1p_magnitude",
        frequency_projection="mlp",
    ):
        super().__init__()
        if frequency_bins != patch_samples // 2 + 1:
            raise ValueError("frequency_bins must match rFFT output")

        self.patch_samples = int(patch_samples)
        self.embed_dim = int(embed_dim)
        self.time_feature_dim = int(time_feature_dim)
        self.frequency_feature_dim = int(frequency_feature_dim)
        self.frequency_input_transform = str(frequency_input_transform)
        self.frequency_projection = str(frequency_projection)
        if self.frequency_input_transform not in {
            "log1p_magnitude", "magnitude"
        }:
            raise ValueError(
                "frequency_input_transform must be log1p_magnitude or magnitude"
            )
        if self.frequency_projection not in {"mlp", "linear_dropout"}:
            raise ValueError(
                "frequency_projection must be mlp or linear_dropout"
            )
        self.fusion = TwoBranchFusion(
            self.time_feature_dim,
            self.frequency_feature_dim,
            self.embed_dim,
            fusion,
        )

        layers = []
        input_channels = 1
        for output_channels, kernel_size, stride in zip(
            time_conv_channels, time_kernel_sizes, time_strides
        ):
            layers.extend((
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.GroupNorm(time_group_norm_groups, output_channels),
                nn.GELU(),
            ))
            input_channels = output_channels
        self.time_convolutions = nn.Sequential(*layers)

        with torch.no_grad():
            output_size = self.time_convolutions(
                torch.zeros(1, 1, self.patch_samples)
            ).numel()
        self.time_projection = nn.Linear(output_size, self.time_feature_dim)
        if self.frequency_projection == "mlp":
            # Historical project tokenizer, including every GR5-C checkpoint.
            self.frequency_mlp = nn.Sequential(
                nn.Linear(frequency_bins, frequency_hidden_dim),
                nn.GELU(),
                nn.Linear(frequency_hidden_dim, self.frequency_feature_dim),
            )
        else:
            # CBraMod/CSBrain-style spectral projection. Keep the historical
            # attribute name so callers need no branch-specific API.
            self.frequency_mlp = nn.Sequential(
                nn.Linear(frequency_bins, self.frequency_feature_dim),
                nn.Dropout(dropout),
            )
        self.output_dropout = nn.Dropout(dropout)

    def _patchify(self, x):
        if x.ndim != 3 or x.shape[-1] % self.patch_samples:
            raise ValueError("expected [B,C,L] with L divisible by patch size")
        batch, channels, signal_length = x.shape
        patches = signal_length // self.patch_samples
        return (
            x.reshape(batch * channels * patches, 1, self.patch_samples),
            batch,
            channels,
            patches,
        )

    def _branches(self, patches):
        temporal = self.time_projection(
            self.time_convolutions(patches).flatten(start_dim=1)
        )
        magnitude = torch.fft.rfft(
            patches[:, 0],
            n=self.patch_samples,
            dim=-1,
            norm="forward",
        ).abs()
        if self.frequency_input_transform == "log1p_magnitude":
            frequency_input = torch.log1p(magnitude)
        else:
            frequency_input = magnitude
        frequency = self.frequency_mlp(frequency_input)
        return temporal, frequency

    def temporal_parameters(self):
        yield from self.time_convolutions.parameters()
        yield from self.time_projection.parameters()

    def frequency_parameters(self):
        yield from self.frequency_mlp.parameters()

    def forward(self, x, return_diagnostics=False):
        patches, batch, channels, patch_count = self._patchify(x)
        temporal, frequency = self._branches(patches)
        tokens = self.output_dropout(self.fusion(temporal, frequency)).reshape(
            batch, channels, patch_count, self.embed_dim
        )
        if not return_diagnostics:
            return tokens
        return tokens, {
            "temporal_feature_norm": temporal.float().norm(
                dim=-1
            ).mean().detach(),
            "frequency_feature_norm": frequency.float().norm(
                dim=-1
            ).mean().detach(),
        }
