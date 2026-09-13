"""Raw-patch extraction, deterministic gathering, and diagnostics."""

import torch
import torch.nn.functional as F


def raw_patch_grid(signals, patch_samples):
    """Return non-overlapping raw targets as ``[B,C,T,P]``."""
    if signals.ndim != 3:
        raise ValueError("signals must have shape [B,C,L]")
    if signals.shape[-1] % int(patch_samples) != 0:
        raise ValueError("signal length must be divisible by patch_samples")
    return signals.unfold(-1, int(patch_samples), int(patch_samples))


def gather_target_blocks(target_grid, target_blocks, return_token_valid=False):
    """Gather blocks in channel-major order, padding to the largest block."""
    batch, channels, patches, dim = target_grid.shape
    flat_blocks = target_blocks.flatten(2)
    counts = flat_blocks.sum(dim=2)
    count = int(counts.max())
    token_valid = (
        torch.arange(count, device=target_grid.device)[None, None]
        < counts[:, :, None]
    )
    indices = torch.arange(
        channels * patches, device=target_grid.device
    ).view(1, 1, -1).expand_as(flat_blocks)
    indices = indices.masked_fill(~flat_blocks, channels * patches)
    indices = indices.sort(dim=2).values[:, :, :count]
    indices = indices.masked_fill(~token_valid, 0)
    grid = target_grid.flatten(1, 2)[:, None].expand(
        -1, target_blocks.shape[1], -1, -1
    )
    gathered = torch.gather(
        grid, 2, indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    ).reshape(batch, target_blocks.shape[1] * count, dim)
    gathered = gathered * token_valid.reshape(
        batch, target_blocks.shape[1] * count, 1
    )
    if return_token_valid:
        return gathered, token_valid
    return gathered


@torch.no_grad()
def representation_diagnostics(representations):
    flattened = representations.reshape(-1, representations.shape[-1]).float()
    variance = flattened.var(dim=0, unbiased=False).mean()
    normalized = F.normalize(flattened, dim=-1)
    count = flattened.shape[0]
    mean_cosine = (
        normalized.sum(dim=0).square().sum() - count
    ) / (count * (count - 1))
    return {
        "feature_variance": variance,
        "mean_pairwise_cosine": mean_cosine,
    }


@torch.no_grad()
def reconstruction_diagnostics(prediction, target, sample_rate):
    """Waveform and canonical EEG-band reconstruction diagnostics."""
    prediction = prediction.float()
    target = target.float()
    prediction_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    correlation = F.cosine_similarity(
        prediction_centered, target_centered, dim=-1
    ).mean()
    amplitude_ratio = (
        prediction_centered.square().mean().sqrt()
        / target_centered.square().mean().sqrt().clamp_min(1e-8)
    )
    prediction_spectrum = torch.log1p(
        torch.fft.rfft(prediction, dim=-1).abs()
    )
    target_spectrum = torch.log1p(torch.fft.rfft(target, dim=-1).abs())
    frequencies = torch.fft.rfftfreq(
        prediction.shape[-1], d=1.0 / float(sample_rate)
    ).to(prediction.device)
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    diagnostics = {
        "waveform_correlation": correlation,
        "amplitude_ratio": amplitude_ratio,
    }
    absolute_error = (prediction_spectrum - target_spectrum).abs()
    for name, (low, high) in bands.items():
        selected = (frequencies >= low) & (frequencies < high)
        diagnostics[f"log_rfft_error_{name}"] = absolute_error[
            ..., selected
        ].mean()
    return diagnostics
