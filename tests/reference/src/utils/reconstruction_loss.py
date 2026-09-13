"""Stateless masked raw-patch reconstruction objectives."""

import torch
import torch.nn.functional as F


SUPPORTED_WAVEFORM_LOSSES = {"mse", "l1", "smooth_l1"}


def _masked_patch_mean(values, valid_mask):
    """Average one scalar per patch over valid target slots only."""
    valid = valid_mask.to(device=values.device, dtype=values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def reconstruction_losses(
    prediction,
    target,
    target_token_valid,
    *,
    waveform_loss="mse",
    smooth_l1_beta=0.1,
    frequency_loss_weight=0.0,
    phase_loss_weight=0.0,
    spectral_epsilon=1e-8,
):
    """Compute waveform and optional spectral losses on masked EEG patches.

    ``prediction`` and ``target`` have shape ``[B,N_target,S]``. Spectral
    losses are always evaluated in FP32 so BF16 pretraining does not feed an
    unsupported or low-precision dtype to the FFT kernels.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B,N,S]")
    if waveform_loss not in SUPPORTED_WAVEFORM_LOSSES:
        raise ValueError(f"unsupported waveform loss: {waveform_loss}")

    valid = target_token_valid.flatten(1)
    if valid.shape != prediction.shape[:2]:
        raise ValueError("target_token_valid does not match target slots")
    sample_mask = valid.unsqueeze(-1)

    if waveform_loss == "mse":
        pointwise = (prediction - target).square()
    elif waveform_loss == "l1":
        pointwise = (prediction - target).abs()
    else:
        pointwise = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
    waveform = (pointwise * sample_mask).sum() / (
        sample_mask.sum() * prediction.shape[-1]
    ).clamp_min(1)

    zero = waveform.new_zeros(())
    frequency = zero
    phase = zero
    if frequency_loss_weight or phase_loss_weight:
        prediction_fft = torch.fft.rfft(
            prediction.float(), dim=-1, norm="forward"
        )
        target_fft = torch.fft.rfft(
            target.float(), dim=-1, norm="forward"
        )

        if frequency_loss_weight:
            prediction_frequency = torch.log1p(prediction_fft.abs())
            target_frequency = torch.log1p(target_fft.abs())
            frequency_per_patch = (
                prediction_frequency - target_frequency
            ).abs().mean(dim=-1)
            frequency = _masked_patch_mean(frequency_per_patch, valid)

        if phase_loss_weight:
            target_magnitude = target_fft.abs()
            phase_weights = target_magnitude / (
                target_magnitude.sum(dim=-1, keepdim=True)
                + float(spectral_epsilon)
            )
            phase_error = 1.0 - torch.cos(
                torch.angle(prediction_fft) - torch.angle(target_fft)
            )
            phase_per_patch = (phase_weights * phase_error).sum(dim=-1)
            phase = _masked_patch_mean(phase_per_patch, valid)

    weighted_frequency = float(frequency_loss_weight) * frequency
    weighted_phase = float(phase_loss_weight) * phase
    total = waveform + weighted_frequency + weighted_phase
    return {
        "waveform_loss": waveform,
        "frequency_loss": frequency,
        "weighted_frequency_loss": weighted_frequency,
        "phase_loss": phase,
        "weighted_phase_loss": weighted_phase,
        "total_loss": total,
    }
