"""Fixed-batch calibration for final-stage branch reconstruction losses."""

import math

import torch


def gradient_rms(loss, parameters):
    """RMS over a stable parameter list; missing gradients count as zero."""
    params = tuple(parameters)
    gradients = torch.autograd.grad(loss, params, allow_unused=True,
                                    retain_graph=True)
    squares = loss.new_zeros(())
    count = 0
    for parameter, gradient in zip(params, gradients):
        count += parameter.numel()
        if gradient is not None:
            squares = squares + gradient.detach().float().square().sum()
    if count == 0:
        raise ValueError("calibration parameter list is empty")
    value = torch.sqrt(squares / count)
    if not torch.isfinite(value) or value.item() == 0.0:
        raise RuntimeError("non-finite or zero calibration gradient RMS")
    return value


def calibrate_auxiliary_lambda(main_losses, s2t_losses, t2s_losses, parameters,
                               target_ratio=0.01):
    """Return fixed lambda from 16 (or another fixed count) diagnostic batches.

    The auxiliary denominator is the mean of the two branch RMS values, not
    the RMS of their sum, so destructive cancellation cannot inflate lambda.
    """
    if not 0.0 < float(target_ratio) <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    batches = list(zip(main_losses, s2t_losses, t2s_losses))
    if not batches:
        raise ValueError("calibration needs at least one fixed minibatch")
    rows = []
    for main, s2t, t2s in batches:
        main_rms = gradient_rms(main, parameters)
        s2t_rms = gradient_rms(s2t, parameters)
        t2s_rms = gradient_rms(t2s, parameters)
        rows.append((main_rms, (s2t_rms + t2s_rms) / 2))
    main_median = torch.stack([x[0] for x in rows]).median()
    aux_median = torch.stack([x[1] for x in rows]).median()
    if not torch.isfinite(main_median) or not torch.isfinite(aux_median):
        raise RuntimeError("non-finite calibration statistic")
    value = float(target_ratio) * main_median / aux_median
    if not torch.isfinite(value) or value.item() <= 0.0:
        raise RuntimeError("invalid calibrated auxiliary lambda")
    return value.detach(), {"main_gradient_rms_median": main_median.detach(),
                            "aux_gradient_rms_median": aux_median.detach(),
                            "target_ratio": float(target_ratio)}
