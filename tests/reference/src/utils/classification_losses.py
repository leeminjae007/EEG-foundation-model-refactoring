"""Classification-loss variants for controlled downstream comparisons."""

import torch
import torch.nn.functional as F


SUPPORTED_CLASSIFICATION_LOSSES = {
    "ce", "weighted_ce", "focal_ce", "class_balanced_ce",
    "balanced_softmax",
}


def normalized_class_weights(class_counts, method, beta=0.9999, device=None):
    """Return positive class weights normalized to mean one."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64, device=device)
    if counts.ndim != 1 or counts.numel() < 2 or torch.any(counts <= 0):
        raise ValueError("class_counts must contain at least two positive counts")
    if method == "weighted_ce":
        weights = counts.reciprocal()
    elif method == "class_balanced_ce":
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("class_balance_beta must be in [0, 1)")
        beta_tensor = counts.new_tensor(float(beta))
        weights = (1.0 - beta_tensor) / (1.0 - beta_tensor.pow(counts))
    else:
        raise ValueError(f"unsupported weighting method: {method}")
    return (weights / weights.mean()).to(torch.float32)


def classification_loss(task, logits, labels, optimization):
    """Compute a configured classification objective."""
    method = optimization.get("classification_loss", "ce")
    if method not in SUPPORTED_CLASSIFICATION_LOSSES:
        raise ValueError(f"unsupported classification loss: {method}")
    if task not in {"binary", "multiclass"}:
        raise ValueError(f"classification loss requires a classification task, got {task}")

    labels = labels.reshape(-1)
    class_weights = None
    if method in {"weighted_ce", "class_balanced_ce"}:
        class_weights = normalized_class_weights(
            optimization["class_counts"],
            method,
            beta=optimization.get("class_balance_beta", 0.9999),
            device=logits.device,
        ).to(dtype=logits.dtype)

    if task == "binary":
        flattened = logits.reshape(-1)
        targets = labels.to(dtype=flattened.dtype)
        if method == "balanced_softmax":
            counts = torch.as_tensor(
                optimization["class_counts"],
                dtype=flattened.dtype,
                device=flattened.device,
            )
            if counts.numel() != 2:
                raise ValueError(
                    "binary balanced_softmax requires two class counts"
                )
            flattened = flattened + (counts[1].log() - counts[0].log())
        losses = F.binary_cross_entropy_with_logits(
            flattened, targets, reduction="none"
        )
        if method == "focal_ce":
            gamma = float(optimization.get("focal_gamma", 2.0))
            if gamma < 0:
                raise ValueError("focal_gamma must be nonnegative")
            probability = torch.sigmoid(flattened)
            p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
            losses = losses * (1.0 - p_t).pow(gamma)
        elif class_weights is not None:
            losses = losses * class_weights[labels.long()]
        return losses.mean()

    flattened = logits.reshape(-1, logits.shape[-1])
    targets = labels.long()
    smoothing = float(optimization.get("label_smoothing", 0.0))
    if method == "balanced_softmax":
        counts = torch.as_tensor(
            optimization["class_counts"],
            dtype=flattened.dtype,
            device=flattened.device,
        )
        if counts.numel() != flattened.shape[-1]:
            raise ValueError(
                "balanced_softmax class_counts must match the logit dimension"
            )
        return F.cross_entropy(
            flattened + counts.log(),
            targets,
            label_smoothing=smoothing,
        )
    if class_weights is not None:
        return F.cross_entropy(
            flattened,
            targets,
            weight=class_weights,
            label_smoothing=smoothing,
        )
    losses = F.cross_entropy(
        flattened,
        targets,
        reduction="none",
        label_smoothing=smoothing,
    )
    if method == "focal_ce":
        gamma = float(optimization.get("focal_gamma", 2.0))
        if gamma < 0:
            raise ValueError("focal_gamma must be nonnegative")
        p_t = flattened.softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
        losses = losses * (1.0 - p_t).pow(gamma)
    return losses.mean()
