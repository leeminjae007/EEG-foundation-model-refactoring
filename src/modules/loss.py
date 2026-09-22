"""Loss는 engine에서 호출한다. Target EEG에는 gradient를 만들지 않는다."""

import torch
from torch.nn import functional as F


def reconstruction_loss(prediction, target, target_valid, beta):
    # [B,N,P]. 유효한 target scalar sample 수로 나눈다.
    # Geometry는 합집합이라 패치당 한 번, I-JEPA는 겹친 패치도 블록마다 포함한다.
    pointwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    sample_mask = target_valid.flatten(1).unsqueeze(-1)
    denominator = (sample_mask.sum() * prediction.shape[-1]).clamp_min(1)
    return (pointwise * sample_mask).sum() / denominator


def downstream_loss(task, logits, labels, config):
    labels = labels.reshape(-1)
    if task == "regression":
        return F.mse_loss(logits.reshape(-1), labels.to(logits.dtype))
    if task == "binary":
        losses = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.to(logits.dtype), reduction="none")
        counts = torch.as_tensor(config["class_counts"], dtype=torch.float64, device=logits.device)
        weights = counts.reciprocal()
        weights = (weights / weights.mean()).to(torch.float32).to(logits.dtype)
        return (losses * weights[labels.long()]).mean()
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.long(),
                             reduction="none", label_smoothing=config["label_smoothing"])
    return losses.mean()
