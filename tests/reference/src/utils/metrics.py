"""Downstream metrics and exact distributed prediction collection."""

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


def gather_dataset_predictions(logits, labels):
    """Gather variable local shards without padding samples or duplicates."""
    if logits.ndim not in (1, 2):
        raise ValueError("logits must have shape [N] or [N,K]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [N] matching logits")
    if not dist.is_available() or not dist.is_initialized():
        return logits.detach().cpu(), labels.detach().cpu()

    world_size = dist.get_world_size()
    local_size = torch.tensor(
        [logits.shape[0]], dtype=torch.long, device=logits.device
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    maximum = max(sizes)

    logit_shape = (maximum,) + tuple(logits.shape[1:])
    padded_logits = torch.zeros(
        logit_shape, dtype=logits.dtype, device=logits.device
    )
    padded_labels = torch.zeros(
        maximum, dtype=labels.dtype, device=labels.device
    )
    padded_logits[: logits.shape[0]] = logits
    padded_labels[: labels.shape[0]] = labels

    gathered_logits = [
        torch.empty_like(padded_logits) for _ in range(world_size)
    ]
    gathered_labels = [
        torch.empty_like(padded_labels) for _ in range(world_size)
    ]
    dist.all_gather(gathered_logits, padded_logits)
    dist.all_gather(gathered_labels, padded_labels)
    return (
        torch.cat(
            [value[:size] for value, size in zip(gathered_logits, sizes)]
        ).cpu(),
        torch.cat(
            [value[:size] for value, size in zip(gathered_labels, sizes)]
        ).cpu(),
    )


def chb_metrics(logits, labels):
    """CHB metrics using the same definitions as CBraMod's evaluator."""
    logits, labels = gather_dataset_predictions(logits, labels)
    probabilities = logits.float().sigmoid().numpy()
    truth = labels.long().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    precision, recall, _ = precision_recall_curve(
        truth, probabilities, pos_label=1
    )
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, predictions)
        ),
        "auprc": float(auc(recall, precision)),
        "auroc": float(roc_auc_score(truth, probabilities)),
    }


def seedv_metrics(logits, labels):
    """SEED-V Balanced Accuracy, Cohen's Kappa, and weighted F1."""
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("SEED-V logits must have shape [N,5]")
    logits, labels = gather_dataset_predictions(logits, labels)
    truth = labels.long().numpy()
    predictions = logits.argmax(dim=-1).numpy()
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, predictions)
        ),
        "kappa": float(cohen_kappa_score(truth, predictions)),
        "weighted_f1": float(
            f1_score(truth, predictions, average="weighted")
        ),
    }


def multiclass_metrics(logits, labels):
    logits, labels = gather_dataset_predictions(logits, labels)
    truth = labels.long().numpy()
    predictions = logits.argmax(dim=-1).numpy()
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, predictions)
        ),
        "kappa": float(cohen_kappa_score(truth, predictions)),
        "weighted_f1": float(
            f1_score(truth, predictions, average="weighted")
        ),
    }


def regression_metrics(predictions, labels):
    predictions, labels = gather_dataset_predictions(predictions, labels)
    prediction = predictions.float().numpy()
    truth = labels.float().numpy()
    correlation = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "pearson": correlation,
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(mean_squared_error(
            truth, prediction, squared=False
        )),
    }


def downstream_metrics(task, logits, labels):
    if task == "binary":
        return chb_metrics(logits, labels)
    if task == "multiclass":
        return multiclass_metrics(logits, labels)
    return regression_metrics(logits, labels)
