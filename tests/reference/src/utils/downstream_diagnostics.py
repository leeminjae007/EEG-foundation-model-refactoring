"""Diagnostics that identify transfer, calibration, and shortcut failures."""

from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn.functional as F
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


def representation_geometry(embeddings):
    """Effective rank and pairwise cosine for one embedding batch."""
    embeddings = embeddings.float().reshape(embeddings.shape[0], -1)
    if embeddings.shape[0] < 2:
        zero = embeddings.new_zeros(())
        return {'effective_rank': zero, 'mean_pairwise_cosine': zero}
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum())
    normalized = F.normalize(embeddings, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    count = embeddings.shape[0]
    mean_cosine = (
        similarities.sum() - similarities.diagonal().sum()
    ) / (count * (count - 1))
    return {
        'effective_rank': effective_rank,
        'mean_pairwise_cosine': mean_cosine,
    }


def linear_cka(reference, current):
    """Linear centered-kernel alignment for one fixed probe batch."""
    reference = reference.float().reshape(reference.shape[0], -1)
    current = current.float().reshape(current.shape[0], -1)
    if reference.shape != current.shape or reference.shape[0] < 2:
        raise ValueError('CKA inputs must share [N,D] with N >= 2')
    reference = reference - reference.mean(dim=0, keepdim=True)
    current = current - current.mean(dim=0, keepdim=True)
    cross = reference.transpose(0, 1) @ current
    numerator = cross.square().sum()
    reference_norm = (
        reference.transpose(0, 1) @ reference).square().sum().sqrt()
    current_norm = (
        current.transpose(0, 1) @ current).square().sum().sqrt()
    return numerator / (reference_norm * current_norm).clamp_min(1e-12)


def nearest_centroid_probe(embeddings, categories, seed=42):
    """Deterministic within-split probe for categorical information.

    Each category is divided into a centroid-fitting half and a held-out half.
    Reporting balanced accuracy and its chance-normalized value makes subject
    identity and task-label probes comparable even when their class counts
    differ substantially. Categories with fewer than two examples are omitted.
    """
    embeddings = embeddings.detach().float().cpu().reshape(
        embeddings.shape[0], -1)
    categories = np.asarray([str(value) for value in categories])
    if embeddings.shape[0] != categories.size:
        raise ValueError('probe embeddings and categories must align')
    generator = np.random.default_rng(int(seed))
    train_indices = []
    validation_indices = []
    retained = []
    for category in sorted(np.unique(categories)):
        indices = np.flatnonzero(categories == category)
        if indices.size < 2:
            continue
        indices = generator.permutation(indices)
        split = max(1, indices.size // 2)
        if split == indices.size:
            split -= 1
        train_indices.extend(indices[:split].tolist())
        validation_indices.extend(indices[split:].tolist())
        retained.append(category)
    if len(retained) < 2 or not validation_indices:
        return {
            'num_classes': len(retained),
            'balanced_accuracy': float('nan'),
            'chance': float('nan'),
            'chance_normalized_accuracy': float('nan'),
        }
    train = embeddings[train_indices]
    validation = embeddings[validation_indices]
    mean = train.mean(dim=0, keepdim=True)
    # Use one global scale. Per-feature standardization can amplify an almost
    # constant nuisance dimension enough to dominate Euclidean distance.
    scale = train.std(unbiased=False).clamp_min(1e-6)
    train = (train - mean) / scale
    validation = (validation - mean) / scale
    train_categories = categories[train_indices]
    validation_categories = categories[validation_indices]
    centroids = torch.stack([
        train[train_categories == category].mean(dim=0)
        for category in retained
    ])
    distances = torch.cdist(validation, centroids)
    predictions = np.asarray(retained)[distances.argmin(dim=1).numpy()]
    score = float(balanced_accuracy_score(
        validation_categories, predictions))
    chance = 1.0 / len(retained)
    return {
        'num_classes': len(retained),
        'balanced_accuracy': score,
        'chance': chance,
        'chance_normalized_accuracy': (
            (score - chance) / max(1.0 - chance, 1e-12)
        ),
    }


def classification_loss_diagnostics(
    task,
    logits,
    labels,
    label_smoothing=0.0,
):
    """Return class-conditional loss from the current microbatch."""
    logits = logits.float()
    labels = labels.reshape(-1)
    if task == 'binary':
        losses = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.float(), reduction='none'
        )
        output = {}
        for value, name in ((0, 'negative'), (1, 'positive')):
            selected = losses[labels.long() == value]
            if selected.numel():
                output[f'loss_{name}'] = selected.mean()
        return output
    if task == 'multiclass':
        flattened = logits.reshape(-1, logits.shape[-1])
        losses = F.cross_entropy(
            flattened,
            labels.long(),
            reduction='none',
            label_smoothing=float(label_smoothing),
        )
        return {
            f'loss_class_{index}': losses[labels.long() == index].mean()
            for index in range(flattened.shape[-1])
            if (labels.long() == index).any()
        }
    return {
        'loss_regression_mse': F.mse_loss(
            logits.reshape(-1), labels.float()
        )
    }


def binary_calibration(logits, labels, num_bins=10):
    probabilities = logits.float().sigmoid().reshape(-1)
    truth = labels.float().reshape(-1)
    brier = (probabilities - truth).square().mean()
    edges = torch.linspace(
        0.0, 1.0, int(num_bins) + 1, device=probabilities.device
    )
    expected_error = probabilities.new_zeros(())
    output = {'brier_score': float(brier)}
    for index in range(int(num_bins)):
        selected = (
            (probabilities >= edges[index])
            & (
                probabilities < edges[index + 1]
                if index + 1 < int(num_bins)
                else probabilities <= edges[index + 1]
            )
        )
        count = int(selected.sum())
        output[f'reliability_bin_{index}/count'] = count
        if not count:
            output[f'reliability_bin_{index}/confidence'] = float('nan')
            output[f'reliability_bin_{index}/accuracy'] = float('nan')
            continue
        confidence = probabilities[selected].mean()
        accuracy = truth[selected].mean()
        expected_error = expected_error + (
            selected.float().mean() * (confidence - accuracy).abs()
        )
        output[f'reliability_bin_{index}/confidence'] = float(confidence)
        output[f'reliability_bin_{index}/accuracy'] = float(accuracy)
    output['ece'] = float(expected_error)
    return output


def regression_calibration(predictions, labels):
    predictions = predictions.float().reshape(-1)
    labels = labels.float().reshape(-1)
    centered = predictions - predictions.mean()
    denominator = centered.square().sum()
    slope = (
        (centered * (labels - labels.mean())).sum()
        / denominator.clamp_min(1e-12)
    )
    intercept = labels.mean() - slope * predictions.mean()
    return {
        'calibration_slope': float(slope),
        'calibration_intercept': float(intercept),
    }


def confusion_diagnostics(logits, labels):
    labels = labels.long().reshape(-1)
    if logits.ndim == 1:
        predictions = logits.float().sigmoid().ge(0.5).long()
        classes = 2
    else:
        predictions = logits.argmax(dim=-1).long().reshape(-1)
        classes = logits.shape[-1]
    encoded = labels * classes + predictions
    matrix = torch.bincount(
        encoded.cpu(), minlength=classes * classes
    ).reshape(classes, classes)
    output = {
        f'confusion/{truth}/{prediction}': int(matrix[truth, prediction])
        for truth in range(classes)
        for prediction in range(classes)
    }
    for index in range(classes):
        support = int(matrix[index].sum())
        output[f'class_recall_defined/{index}'] = int(support > 0)
        if support:
            output[f'class_recall/{index}'] = float(matrix[index, index] / support)
    return output


def _metrics_for_subject(task, logits, labels):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if task == 'binary':
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictions = (probabilities >= 0.5).astype(np.int64)
        output = {
            'accuracy': float((predictions == labels).mean()),
            'sample_count': int(labels.size),
            'positive_count': int((labels == 1).sum()),
            'negative_count': int((labels == 0).sum()),
            'true_positive': int(((predictions == 1) & (labels == 1)).sum()),
            'true_negative': int(((predictions == 0) & (labels == 0)).sum()),
            'false_positive': int(((predictions == 1) & (labels == 0)).sum()),
            'false_negative': int(((predictions == 0) & (labels == 1)).sum()),
            'ranking_metrics_eligible': int(np.unique(labels).size == 2),
        }
        for index in range(2):
            support = int((labels == index).sum())
            if support:
                output[f'recall_class_{index}'] = float((predictions[labels == index] == index).mean())
        if np.unique(labels).size == 2:
            output['balanced_accuracy'] = float(
                balanced_accuracy_score(labels, predictions)
            )
            precision, recall, _ = precision_recall_curve(
                labels, probabilities, pos_label=1
            )
            output['auroc'] = float(roc_auc_score(labels, probabilities))
            output['auprc'] = float(auc(recall, precision))
        return output
    if task == 'multiclass':
        predictions = logits.argmax(axis=-1)
        output = {
            'balanced_accuracy': float(
                balanced_accuracy_score(labels, predictions)
            ),
            'kappa': float(cohen_kappa_score(labels, predictions)),
            'weighted_f1': float(
                f1_score(labels, predictions, average='weighted')
            ),
        }
        for index in range(logits.shape[-1]):
            support = int((labels == index).sum())
            if support:
                output[f'recall_class_{index}'] = float((predictions[labels == index] == index).mean())
        return output
    output = {
        'rmse': float(mean_squared_error(labels, logits, squared=False))
    }
    if labels.size >= 2 and labels.std() > 0 and logits.std() > 0:
        output['pearson'] = float(np.corrcoef(labels, logits)[0, 1])
    if labels.size >= 2 and labels.var() > 0:
        output['r2'] = float(r2_score(labels, logits))
    return output


def subject_metric_diagnostics(task, logits, labels, subject_ids):
    logits = logits.detach().float().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    groups = defaultdict(list)
    missing_subject_count = 0
    for index, subject_id in enumerate(subject_ids):
        if str(subject_id).lower() in {'unavailable', 'unknown', 'none', ''}:
            missing_subject_count += 1
            continue
        groups[str(subject_id)].append(index)
    per_subject = {}
    for subject_id, indices in sorted(groups.items()):
        per_subject[subject_id] = _metrics_for_subject(
            task, logits[indices], labels[indices]
        )
    metric_names = sorted({
        metric for metrics in per_subject.values() for metric in metrics
    })
    macro = {}
    for metric in metric_names:
        values = [
            metrics[metric]
            for metrics in per_subject.values()
            if metric in metrics and np.isfinite(metrics[metric])
        ]
        if values:
            macro[metric] = float(np.mean(values))
    return {
        'macro': macro, 'per_subject': per_subject,
        'subject_id_available_sample_count': len(subject_ids) - missing_subject_count,
        'subject_id_unavailable_sample_count': missing_subject_count,
        'subject_count': len(per_subject),
        'ranking_metrics_policy': 'AUROC and trapezoidal AUPRC reported only for subjects containing both classes; counts remain available for one-class subjects.',
    }
