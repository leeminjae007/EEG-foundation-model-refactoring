"""Validation-only common-epoch selection, independent of model modules."""

import math


def select_common_epoch(histories, expected_epochs):
    """Equal subject/fold weights, earliest epoch on an exact tie."""
    if not histories:
        raise ValueError('No completed folds')
    curves = []
    for history in histories:
        if [row['epoch'] for row in history] != list(range(1, expected_epochs + 1)):
            raise ValueError('Incomplete or inconsistent fold epoch history')
        curve = [float(row['validation']['balanced_accuracy']) for row in history]
        if not all(math.isfinite(value) for value in curve):
            raise ValueError('Non-finite fold BAcc')
        curves.append(curve)
    means = [sum(values) / len(values) for values in zip(*curves)]
    selected = max(range(expected_epochs), key=lambda i: means[i])
    values = [curve[selected] for curve in curves]
    return {
        'selected_epoch': selected + 1,
        'selection_metric': 'mean_subject_validation_balanced_accuracy',
        'mean_bacc_curve': means,
        'selected_fold_bacc': values,
        'selected_mean_bacc': means[selected],
        'selected_fold_sd': math.sqrt(sum((v - means[selected]) ** 2 for v in values) / len(values)),
        'note': 'Fold SD is not seed SD. Selection score is not an unbiased test estimate.',
    }
