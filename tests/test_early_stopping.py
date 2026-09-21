"""Hyperparameter search stopping must depend on validation only."""

import pytest

from src.training.early_stopping import ValidationEarlyStopper


CONFIG = dict(monitor='balanced_accuracy', patience=3, min_epochs=5, min_delta=0.0)


def test_stops_only_after_patience_and_minimum_epochs():
    stopper = ValidationEarlyStopper(CONFIG)
    assert not stopper.step(1, 0.5)
    assert not stopper.step(2, 0.5)
    assert not stopper.step(3, 0.5)
    assert not stopper.step(4, 0.6)
    assert not stopper.step(5, 0.6)
    assert not stopper.step(6, 0.6)
    assert stopper.step(7, 0.6)


def test_resume_restores_patience_counter():
    first = ValidationEarlyStopper(CONFIG)
    first.step(1, 0.5)
    first.step(2, 0.5)
    resumed = ValidationEarlyStopper(CONFIG, first.state())
    assert not resumed.step(3, 0.5)
    assert not resumed.step(4, 0.5)
    assert resumed.step(5, 0.5)


def test_rejects_non_validation_monitor_and_nonfinite_values():
    with pytest.raises(ValueError):
        ValidationEarlyStopper(dict(CONFIG, monitor='test_balanced_accuracy'))
    with pytest.raises(ValueError):
        ValidationEarlyStopper(CONFIG).step(1, float('nan'))
