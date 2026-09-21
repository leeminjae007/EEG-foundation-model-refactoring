"""Validation-only early stopping, opt-in for hyperparameter searches."""

import math


class ValidationEarlyStopper:
    def __init__(self, config, state=None):
        if config.get('monitor') != 'balanced_accuracy':
            raise ValueError('Early stopping must monitor validation balanced_accuracy')
        self.patience = int(config['patience'])
        self.min_epochs = int(config['min_epochs'])
        self.min_delta = float(config.get('min_delta', 0.0))
        if self.patience < 1 or self.min_epochs < 1 or self.min_delta < 0:
            raise ValueError('Invalid early-stopping patience/min_epochs/min_delta')
        state = state or {}
        self.best = float(state.get('best', -math.inf))
        self.stale_epochs = int(state.get('stale_epochs', 0))

    def step(self, epoch, validation_bacc):
        score = float(validation_bacc)
        if not math.isfinite(score):
            raise ValueError('Non-finite validation balanced_accuracy')
        if score > self.best + self.min_delta:
            self.best = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return epoch >= self.min_epochs and self.stale_epochs >= self.patience

    def state(self):
        return dict(best=self.best, stale_epochs=self.stale_epochs)
