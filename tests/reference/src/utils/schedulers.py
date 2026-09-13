"""Learning-rate schedules retained by downstream fine-tuning."""

import math


class GroupCosineScheduler:
    """Cosine schedule that preserves optimizer-group base learning rates."""

    def __init__(self, optimizer, total_steps, min_learning_rate, warmup_steps=0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.min_learning_rate = float(min_learning_rate)
        self.warmup_steps = int(warmup_steps)
        if self.total_steps <= 0 or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        self.base_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        self.step_index = 0

    def step(self):
        progress = (self.step_index - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        learning_rates = [
            self.min_learning_rate
            + (base - self.min_learning_rate) * cosine
            for base in self.base_learning_rates
        ]
        if self.step_index < self.warmup_steps:
            learning_rates = [
                base * (self.step_index + 1) / self.warmup_steps
                for base in self.base_learning_rates
            ]
        for group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate
        self.step_index += 1
        return learning_rates

    def state_dict(self):
        return {"step_index": self.step_index}

    def load_state_dict(self, state):
        self.step_index = int(state["step_index"])
