"""Group-specific cosine floors and epoch-based head preparation."""

import math

from src.training.scheduler import GroupCosineScheduler


class PerGroupCosineScheduler(GroupCosineScheduler):
    def __init__(self, optimizer, total_steps, min_learning_rate, minimums=None):
        super().__init__(optimizer, total_steps, min_learning_rate)
        names = [group["name"] for group in optimizer.param_groups]
        if set(minimums) != set(names):
            raise ValueError("Minimum learning rates must specify every optimizer group")
        self.minimums = [float(minimums[name]) for name in names]
        if any(not 0 <= low <= base for low, base in zip(self.minimums, self.base_learning_rates)):
            raise ValueError("Learning-rate floors must be between zero and the base LR")

    def step(self):
        progress = self.step_index / max(1, self.total_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        rates = [low + (base - low) * cosine
                 for base, low in zip(self.base_learning_rates, self.minimums)]
        for group, rate in zip(self.optimizer.param_groups, rates):
            group["lr"] = rate
        self.step_index += 1
        return rates

    def state_dict(self):
        return {"step_index": self.step_index, "minimums": self.minimums,
                "base_learning_rates": self.base_learning_rates,
                "total_steps": self.total_steps}

    def load_state_dict(self, state):
        for key in ("minimums", "base_learning_rates", "total_steps"):
            if state[key] != getattr(self, key):
                raise ValueError("Resume scheduler differs: " + key)
        super().load_state_dict(state)


class FineTunePolicy:
    def __init__(self, config):
        self.head_first_epochs = config.get("head_first_epochs", 0)
        if not isinstance(self.head_first_epochs, int) or self.head_first_epochs < 0:
            raise ValueError("head_first_epochs must be a nonnegative integer")
        self.minimums = config.get("min_learning_rates")
        self._original_grad_flags = None

    def make_scheduler(self, optimizer, total_steps, minimum):
        if self.minimums is None:
            return GroupCosineScheduler(optimizer, total_steps, minimum)
        return PerGroupCosineScheduler(optimizer, total_steps, minimum, self.minimums)

    def on_epoch_start(self, model, epoch):
        if not self.head_first_epochs:
            return
        parameters = list(model.backbone.parameters())
        if self._original_grad_flags is None:
            self._original_grad_flags = [p.requires_grad for p in parameters]
        frozen = epoch < self.head_first_epochs
        for parameter, original in zip(parameters, self._original_grad_flags):
            parameter.requires_grad_(False if frozen else original)
            if frozen:
                parameter.grad = None
        # trainer.train() runs first; disable backbone dropout during head preparation.
        # Epoch is taken from the resumed training loop, never inferred from train() calls.
        if frozen:
            model.backbone.eval()
