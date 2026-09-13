"""Optimizer and learning-rate schedule construction for pretraining."""

import math

import torch
from src.utils.schedulers import GroupCosineScheduler


class WarmupCosineScheduler:
    """Historical per-update linear-warmup cosine schedule."""

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps,
        start_learning_rate,
        base_learning_rate,
        min_learning_rate,
        final_weight_decay,
    ):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.start_learning_rate = float(start_learning_rate)
        self.base_learning_rate = float(base_learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.final_weight_decay = float(final_weight_decay)
        self.start_weight_decays = [
            float(group["weight_decay"]) for group in optimizer.param_groups
        ]
        self.step_index = 0

    def step(self):
        self.step_index += 1
        if self.step_index <= self.warmup_steps:
            progress = self.step_index / max(1, self.warmup_steps)
            learning_rate = self.start_learning_rate + progress * (
                self.base_learning_rate - self.start_learning_rate
            )
        else:
            progress = (
                (self.step_index - self.warmup_steps)
                / max(1, self.total_steps - self.warmup_steps)
            )
            learning_rate = self.min_learning_rate + 0.5 * (
                self.base_learning_rate - self.min_learning_rate
            ) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        decay_progress = min(self.step_index / max(1, self.total_steps), 1.0)
        for group, start_weight_decay in zip(
            self.optimizer.param_groups, self.start_weight_decays
        ):
            if start_weight_decay:
                group["weight_decay"] = self.final_weight_decay + 0.5 * (
                    start_weight_decay - self.final_weight_decay
                ) * (1.0 + math.cos(math.pi * decay_progress))
        return learning_rate

    def state_dict(self):
        return {"step_index": self.step_index}

    def load_state_dict(self, state):
        self.step_index = int(state["step_index"])


def build_pretrain_optimizer(context_encoder, decoder, updates_per_epoch, config):
    optimization = config["optimization"]
    named_parameters = [
        (name, parameter)
        for module_name, module in (
            ("context_encoder", context_encoder),
            ("decoder", decoder),
        )
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [
        parameter
        for _, parameter in named_parameters
    ]
    if not parameters:
        raise ValueError("No trainable pretraining parameters")

    historical = (
        config["experiment"].get("profile") == "historical_d192_reproduction"
    )
    if historical:
        decay = [
            parameter for name, parameter in named_parameters
            if parameter.ndim != 1 and not name.endswith(".bias")
        ]
        no_decay = [
            parameter for name, parameter in named_parameters
            if parameter.ndim == 1 or name.endswith(".bias")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=optimization["base_learning_rate"],
            betas=tuple(optimization["adam_betas"]),
            eps=optimization["adam_epsilon"],
            weight_decay=optimization["weight_decay"],
        )
        total_steps = optimization["epochs"] * updates_per_epoch
        warmup_steps = round(total_steps * optimization["warmup_ratio"])
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_learning_rate=optimization["start_learning_rate"],
            base_learning_rate=optimization["base_learning_rate"],
            min_learning_rate=optimization["min_learning_rate"],
            final_weight_decay=optimization["final_weight_decay"],
        )
        return optimizer, scheduler

    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimization["base_learning_rate"],
        betas=tuple(optimization["adam_betas"]),
        eps=optimization["adam_epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    total_steps = optimization["epochs"] * updates_per_epoch
    if optimization.get('warmup_epochs', 0) > 0:
        scheduler = GroupCosineScheduler(
            optimizer, total_steps=total_steps,
            min_learning_rate=optimization['min_learning_rate'],
            warmup_steps=round(optimization['warmup_epochs'] * updates_per_epoch),
        )
        return optimizer, scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=optimization["min_learning_rate"],
    )
    return optimizer, scheduler
