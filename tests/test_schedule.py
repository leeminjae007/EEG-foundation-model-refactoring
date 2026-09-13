"""실제 두 LR 갱신 순서와 마지막 accumulation group을 확인한다."""

import math
import torch
from src.training.scheduler import GroupCosineScheduler


def test_pretrain_cosine_steps_after_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)
    used = []
    for _ in range(120):
        used.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    for step, value in enumerate(used):
        expected = 1e-5 + (5e-4 - 1e-5) * (1 + math.cos(math.pi * step / 120)) / 2
        assert abs(value - expected) < 1e-15


def test_downstream_five_epoch_warmup_and_tail_group():
    # 17 microbatch, accumulation 8 → 마지막 1개도 별도의 update가 된다.
    updates = math.ceil(17 / 8)
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(3)]
    optimizer = torch.optim.AdamW([
        {"params": [parameter], "lr": rate}
        for parameter, rate in zip(parameters, [1e-5, 5e-6, 1e-4])])
    scheduler = GroupCosineScheduler(optimizer, 20 * updates, 1e-6, 5 * updates)
    rates = []
    for _ in range(20 * updates):
        rates.append(scheduler.step())
        optimizer.step()
    assert rates[0] == [1e-5 / 15, 5e-6 / 15, 1e-4 / 15]
    assert rates[14] == [1e-5, 5e-6, 1e-4]
    assert rates[15] == [1e-5, 5e-6, 1e-4]
    assert rates[-1] == [1e-6, 1e-6, 1e-6]
