"""Bounded real-data diagnostics; never change an ordinary training run."""
import time

import torch
import torch.distributed as dist


class TimedSmoke:
    def __init__(self, seconds, device, world):
        self.seconds = seconds
        self.device, self.world = device, world
        self.started = None

    def start(self):
        self.started = time.monotonic()

    def finished(self):
        if not self.seconds:
            return False
        flag = torch.tensor(int(time.monotonic() - self.started >= self.seconds), device=self.device)
        if self.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


class LimitedLoader:
    def __init__(self, loader, batches):
        self.loader, self.batches = loader, batches
        self.sampler = loader.sampler

    def __len__(self):
        return min(len(self.loader), self.batches)

    def __iter__(self):
        from itertools import islice
        yield from islice(self.loader, len(self))
