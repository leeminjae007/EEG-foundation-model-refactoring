"""새 프로젝트 안에서 epoch 상태와 난수를 함께 저장한다."""

import random
import numpy as np
import torch


def rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, states, extra):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "epoch": epoch,
               "config": config, "rng_states": states, "extra": extra}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, device, rank, expected_masking=None):
    payload = torch.load(path, map_location="cpu")
    if expected_masking is not None:
        saved = dict(payload["config"]["masking"])
        expected = dict(expected_masking)
        saved.setdefault("policy", "ijepa_multiblock")
        expected.setdefault("policy", "ijepa_multiblock")
        if saved != expected:
            raise ValueError("checkpoint masking config differs; use its original config to resume")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng_states"][rank], device)
    return payload["epoch"], payload["extra"]
