"""Atomic checkpoint I/O, distributed unwrapping, and RNG state handling."""

import os
import random
import tempfile

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def atomic_save(payload, path):
    descriptor, temporary = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["state"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def load_pretrain_checkpoint(
    path,
    context_encoder,
    decoder,
    optimizer,
    lr_scheduler,
    config_sha256,
    dataset_fingerprint,
    rank,
):
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["config_sha256"] == config_sha256
    assert checkpoint["dataset_fingerprint"] == dataset_fingerprint
    unwrap(context_encoder).load_state_dict(checkpoint["context_encoder"], strict=True)
    decoder_state = checkpoint.get("decoder", checkpoint.get("predictor"))
    if decoder_state is None:
        raise KeyError("checkpoint has no MAE decoder state")
    unwrap(decoder).load_state_dict(decoder_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    restore_rng_state(checkpoint["rng_states"][rank])
    return checkpoint
