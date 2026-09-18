"""Opt-in rank-specific training RNG, applied after common model initialization."""
import hashlib

import torch


def pretrain_rng_mode(config):
    mode = config.get("runtime", {}).get("pretrain_rank_rng", "shared")
    if mode not in ("shared", "independent"):
        raise ValueError("Unknown pretrain_rank_rng: " + str(mode))
    return mode


def initialize_pretrain_rng(config, rank):
    if pretrain_rng_mode(config) == "independent":
        # Match GR2: torch drives masking; each mask gets a local numpy generator.
        torch.manual_seed(config["seed"] + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"] + rank)


def validate_pretrain_rng(saved_config, config):
    if pretrain_rng_mode(saved_config) != pretrain_rng_mode(config):
        raise ValueError("Resume config differs from checkpoint: pretrain_rank_rng")


def pretrain_rng_report(config, device, rank, resumed):
    report = dict(mode=pretrain_rng_mode(config), rank=rank,
                  source="checkpoint" if resumed else "fresh",
                  seed=config["seed"] + rank if pretrain_rng_mode(config) == "independent" else config["seed"],
                  torch_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if device.type == "cuda":
        report["cuda_sha256"] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    return report
