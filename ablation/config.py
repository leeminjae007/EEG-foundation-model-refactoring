"""Resolve small ablation overlays without changing existing YAML files."""

from copy import deepcopy
from pathlib import Path

import yaml

from ablation.bootstrap import ROOT
from ablation.encoders import ENCODERS
from ablation.positions import POSITIONS


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, seen=()):
    path = (ROOT / path).resolve()
    if path in seen:
        raise ValueError("Circular base_config: " + str(path))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("base_config", None)
    if parent:
        config = merge(load_config(parent, seen + (path,)), config)
    return config


def resolve_ablation(config):
    config = deepcopy(config)
    settings = config.setdefault("ablation", {})
    allowed = {"encoder", "position", "mask_mode", "pe_scope", "depth", "name",
               "reve_freqs", "reve_noise_ratio", "channel_vocabulary"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError("Unknown ablation options: " + str(sorted(unknown)))
    defaults = {"encoder": "mjde", "position": "shpe", "mask_mode": "context_only",
                "pe_scope": "both", "reve_freqs": 4, "reve_noise_ratio": 0.0}
    for key, value in defaults.items():
        settings.setdefault(key, value)
    if settings["encoder"] not in ENCODERS or settings["position"] not in POSITIONS:
        raise ValueError("Unknown encoder or position embedding")
    if settings["mask_mode"] not in ("context_only", "dense_zero"):
        raise ValueError("mask_mode must be context_only or dense_zero")
    if settings["pe_scope"] not in ("encoder", "both"):
        raise ValueError("pe_scope must be encoder or both")
    if settings["encoder"] in ("labram", "cbramod", "csbrain") and settings["mask_mode"] != "dense_zero":
        raise ValueError("Unmodified paper encoders require the common dense_zero protocol")
    if "depth" in settings and (settings["encoder"] in ("mjde", "mjde_lite") or
                                not isinstance(settings["depth"], int) or settings["depth"] < 1):
        raise ValueError("depth is a positive integer for paper encoders only; MJDE-lite is one stage")
    for location, expected in (("encoder", config["encoder"]["embed_dim"]),
                               ("decoder", config["mae"]["decoder_dim"])):
        dim = config["position"][location + "_spatial_dim"] + config["position"][location + "_temporal_dim"]
        if dim != expected:
            raise ValueError(location + " PE width differs from model width")
    if config["patch_encoder"]["embed_dim"] != config["encoder"]["embed_dim"]:
        raise ValueError("Tokenizer and encoder widths must match")
    if settings["position"] == "channel_id" and "channel_vocabulary" not in settings:
        import mne
        from ablation.montage import canonical
        from src.data.datasets.registry import DATASET_SPECS
        names = set(mne.channels.make_standard_montage("standard_1005").ch_names)
        names.update(config["data"]["channel_names"])
        for spec in DATASET_SPECS.values():
            names.update(spec.dataset_class.channel_names)
        settings["channel_vocabulary"] = sorted({canonical(name) for name in names})
    settings.setdefault("name", settings["encoder"] + "_" + settings["position"] + "_" + settings["mask_mode"])
    if not settings["name"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in settings["name"]):
        raise ValueError("ablation.name must contain only letters, digits, '_' and '-'")
    return config
