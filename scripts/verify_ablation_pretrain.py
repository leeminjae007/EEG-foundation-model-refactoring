"""Audit a raw ablation epoch-40 checkpoint before downstream submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(experiment: Path) -> dict:
    import torch

    experiment = experiment.resolve()
    checkpoint = experiment / "pretrain" / "checkpoint-epoch-0040.pth"
    saved = torch.load(checkpoint, map_location="cpu")
    required = ("config", "model", "optimizer", "scheduler", "rng_states", "epoch", "extra")
    missing = [key for key in required if key not in saved or saved[key] is None]
    if missing:
        raise ValueError("Missing checkpoint state: " + ", ".join(missing))
    config = saved["config"]
    expected_epoch = int(config["optimization"]["epochs"])
    if saved["epoch"] != expected_epoch or expected_epoch != 40:
        raise ValueError("Expected complete epoch-40 checkpoint")
    extra = saved["extra"]
    if extra.get("partial_epoch_smoke") or extra.get("step", 0) < 1 or not extra.get("dataset_fingerprint"):
        raise ValueError("Checkpoint is a smoke/incomplete run or lacks dataset provenance")
    if len(saved["rng_states"]) != 4:
        raise ValueError("Expected four rank RNG states")
    rng_independent = {
        key: len({state[key].numpy().tobytes() for state in saved["rng_states"]}) == 4
        for key in ("torch", "cuda")
    }
    if any(not torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("Checkpoint contains non-finite model weights")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from ablation.models import build_pretrain

    model = build_pretrain(config, torch.device("cpu"))
    model.load_state_dict(saved["model"], strict=True)
    report = {
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "epoch": saved["epoch"],
        "step": extra["step"],
        "strict_load": True,
        "partial_epoch_smoke": False,
        "world_size": 4,
        "rng_states_independent": rng_independent,
        "kind": "raw_ablation_pretrain",
    }
    (experiment / "pretrain" / "verified.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().experiment), indent=2))
