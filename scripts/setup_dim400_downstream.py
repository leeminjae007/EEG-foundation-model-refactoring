"""Prepare D=400 five-seed downstream configs without touching base campaigns."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CAMPAIGN = ROOT / "outputs/dim400_downstream_20260917"
CHECKPOINT = "outputs/pretrain_dim400_nearest3_7_20260917/checkpoint-epoch-0040.pth"
DATASETS = (
    ("tuab", "TUAB"),
    ("tuev", "TUEV"),
    ("chb", "CHB-MIT"),
    ("siena", "SIENA"),
    ("physionet_mi", "PHYSIONET-MI"),
    ("faced", "FACED"),
    ("seedv", "SEED-V"),
    ("mentalarithmetic", "Mental Arithmetic"),
    ("isruc", "ISRUC"),
    ("hmc", "HMC"),
)
SEEDS = (42, 696, 1001, 1234, 3407)


def contains_warmup(value):
    if isinstance(value, dict):
        return any("warmup" in str(key).lower() or contains_warmup(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(contains_warmup(child) for child in value)
    return False


def main():
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    CAMPAIGN.mkdir(parents=True)
    (CAMPAIGN / "configs").mkdir()
    entries = []
    for slug, display in DATASETS:
        for seed in SEEDS:
            template = ROOT / f"configs/downstream/gr9-1_{slug}_seed{seed}.yaml"
            config = yaml.safe_load(template.read_text(encoding="utf-8"))
            if contains_warmup(config):
                raise ValueError(f"warmup forbidden: {template}")
            if config["seed"] != seed:
                raise ValueError(f"seed mismatch: {template}")
            config["model"]["checkpoint"] = CHECKPOINT
            # Preserve effective batch size while fitting the wider backbone.
            old_batch = config["optimization"]["batch_size_per_gpu"]
            if old_batch >= 2:
                config["optimization"]["batch_size_per_gpu"] = old_batch // 2
                config["optimization"]["gradient_accumulation_steps"] *= 2
            config["data"]["num_workers"] = 2
            output_relative = f"outputs/dim400_downstream_20260917/downstream/{slug}_seed{seed}"
            config["runtime"]["output"] = output_relative
            path = CAMPAIGN / "configs" / f"{slug}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            entries.append({
                "index": len(entries), "dataset": config["data"]["dataset"],
                "display_name": display, "slug": slug, "seed": seed,
                "config": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output": str(ROOT / output_relative), "result_dir": str(ROOT / output_relative),
                "checkpoint": str(ROOT / CHECKPOINT),
            })
    manifest = {
        "pretrain_job_id": "27515757",
        "checkpoint": str(ROOT / CHECKPOINT),
        "scope": "10 datasets, five seeds each",
        "policy": "Validation-balanced-accuracy checkpoint selection; no downstream warmup",
        "submitted_entries": entries,
        "reused_entries": [],
    }
    (CAMPAIGN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (CAMPAIGN / "array.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"campaign": str(CAMPAIGN), "entries": len(entries)}))


if __name__ == "__main__":
    main()
