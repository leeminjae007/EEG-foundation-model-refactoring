"""CPU-only gate before submitting the dependent D=400 downstream array."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CHECKPOINT = ROOT / "outputs/pretrain_dim400_nearest3_7_20260917/checkpoint-epoch-0040.pth"
STATUS = ROOT / "outputs/dim400_downstream_20260917/pretrain_verified.json"


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    if checkpoint["epoch"] != 40 or config["encoder"]["embed_dim"] != 400:
        raise ValueError("D=400 epoch-40 checkpoint identity mismatch")
    if config["masking"]["spatial_selection"] != "nearest_channels":
        raise ValueError("Unexpected masking policy")
    from src.model import PretrainModel

    model = PretrainModel(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model"], strict=True)
    status = {
        "verified": True, "checkpoint": str(CHECKPOINT),
        "epoch": 40, "encoder_dim": 400,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
