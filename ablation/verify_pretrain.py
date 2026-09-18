"""CPU-only completion check for one ablation pretraining run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def verify(output, arm):
    checkpoint = output / "checkpoint-epoch-0040.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Expected final epoch-40 checkpoint: " + str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("epoch") != 40 or payload.get("extra", {}).get("partial_epoch_smoke"):
        raise ValueError("Checkpoint is not a complete epoch-40 pretraining run")
    if payload["config"]["ablation"]["name"] != arm:
        raise ValueError("Checkpoint ablation identity differs")
    weights = payload.get("model", {})
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint has missing or non-finite weights")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report = {"passed": True, "arm": arm, "epoch": 40,
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    path = output.parent / "checkpoint_verification.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    verify(args.output.resolve(), args.arm)
