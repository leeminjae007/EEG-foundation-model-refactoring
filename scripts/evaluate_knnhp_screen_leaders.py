"""Test validation-locked, single-seed screen leaders without changing the search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/knn37_local_search_20260916_1157"
SLUGS = ("mentalarithmetic", "isruc", "hmc")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(slug: str) -> Path:
    return CAMPAIGN / "datasets" / slug / "screen_test_preview"


def prepare() -> None:
    for slug in SLUGS:
        target = output(slug)
        if (target / "selection.json").exists():
            raise FileExistsError(target / "selection.json")
        plan = read(CAMPAIGN / "datasets" / slug / "screen/plan.json")
        if len(plan["entries"]) != 7:
            raise ValueError(f"{slug}: expected seven screen candidates")
        rows = []
        for entry in plan["entries"]:
            run = Path(entry["output"])
            result = read(run / "result.json")
            if result["balanced_accuracy"].get("test") is not None:
                raise ValueError(f"{slug}: screen unexpectedly accessed test")
            checkpoint = run / "best-balanced_accuracy.pth"
            if not checkpoint.is_file() or digest(Path(entry["config"])) != entry["config_sha256"]:
                raise ValueError(f"{slug}: checkpoint or config verification failed")
            rows.append((float(result["balanced_accuracy"]["selection"]["score"]), entry, checkpoint))
        score, entry, checkpoint = max(rows, key=lambda item: (item[0], item[1]["candidate"]))
        lock = {"locked_before_test": True, "decision": "best single-seed validation balanced_accuracy",
                "provisional": True, "slug": slug, "seed": entry["seed"], "candidate": entry["candidate"],
                "hp": entry["hp"], "validation_balanced_accuracy": score,
                "config": entry["config"], "config_sha256": entry["config_sha256"],
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "locked_utc": datetime.now(timezone.utc).isoformat()}
        write(target / "selection.json", lock)
        print(json.dumps(lock), flush=True)


def run(index: int) -> None:
    if index not in range(len(SLUGS)):
        raise ValueError(index)
    slug = SLUGS[index]
    target = output(slug)
    if (target / "result.json").exists():
        print(f"Already tested: {slug}", flush=True)
        return
    lock = read(target / "selection.json")
    if not lock["locked_before_test"] or digest(Path(lock["checkpoint"])) != lock["checkpoint_sha256"]:
        raise ValueError("Selection was not locked or checkpoint changed")
    if digest(Path(lock["config"])) != lock["config_sha256"]:
        raise ValueError("Config changed after selection")

    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from src.training import engine

    config = yaml.safe_load(Path(lock["config"]).read_text(encoding="utf-8"))
    device, rank, world = engine.setup("cuda", False, config["seed"], False, False, True)
    spec = engine.get_dataset_spec(config["data"]["dataset"])
    dataset = spec.dataset_class(config["data"]["dataset_dir"], "test")
    try:
        dataset.enable_coordinate_only_channels()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config["optimization"]["batch_size_per_gpu"],
            sampler=range(len(dataset)), num_workers=config["data"]["num_workers"],
            pin_memory=True, drop_last=False)
        model = engine.build_finetune(config).to(device)
        model.load_state_dict(torch.load(lock["checkpoint"], map_location=device, weights_only=False), strict=True)
        metrics = engine.evaluate(model, loader, spec.task, config["data"]["dataset"], device, world)
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    write(target / "result.json", {"balanced_accuracy": {
        "selection": {"score": lock["validation_balanced_accuracy"]}, "test": metrics},
        "provisional_single_seed": True, "selection_file": str(target / "selection.json"),
        "tested_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"slug": slug, "seed": lock["seed"], "test": metrics}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.index is None:
            raise ValueError("--index required")
        run(args.index)


if __name__ == "__main__":
    main()
