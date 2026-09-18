"""Train and test five seeds for three validation-locked KNN37 hyperparameters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/knn37_local_search_20260916_1157"
SLUGS = ("mentalarithmetic", "isruc", "hmc")
SEEDS = (42, 696, 1001, 1234, 3407)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base(slug: str) -> Path:
    return CAMPAIGN / "datasets" / slug / "locked_five_seed"


def plan(slug: str) -> dict:
    return read(base(slug) / "plan.json")


def prepare() -> None:
    import yaml
    from ablation.local_search.campaign import cid, config_for

    manifest = read(CAMPAIGN / "manifest.json")
    for slug in SLUGS:
        directory = base(slug)
        if directory.exists():
            raise FileExistsError(directory)
        lock = read(CAMPAIGN / "datasets" / slug / "screen_test_preview/selection.json")
        if not lock["locked_before_test"] or not lock["provisional"]:
            raise ValueError(f"{slug}: screen selection not locked")
        if cid(lock["hp"]) != lock["candidate"]:
            raise ValueError(f"{slug}: candidate hash mismatch")
        dataset = manifest["datasets"][slug]
        entries = []
        for seed in SEEDS:
            baseline = next(e for e in dataset["baseline"] if int(e["seed"]) == seed)
            if seed == int(lock["seed"]):
                origin = next(e for e in read(CAMPAIGN / "datasets" / slug / "screen/plan.json")["entries"]
                              if e["candidate"] == lock["candidate"] and int(e["seed"]) == seed)
                config = Path(origin["config"])
                output = Path(origin["output"])
                if digest(config) != lock["config_sha256"] or digest(output / "best-balanced_accuracy.pth") != lock["checkpoint_sha256"]:
                    raise ValueError(f"{slug}: selected screen checkpoint changed")
                reuse = True
            else:
                output = directory / "train" / f"seed{seed}"
                config = directory / "configs" / f"seed{seed}.yaml"
                content = config_for(yaml.safe_load(Path(baseline["config"]).read_text()), lock["hp"], output)
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
                reuse = False
            entries.append(dict(slug=slug, seed=seed, hp=lock["hp"], candidate=lock["candidate"],
                                config=str(config), config_sha256=digest(config), output=str(output),
                                reused_screen_training=reuse, test_output=str(directory / "test" / f"seed{seed}")))
        write(directory / "plan.json", dict(slug=slug, seeds=list(SEEDS),
              decision="single-seed validation screen leader locked before test",
              selection_file=str(CAMPAIGN / "datasets" / slug / "screen_test_preview/selection.json"),
              entries=entries, prepared_utc=datetime.now(timezone.utc).isoformat()))
        print(f"prepared {slug}: four new training seeds, five test seeds", flush=True)


def train(slug: str, index: int) -> None:
    import torch
    import yaml
    from src.training import engine
    from src.training.runtime import set_paths
    entry = [e for e in plan(slug)["entries"] if not e["reused_screen_training"]][index]
    output = Path(entry["output"])
    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    if digest(Path(entry["config"])) != entry["config_sha256"] or config["runtime"].get("evaluate_test") is not False:
        raise ValueError("Training config changed or test gate disabled")
    set_paths()
    checkpoint = output / "last.pth"
    if (output / "result.json").exists():
        verify_completed_training(entry)
        print(f"already trained {slug} seed {entry['seed']}", flush=True)
        return
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume config changed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    engine.run_finetune(config, SimpleNamespace(device="cuda", distributed=False, smoke=False,
                        resume=str(checkpoint) if checkpoint.exists() else None))
    verify_completed_training(entry)
    print(f"trained {slug} seed {entry['seed']}", flush=True)


def verify_completed_training(entry: dict) -> dict:
    """The standard finetune engine tests validation-selected checkpoints at exit."""
    output = Path(entry["output"])
    result = read(output / "result.json")
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines() if line.strip()]
    if not history or not (output / "best-balanced_accuracy.pth").is_file():
        raise ValueError("Missing validation history or selected checkpoint")
    selected = result["balanced_accuracy"]["selection"]
    if selected["score"] != max(row["balanced_accuracy"] for row in history):
        raise ValueError("Checkpoint was not selected by validation")
    if history[selected["epoch"] - 1]["balanced_accuracy"] != selected["score"]:
        raise ValueError("Selected epoch does not match validation history")
    metrics = result["balanced_accuracy"].get("test")
    if not isinstance(metrics, dict) or not math.isfinite(float(metrics["balanced_accuracy"])):
        raise ValueError("Missing or invalid test result")
    return result


def evaluate(slug: str, index: int) -> None:
    import torch
    import yaml
    from src.training import engine
    from src.training.runtime import set_paths

    entry = plan(slug)["entries"][index]
    target = Path(entry["test_output"])
    if (target / "result.json").exists():
        print(f"already tested {slug} seed {entry['seed']}", flush=True)
        return
    lock = read(CAMPAIGN / "datasets" / slug / "screen_test_preview/selection.json")
    if entry["hp"] != lock["hp"] or digest(Path(entry["config"])) != entry["config_sha256"]:
        raise ValueError("Locked hyperparameter or config changed")
    origin = Path(entry["output"])
    trained = read(origin / "result.json")
    selected = trained["balanced_accuracy"]["selection"]
    checkpoint = origin / "best-balanced_accuracy.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if entry["reused_screen_training"]:
        preview = read(CAMPAIGN / "datasets" / slug / "screen_test_preview/result.json")
        metrics = preview["balanced_accuracy"]["test"]
        if selected["score"] != lock["validation_balanced_accuracy"]:
            raise ValueError("Screen selection changed")
    else:
        metrics = verify_completed_training(entry)["balanced_accuracy"]["test"]
    write(target / "result.json", {"balanced_accuracy": {"selection": selected, "test": metrics},
          "slug": slug, "seed": entry["seed"], "hp": entry["hp"],
          "training_origin": str(origin), "tested_utc": datetime.now(timezone.utc).isoformat()})
    print(f"tested {slug} seed {entry['seed']}: {metrics['balanced_accuracy']}", flush=True)


def aggregate(slug: str) -> None:
    import csv
    entries = plan(slug)["entries"]
    if len(entries) != 5 or {int(e["seed"]) for e in entries} != set(SEEDS):
        raise ValueError("Need exactly five seeds")
    results = []
    for entry in entries:
        result = read(Path(entry["test_output"]) / "result.json")
        metrics = result["balanced_accuracy"]["test"]
        if not math.isfinite(float(metrics["balanced_accuracy"])):
            raise ValueError("Nonfinite test metric")
        results.append(dict(seed=entry["seed"], validation=result["balanced_accuracy"]["selection"]["score"],
                            test={k: float(v) for k, v in metrics.items()
                                  if isinstance(v, (float, int)) and math.isfinite(float(v))}))
    metrics = sorted(set.intersection(*(set(r["test"]) for r in results)))
    summary = {metric: {"mean": statistics.fmean(r["test"][metric] for r in results),
                        "population_sd": statistics.pstdev(r["test"][metric] for r in results), "n": 5}
               for metric in metrics}
    payload = dict(slug=slug, hp=entries[0]["hp"], selection="one-seed validation screen leader",
                   rows=results, summary=summary, completed_utc=datetime.now(timezone.utc).isoformat())
    target = base(slug)
    write(target / "summary.json", payload)
    with (target / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "mean", "population_sd", "n"])
        writer.writeheader()
        writer.writerows(dict(metric=metric, **values) for metric, values in summary.items())
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "aggregate"))
    parser.add_argument("--slug", choices=SLUGS)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.slug is None or (args.action != "aggregate" and args.index is None):
            raise ValueError("--slug and --index required for this action")
        {"train": train, "evaluate": evaluate, "aggregate": aggregate}[args.action](
            args.slug, *([] if args.action == "aggregate" else [args.index]))


if __name__ == "__main__":
    main()
