"""Resume pretraining at a committed epoch boundary after a shorter allocation."""
import json
from pathlib import Path


def validate_resume(saved, config, world_size):
    from src.training.pretrain_rng import validate_pretrain_rng
    validate_pretrain_rng(saved["config"], config)
    if saved.get("extra", {}).get("partial_epoch_smoke"):
        raise ValueError("Cannot resume from a partial smoke epoch")
    for key in ("ablation", "encoder", "position", "patch_encoder", "mae", "masking", "data", "optimization", "seed"):
        if saved["config"][key] != config[key]:
            raise ValueError("Resume config differs from checkpoint: " + key)
    if len(saved["rng_states"]) != world_size:
        raise ValueError("Resume must retain the checkpoint's distributed world size")
    if not 1 <= saved["epoch"] <= config["optimization"]["epochs"]:
        raise ValueError("Invalid checkpoint epoch")


def trim_partial_metrics(output, saved, tag):
    """Keep completed updates and preserve any interrupted-epoch log in a backup."""
    path = Path(output) / "metrics.jsonl"
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, discarded = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt interior metric record")
            discarded.append(line)
            continue
        if row["epoch"] <= saved["epoch"] and row["step"] <= saved["extra"]["step"]:
            kept.append(line)
        else:
            discarded.append(line)
    if discarded:
        backup = path.with_name("interrupted-metrics-" + tag + ".jsonl")
        if backup.exists():
            raise FileExistsError("Preserve the previous recovery audit: " + str(backup))
        backup.write_text("".join(discarded), encoding="utf-8")
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(kept), encoding="utf-8")
        temporary.replace(path)
    return len(discarded)
