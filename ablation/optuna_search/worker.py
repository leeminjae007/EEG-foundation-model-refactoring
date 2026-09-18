"""One GPU executes one fixed seed. It never writes the Optuna database."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from ablation.optuna_search.campaign import read, write, verify, digest, validate_seed


def run(campaign, plan_path):
    verify(campaign)
    plan = read(plan_path)
    entry = plan["entries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = Path(entry["output"])
    if (output / "completed.json").exists():
        validate_seed(entry)
        return
    cfg = yaml.safe_load(Path(entry["config"]).read_text())
    if digest(entry["config"]) != entry["config_sha256"]:
        raise ValueError("Config hash changed")
    from src.training.runtime import set_paths
    set_paths()
    import torch
    from src.training import engine
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pth"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["extra"].get("partial_epoch_smoke"):
            raise ValueError("Unsafe resume config or partial epoch")
        last_epoch = state["epoch"]
        del state
        # An interrupted epoch can have logged validation before last.pth was committed.
        for name in ("validation.jsonl", "test_history.jsonl", "metrics.jsonl"):
            path = output / name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                rows = [r for r in rows if r["epoch"] <= last_epoch]
                tmp = path.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
                tmp.replace(path)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    engine.run_finetune(cfg, SimpleNamespace(device="cuda", distributed=False, smoke=False,
                        resume=str(checkpoint) if checkpoint.exists() else None))
    summary = validate_seed(entry)
    write(output / "completed.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    a = p.parse_args()
    run(a.campaign, a.plan)
