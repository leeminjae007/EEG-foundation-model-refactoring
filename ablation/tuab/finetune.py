"""Run one TUAB configuration through the original engine with an optional policy."""

import argparse
import json
import os
from pathlib import Path

from ablation.bootstrap import ensure_data_imports


def validate_resume(config, saved):
    if saved.get("extra", {}).get("partial_epoch_smoke"):
        raise ValueError("A partial-epoch smoke checkpoint cannot resume a scientific run")
    if saved["config"] != config:
        raise ValueError("Resume requires the exact saved configuration, including output path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.smoke = False
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This campaign uses one GPU per run to preserve effective batch size 512")
    from src.training.runtime import set_paths
    set_paths()
    ensure_data_imports()
    import torch
    import yaml
    from src.training.engine import run_finetune
    from ablation.tuab.policy import FineTunePolicy

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["data"]["dataset"] != "tuab":
        raise ValueError("This launcher is scoped to TUAB")
    policy = FineTunePolicy(config["finetune_ablation"])
    if policy.head_first_epochs >= config["optimization"]["epochs"]:
        raise ValueError("Head preparation must leave full fine-tuning epochs")
    output = Path(config["runtime"]["output"])
    if not output.is_absolute():
        raise ValueError("Campaign output paths must be absolute")
    if args.resume:
        args.resume = str(args.resume.resolve())
        validate_resume(config, torch.load(args.resume, map_location="cpu"))
    elif (output / "last.pth").exists() or (output / "resolved_config.yaml").exists():
        raise FileExistsError("Run already started; resume its last.pth explicitly: " + str(output))
    if args.device == "cpu":
        torch.set_num_threads(2)
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(json.dumps({"config": str(args.config), "output": str(output),
                          "policy": config["finetune_ablation"]}), flush=True)
    run_finetune(config, args, policy=policy)


if __name__ == "__main__":
    main()
