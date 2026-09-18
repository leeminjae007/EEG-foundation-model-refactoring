"""Run a validated BCIC-IV-2a candidate through the isolated baseline engine."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one GPU per run to retain the requested effective batch")
    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from src.training.engine import run_finetune
    from ablation.bciciv2a.campaign import validate_recipe

    config = yaml.safe_load(args.config.read_text())
    validate_recipe(config)
    output = Path(config["runtime"]["output"])
    if not output.is_absolute():
        raise ValueError("Campaign output must be absolute")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        if saved["config"] != config or saved.get("extra", {}).get("partial_epoch_smoke"):
            raise ValueError("Resume requires the identical full-run config")
        del saved
    elif (output / "resolved_config.yaml").exists() or (output / "last.pth").exists():
        raise FileExistsError("Run already started; resume its last.pth explicitly")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_finetune(config, SimpleNamespace(device="cuda", distributed=True, smoke=False,
                                        resume=str(args.resume.resolve()) if args.resume else None))


if __name__ == "__main__":
    main()
