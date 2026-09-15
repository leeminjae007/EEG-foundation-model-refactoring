"""Fine-tune the exact architecture recorded by an ablation pretrain checkpoint."""

import argparse
from copy import deepcopy
import json
import os

from ablation.bootstrap import ROOT, ensure_data_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="An existing configs/downstream YAML")
    parser.add_argument("--checkpoint", required=True, help="Corresponding ablation pretraining checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--data-dir")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.smoke = False

    from src.training.runtime import set_paths
    set_paths()
    data_source = ensure_data_imports()
    import torch
    import yaml
    from ablation.config import load_config
    from ablation.integration import connected_engine
    from ablation.sources import verify_sources

    provenance = verify_sources()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ablation" not in checkpoint["config"]:
        raise ValueError("Use an ablation pretraining checkpoint, not an old MJDE or upstream full-model checkpoint")
    if checkpoint.get("extra", {}).get("partial_epoch_smoke"):
        raise ValueError("A smoke checkpoint cannot be used for downstream comparisons")
    config = load_config(args.config)
    config["ablation"] = deepcopy(checkpoint["config"]["ablation"])
    config["model"]["checkpoint"] = str(checkpoint_path)
    if args.data_dir:
        config["data"]["dataset_dir"] = str((ROOT / args.data_dir).resolve())
    if args.seed is not None:
        config["seed"] = args.seed
    dataset = config["data"]["dataset"]
    name = config["ablation"]["name"]
    # Include the downstream config stem, preserving extra LR arms as separate runs.
    config["runtime"]["output"] = args.output or ("outputs/ablation/" + name + "/finetune/" +
                                                   dataset + "/" + os.path.splitext(os.path.basename(args.config))[0] +
                                                   "_seed" + str(config["seed"]))
    if args.device == "cpu" or args.dry_run:
        torch.set_num_threads(2)
    if args.resume:
        args.resume = str((ROOT / args.resume).resolve())
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("ablation", "model", "data", "optimization", "seed"):
            if saved["config"][key] != config[key]:
                raise ValueError("Resume config differs from checkpoint: " + key)
    output = ROOT / config["runtime"]["output"]
    report = {"ablation": config["ablation"], "data_source": data_source, "sources": provenance,
              "pretrain_checkpoint": str(checkpoint_path), "output": str(output)}
    with connected_engine() as engine:
        if args.dry_run:
            model = engine.build_finetune(config)
            report["parameters"] = sum(p.numel() for p in model.parameters())
            print(json.dumps(report, indent=2))
            return
        if int(os.environ.get("RANK", "0")) == 0:
            if (output / "last.pth").exists() and not args.resume:
                raise FileExistsError("Run exists; use --resume or a different --output: " + str(output))
            output.mkdir(parents=True, exist_ok=True)
            (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            (output / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
        engine.run_finetune(config, args)


if __name__ == "__main__":
    main()
