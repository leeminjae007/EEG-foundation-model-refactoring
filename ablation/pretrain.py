"""python -m ablation.pretrain --config ablation/configs/encoder_labram.yaml"""

import argparse
import json
import multiprocessing
import os

from ablation.bootstrap import ROOT, ensure_data_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One update on real training data")
    parser.add_argument("--resume")
    parser.add_argument("--data-dir")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and count model parameters, no data/GPU required")
    args = parser.parse_args()

    from src.training.runtime import set_paths
    set_paths()
    data_source = ensure_data_imports()
    import torch
    import yaml
    from ablation.config import load_config, resolve_ablation
    from ablation.integration import connected_engine
    from ablation.models import build_pretrain, parameter_report
    from ablation.sources import verify_sources

    provenance = verify_sources()
    config = resolve_ablation(load_config(args.config))
    if args.data_dir:
        config["data"]["dataset_dir"] = str((ROOT / args.data_dir).resolve())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        config["optimization"]["batch_size_per_gpu"] = args.batch_size
    name = config["ablation"]["name"]
    config["runtime"]["output"] = args.output or "outputs/ablation/" + name + "/seed" + str(config["seed"])
    output = ROOT / config["runtime"]["output"]
    if args.smoke:
        # The unchanged engine uses the output basename for geometry smoke runs.
        config["runtime"]["output"] = str(output / (name + "_seed" + str(config["seed"])))
        output = ROOT / "outputs/smoke" / (name + "_seed" + str(config["seed"]))
        if config["masking"].get("policy") != "geometry_tubelet":
            parser.error("Real-data --smoke uses geometry_tubelet; use ablation.smoke for synthetic checks")
    if args.device == "cpu" or args.dry_run:
        torch.set_num_threads(2)
    if args.resume:
        args.resume = str((ROOT / args.resume).resolve())
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["extra"].get("partial_epoch_smoke"):
            raise ValueError("Cannot resume training from a one-update smoke checkpoint")
        if saved["config"]["ablation"] != config["ablation"]:
            raise ValueError("Resume ablation config differs from checkpoint")
        for key in ("encoder", "position", "patch_encoder", "mae", "data", "optimization", "seed"):
            if saved["config"][key] != config[key]:
                raise ValueError("Resume config differs from checkpoint: " + key)
    report = {"ablation": config["ablation"], "data_source": data_source, "sources": provenance,
              "output": str(output), "seed": config["seed"]}
    if args.dry_run:
        torch.manual_seed(config["seed"])
        report["parameters"] = parameter_report(build_pretrain(config, torch.device("cpu")))
        print(json.dumps(report, indent=2))
        return
    if int(os.environ.get("RANK", "0")) == 0:
        if (output / "last.pth").exists() and not args.resume:
            raise FileExistsError("Run exists; use --resume or a different --output: " + str(output))
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (output / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    multiprocessing.current_process()._config["tempdir"] = os.environ.get("TMPDIR", "/tmp")
    if "file_descriptor" in torch.multiprocessing.get_all_sharing_strategies():
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
    with connected_engine() as engine:
        engine.run_pretrain(config, args)


if __name__ == "__main__":
    main()
