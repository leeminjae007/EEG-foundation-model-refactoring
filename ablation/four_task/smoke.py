"""Check the four full models against real train samples before GPU submission."""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from ablation.four_task.campaign import RECIPES, validate_recipe
from ablation.tuab.campaign import digest, verify_source, write_json


def run(campaign):
    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from src.training import engine
    from src.training.checkpoint import load_checkpoint, rng_state, save_checkpoint
    from src.training.diagnostics import fingerprint

    manifest = verify_source(campaign)
    manifest_hash = digest(campaign / "manifest.json")
    destination = campaign / "smoke_validation.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["passed"] and old["manifest_sha256"] == manifest_hash:
            print("Exact snapshot already passed smoke")
            return
    torch.set_num_threads(2)
    device = torch.device("cpu")
    checks = []
    for name in RECIPES:
        entry = next(e for e in manifest["entries"] if e["dataset"] == name and e["seed"] == 42)
        config = yaml.safe_load(Path(entry["config"]).read_text())
        validate_recipe(config)
        spec = engine.get_dataset_spec(name)
        dataset = spec.dataset_class(config["data"]["dataset_dir"], "train")
        dataset.enable_coordinate_only_channels()
        if len(dataset) < 2:
            raise ValueError("Training split is empty")
        # ISRUC keeps the complete 20-epoch sequence, using one sequence per CPU update.
        batch_size = 1 if name == "isruc" else 2
        fixture = Subset(dataset, [0, len(dataset) - 1] * (1 if name == "isruc" else 2))
        loader = DataLoader(fixture, batch_size=batch_size, num_workers=0,
                            sampler=DistributedSampler(fixture, num_replicas=1, rank=0, seed=42))
        engine.setup("cpu", False, 42, False, False, True)
        model = engine.build_finetune(config)
        before = fingerprint(model)["tensors"]
        baseline = json.loads((Path(entry["baseline_result_dir"]) / "initialization.json").read_text())["tensors"]
        if {k: v for k, v in before.items() if k.startswith("backbone.")} != {k: v for k, v in baseline.items() if k.startswith("backbone.")}:
            raise AssertionError("Pretrained backbone differs from the completed baseline")
        if name != "hmc" and before != baseline:
            raise AssertionError("Unchanged head initialization differs from baseline")
        optimization = config["optimization"]
        optimizer = torch.optim.AdamW(engine.optimizer_groups(model, optimization),
                    betas=tuple(optimization["adam_betas"]), eps=optimization["adam_epsilon"],
                    weight_decay=optimization["weight_decay"])
        scheduler = engine.GroupCosineScheduler(optimizer, 100, optimization["min_learning_rate"])
        smoke_config = deepcopy(config)
        smoke_config["optimization"]["gradient_accumulation_steps"] = 1
        smoke_config["runtime"]["log_every_steps"] = 1
        work = campaign / "smoke" / RECIPES[name]["slug"]
        work.mkdir(parents=True, exist_ok=True)
        store_path = work / ("ddp-" + uuid.uuid4().hex)
        torch.distributed.init_process_group("gloo", store=torch.distributed.FileStore(str(store_path), 1), rank=0, world_size=1)
        try:
            trainer = DistributedDataParallel(model)
            steps = engine.train_finetune_epoch(model, trainer, loader, optimizer, scheduler, spec,
                    smoke_config, SimpleNamespace(smoke=False), device, 0, 0, 0, work)
            if steps != 2:
                raise AssertionError("Expected two optimizer updates")
            after = fingerprint(model)["tensors"]
            for prefix in ("backbone.", "head."):
                if all(value == before[key] for key, value in after.items() if key.startswith(prefix)):
                    raise AssertionError("Parameters did not update: " + prefix)
            checkpoint = work / "last.pth"
            save_checkpoint(checkpoint, model, optimizer, scheduler, 1, config, [rng_state(device)],
                            {"step": steps, "partial_epoch_smoke": True})
            epoch, extra = load_checkpoint(checkpoint, model, optimizer, scheduler, device, 0)
            if epoch != 1 or extra["step"] != steps or fingerprint(model)["tensors"] != after:
                raise AssertionError("Checkpoint round-trip differs")
            checkpoint.unlink()
            records = [json.loads(line) for line in (work / "metrics.jsonl").read_text().splitlines()[-2:]]
            check = {"dataset": name, "passed": True, "training_samples": len(dataset),
                     "smoke_batch": batch_size, "updates": steps, "effective_training_batch": 64,
                     "head_parameters": sum(p.numel() for p in model.head.parameters()),
                     "losses": [r["loss"] for r in records], "first_lr": records[0]["lr_used"],
                     "strict_checkpoint_roundtrip": True, "test_data_read": False,
                     "backbone_initialization_matches_baseline": True}
            checks.append(check)
            print(json.dumps(check), flush=True)
        finally:
            torch.distributed.destroy_process_group()
            store_path.unlink(missing_ok=True)
            if getattr(dataset, "database", None) is not None:
                dataset.database.close()
        del trainer, model, optimizer, scheduler, loader, fixture, dataset
        gc.collect()
    write_json(destination, {"passed": True, "manifest_sha256": manifest_hash,
                            "checks": checks, "device": "cpu", "torch_version": torch.__version__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    run(parser.parse_args().campaign.resolve())
