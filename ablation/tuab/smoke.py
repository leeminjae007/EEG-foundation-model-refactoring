"""CPU check with two real training samples, full model and one-rank DDP."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import uuid

from ablation.bootstrap import ensure_data_imports
from ablation.tuab.campaign import ARMS, digest, verify_source, write_json


def run(campaign):
    from src.training.runtime import set_paths
    set_paths()
    ensure_data_imports()
    import torch
    import yaml
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, Subset, DistributedSampler
    from src.training import engine
    from src.training.checkpoint import load_checkpoint, rng_state, save_checkpoint
    from src.training.diagnostics import fingerprint
    from src.modules.loss import downstream_loss
    from ablation.tuab.policy import FineTunePolicy

    manifest = verify_source(campaign)
    manifest_hash = digest(campaign / "manifest.json")
    report_path = campaign / "smoke_validation.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["manifest_sha256"] == manifest_hash and previous["passed"]:
            print("Exact snapshot already passed real-data smoke")
            return
    torch.set_num_threads(2)
    device = torch.device("cpu")
    work = campaign / "smoke"
    work.mkdir(exist_ok=True)
    checks = []
    common_initialization = None
    for arm in manifest["arms"]:
        entry = next(e for e in manifest["entries"] if e["arm"] == arm and e["seed"] == 42)
        config = yaml.safe_load(Path(entry["config"]).read_text())
        dataset = engine.get_dataset_spec("tuab").dataset_class(config["data"]["dataset_dir"], "train")
        dataset.enable_coordinate_only_channels()
        if len(dataset) != sum(config["optimization"]["class_counts"]):
            raise ValueError("TUAB training sample count differs from the baseline")
        batch = next(iter(DataLoader(Subset(dataset, [0, 1]), batch_size=2, num_workers=0)))
        engine.setup("cpu", False, 42, False, False, True)
        model = engine.build_finetune(config)
        initial = fingerprint(model)["tensors"]
        baseline_initial = json.loads((Path(entry["baseline_result_dir"]) / "initialization.json").read_text())["tensors"]
        if {k: v for k, v in initial.items() if k.startswith("backbone.")} != {k: v for k, v in baseline_initial.items() if k.startswith("backbone.")}:
            raise AssertionError("Backbone initialization differs from the completed baseline")
        if arm in ARMS[:3]:
            if initial != baseline_initial:
                raise AssertionError("Initialization differs from the completed baseline")
            if common_initialization is None:
                common_initialization = initial
            elif initial != common_initialization:
                raise AssertionError("Common-head arms must initialize identically")
        backbone_before = fingerprint(model.backbone)["tensors"]
        head_before = fingerprint(model.head)["tensors"]
        opt_config = config["optimization"]
        optimizer = torch.optim.AdamW(engine.optimizer_groups(model, opt_config),
                    betas=tuple(opt_config["adam_betas"]), eps=opt_config["adam_epsilon"],
                    weight_decay=opt_config["weight_decay"])
        updates = math.ceil(math.ceil(len(dataset) / opt_config["batch_size_per_gpu"]) /
                            opt_config["gradient_accumulation_steps"])
        policy = FineTunePolicy(config["finetune_ablation"])
        scheduler = policy.make_scheduler(optimizer, 20 * updates, opt_config["min_learning_rate"])
        store_path = work / ("ddp-" + uuid.uuid4().hex)
        torch.distributed.init_process_group("gloo", store=torch.distributed.FileStore(str(store_path), 1), rank=0, world_size=1)
        phases = []
        try:
            trainer = DistributedDataParallel(model, find_unused_parameters=bool(policy.head_first_epochs))
            phase_output = work / arm
            phase_output.mkdir(exist_ok=True)
            fixture = Subset(dataset, [0, 1, 0, 1])
            loader = DataLoader(fixture, batch_size=2,
                                sampler=DistributedSampler(fixture, num_replicas=1, rank=0, seed=42), num_workers=0)
            smoke_config = deepcopy(config)
            smoke_config["optimization"]["gradient_accumulation_steps"] = 1
            smoke_config["runtime"]["log_every_steps"] = 1
            epochs = range(4) if policy.head_first_epochs else range(1)
            for epoch in epochs:
                # Use the real start-of-epoch LR, while exercising two small updates per phase.
                scheduler.step_index = epoch * updates
                if hasattr(engine, "train_finetune_epoch"):
                    engine.train_finetune_epoch(model, trainer, loader, optimizer, scheduler,
                        engine.get_dataset_spec("tuab"), smoke_config, SimpleNamespace(smoke=False),
                        device, 0, epoch, epoch * updates, phase_output, policy=policy)
                    last = json.loads((phase_output / "metrics.jsonl").read_text().splitlines()[-1])
                    loss, norm, rates = last["loss"], last["pre_clip_norm"], last["lr_used"]
                else:
                    trainer.train()
                    policy.on_epoch_start(model, epoch)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        logits = engine.predict(trainer, batch, device)
                        loss = downstream_loss("binary", logits, batch["label"], opt_config)
                        if not torch.isfinite(loss):
                            raise AssertionError("Nonfinite smoke loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt_config["gradient_clip_norm"], error_if_nonfinite=True)
                        rates = scheduler.step()
                        optimizer.step()
                frozen = epoch < policy.head_first_epochs
                backbone_same = fingerprint(model.backbone)["tensors"] == backbone_before
                if frozen:
                    if not backbone_same or any(p in optimizer.state for p in model.backbone.parameters()) or model.backbone.training:
                        raise AssertionError("Frozen backbone changed weights, optimizer state or training mode")
                elif backbone_same or not any(p in optimizer.state for p in model.backbone.parameters()):
                    raise AssertionError("Unfrozen backbone did not update")
                if fingerprint(model.head)["tensors"] == head_before:
                    raise AssertionError("Head did not update")
                phases.append({"epoch": epoch + 1, "frozen": frozen, "loss": float(loss),
                               "pre_clip_norm": float(norm), "lr_used": rates})
            # Full real-model checkpoint round-trip, kept separate from scientific outputs.
            checkpoint = work / (arm + "-last.pth")
            expected = fingerprint(model)["tensors"]
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch + 1, config, [rng_state(device)],
                            {"step": scheduler.step_index, "partial_epoch_smoke": True})
            epoch_saved, _ = load_checkpoint(checkpoint, model, optimizer, scheduler, device, 0)
            if epoch_saved != epoch + 1 or fingerprint(model)["tensors"] != expected:
                raise AssertionError("Checkpoint round-trip differs")
            checkpoint.unlink()
            checks.append({"arm": arm, "passed": True, "training_samples": len(dataset),
                           "smoke_batch_size": 2, "head_parameters": sum(p.numel() for p in model.head.parameters()),
                           "phases": phases, "strict_checkpoint_roundtrip": True,
                           "ddp_world_size": 1, "test_data_read": False})
            print(json.dumps(checks[-1]), flush=True)
        finally:
            torch.distributed.destroy_process_group()
            store_path.unlink(missing_ok=True)
        del trainer, model, optimizer, scheduler, dataset, batch
    write_json(report_path, {"passed": True, "manifest_sha256": manifest_hash,
                            "torch_version": torch.__version__, "device": "cpu", "checks": checks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    run(parser.parse_args().campaign.resolve())
