"""Exercise the complete validation-only engine on real CPU train/val samples."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from ablation.bciciv2a.campaign import load_plan, validate_recipe
from ablation.tuab.campaign import digest, verify_source, write_json


def run(campaign):
    manifest = verify_source(campaign)
    manifest_hash = digest(campaign / "manifest.json")
    report_path = campaign / "smoke_validation.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old["passed"] and old["manifest_sha256"] == manifest_hash:
            print("Exact snapshot already passed full validation-only smoke", flush=True)
            return
    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from torch.utils.data import Subset
    from src.training import engine
    from src.training.checkpoint import load_checkpoint
    from src.training.diagnostics import fingerprint
    torch.set_num_threads(2)
    plan = load_plan(campaign, "round_00")
    for e in plan["entries"]:
        validate_recipe(yaml.safe_load(Path(e["config"]).read_text()))
    config = yaml.safe_load(Path(plan["entries"][0]["config"]).read_text())
    work = campaign / "smoke" / uuid.uuid4().hex
    work.mkdir(parents=True)
    fixture_config = deepcopy(config)
    fixture_config["optimization"].update(epochs=1, batch_size_per_gpu=2)
    fixture_config["data"]["num_workers"] = 0
    fixture_config["runtime"].update(output=str(work), log_every_steps=1)
    original_spec = engine.get_dataset_spec
    spec = original_spec("bciciv2a")
    opened = []
    lengths = {}

    class Fixture(Subset):
        def enable_coordinate_only_channels(self):
            self.dataset.enable_coordinate_only_channels()

    def train_val_only(path, split):
        if split == "test":
            raise AssertionError("Search opened the held-out test dataset")
        dataset = spec.dataset_class(path, split)
        opened.append(split)
        lengths[split] = len(dataset)
        return Fixture(dataset, [round(i * (len(dataset) - 1) / 7) for i in range(8)])

    engine.get_dataset_spec = lambda name: SimpleNamespace(**dict(vars(spec), dataset_class=train_val_only))
    try:
        engine.run_finetune(fixture_config, SimpleNamespace(device="cpu", distributed=False, smoke=False, resume=None))
    finally:
        engine.get_dataset_spec = original_spec
    result = json.loads((work / "result.json").read_text())
    if opened != ["train", "val"] or any(r["test"] is not None for r in result.values()):
        raise AssertionError("Validation-only gate failed")
    initialization = json.loads((work / "initialization.json").read_text())["tensors"]
    base = next(e for e in manifest["baseline_entries"] if e["seed"] == 42)
    baseline = json.loads((Path(base["result_dir"]) / "initialization.json").read_text())["tensors"]
    if initialization != baseline:
        raise AssertionError("Backbone/head initialization differs from the baseline")
    model = engine.build_finetune(config)
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(engine.optimizer_groups(model, opt), betas=tuple(opt["adam_betas"]),
                                 eps=opt["adam_epsilon"], weight_decay=opt["weight_decay"])
    scheduler = engine.GroupCosineScheduler(optimizer, 4, opt["min_learning_rate"])
    epoch, extra = load_checkpoint(work / "last.pth", model, optimizer, scheduler, torch.device("cpu"), 0)
    if epoch != 1 or extra["step"] != 4:
        raise AssertionError("Full training/checkpoint round trip failed")
    after = fingerprint(model)["tensors"]
    for prefix in ("head.", "backbone."):
        if all(v == initialization[k] for k, v in after.items() if k.startswith(prefix)):
            raise AssertionError("No optimizer update for " + prefix)
    records = [json.loads(line) for line in (work / "metrics.jsonl").read_text().splitlines()]
    if any(value != 1e-4 for value in records[0]["lr_used"]):
        raise AssertionError("First update should use the full learning rate")
    # Smoke artifacts cannot be resumed by the strict 50-epoch finetune wrapper.
    report = {"passed": True, "manifest_sha256": manifest_hash, "device": "cpu",
              "torch_version": torch.__version__, "opened_splits": opened, "dataset_sizes": lengths,
              "full_engine_epochs": 1, "optimizer_updates": 4,
              "first_lr": records[0]["lr_used"], "losses": [r["loss"] for r in records],
              "strict_checkpoint_roundtrip": True, "initialization_matches_baseline": True,
              "test_data_read": False, "candidate_configs_checked": len(plan["entries"])}
    write_json(report_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    run(parser.parse_args().campaign.resolve())
