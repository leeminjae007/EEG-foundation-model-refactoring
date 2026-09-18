"""CPU-only preflight: real data and model, five short epochs per dataset."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from ablation.optuna_search.campaign import verify, read, write, digest, config_for, LR_KEYS


def run(campaign):
    m = verify(campaign)
    from src.training.runtime import set_paths
    set_paths()
    import torch
    from torch.utils.data import Subset
    from src.training import engine
    torch.set_num_threads(2)
    checks = []
    for slug, ds in m["datasets"].items():
        base = yaml.safe_load(Path(ds["bases"]["42"]).read_text())
        output = campaign / "smoke" / slug
        output.mkdir(parents=True, exist_ok=False)
        cfg = config_for(base, slug, ds["initial"][0], output)
        cfg["data"]["num_workers"] = 0
        cfg["optimization"].update(epochs=5, batch_size_per_gpu=1, gradient_accumulation_steps=1)
        cfg["runtime"].update(log_every_steps=1, early_stopping={"min_epochs": 5, "patience": 3})
        original = engine.get_dataset_spec
        spec = original(cfg["data"]["dataset"])
        opened = []

        class Fixture(Subset):
            def enable_coordinate_only_channels(self):
                self.dataset.enable_coordinate_only_channels()

        def small_dataset(path, split):
            dataset = spec.dataset_class(path, split)
            opened.append(split)
            n = 2 if split == "train" else 8
            return Fixture(dataset, [round(i * (len(dataset) - 1) / (n - 1)) for i in range(n)])

        engine.get_dataset_spec = lambda name: SimpleNamespace(**dict(vars(spec), dataset_class=small_dataset))
        try:
            engine.run_finetune(cfg, SimpleNamespace(device="cpu", distributed=False, smoke=False, resume=None))
        finally:
            engine.get_dataset_spec = original
        result = read(output / "result.json")["balanced_accuracy"]
        history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
        if opened != ["train", "val", "test"] or len(history) != 5:
            raise AssertionError("All splits and the five-epoch observation are required")
        if result["selection"]["score"] != max(r["balanced_accuracy"] for r in history):
            raise AssertionError("Validation checkpoint selector failed")
        if not 0 <= result["test"]["balanced_accuracy"] <= 1:
            raise AssertionError("Selected checkpoint has no test measurement")
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        if records[0]["lr_used"] != [cfg["optimization"][name] for name in LR_KEYS]:
            raise AssertionError("First update did not use common base LR")
        initial = read(output / "initialization.json")["tensors"]
        expected = read(Path(base["runtime"]["output"]) / "initialization.json")["tensors"]
        if initial != expected:
            raise AssertionError("Initialization differs from the KNN37 baseline")
        saved = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["epoch"] != 5:
            raise AssertionError("Checkpoint roundtrip failed")
        del saved
        checks.append(dict(slug=slug, epochs=5, first_lr=records[0]["lr_used"],
                           initialization_matches=True, validation_selector=True, checkpoint_reload=True))
        print(slug + " CPU preflight passed", flush=True)
        gc.collect()
    write(campaign / "smoke_validation.json", dict(passed=True, manifest_sha256=digest(campaign / "manifest.json"),
          device="cpu", checks=checks, fixture="2 train and 8 val/test examples; not experiment results"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, type=Path)
    run(p.parse_args().campaign)
