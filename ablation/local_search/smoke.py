"""CPU smoke on real train/validation samples for each dataset; no test access."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace

from ablation.local_search.campaign import verify, digest, read, write_json, load_plan


def run(campaign):
    m = verify(campaign)
    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from torch.utils.data import Subset
    from src.training import engine
    torch.set_num_threads(2)
    checks = []
    for slug, ds in m["datasets"].items():
        for stage in ("screen",):
            load_plan(campaign, slug, stage)
        base = next(e for e in ds["baseline"] if e["seed"] == ds["start_seed"])
        cfg = yaml.safe_load(Path(base["config"]).read_text())
        cfg = deepcopy(cfg)
        cfg["optimization"].update(epochs=1, batch_size_per_gpu=1, gradient_accumulation_steps=1)
        cfg["data"]["num_workers"] = 0
        output = campaign / "smoke" / slug
        output.mkdir(parents=True, exist_ok=False)
        cfg["runtime"].update(output=str(output), evaluate_test=False, log_every_steps=1)
        original_spec = engine.get_dataset_spec
        spec = original_spec(cfg["data"]["dataset"])
        opened = []

        class Fixture(Subset):
            def enable_coordinate_only_channels(self):
                self.dataset.enable_coordinate_only_channels()

        def train_val_only(path, split):
            if split == "test":
                raise AssertionError("Test dataset opened during search")
            dataset = spec.dataset_class(path, split)
            opened.append(split)
            # A spread of validation samples exercises binary and multiclass metrics.
            n = 2 if split == "train" else 8
            return Fixture(dataset, [round(i * (len(dataset) - 1) / (n - 1)) for i in range(n)])

        engine.get_dataset_spec = lambda name: SimpleNamespace(**dict(vars(spec), dataset_class=train_val_only))
        try:
            engine.run_finetune(cfg, SimpleNamespace(device="cpu", distributed=False, smoke=False, resume=None))
        finally:
            engine.get_dataset_spec = original_spec
        result = read(output / "result.json")
        if opened != ["train", "val"] or any(v["test"] is not None for v in result.values()):
            raise AssertionError("Test gate failed")
        init = read(output / "initialization.json")["tensors"]
        baseline_init = read(Path(base["output"]) / "initialization.json")["tensors"]
        if init != baseline_init:
            raise AssertionError("Initialization differs from baseline: " + slug)
        import json
        records = [json.loads(l) for l in (output / "metrics.jsonl").read_text().splitlines()]
        if records[0]["lr_used"] != [ds["anchor"]["lr"]] * 3:
            raise AssertionError("First update did not use the common base LR")
        saved = torch.load(output / "last.pth", map_location="cpu")
        if saved["config"] != cfg:
            raise AssertionError("Checkpoint/config roundtrip mismatch")
        del saved
        checks.append(dict(slug=slug, opened_splits=opened, initialization_matches=True,
                           first_lr=records[0]["lr_used"], result_readable=True))
        print(slug + " smoke passed", flush=True)
        gc.collect()
    report = dict(passed=True, manifest_sha256=digest(campaign / "manifest.json"), checks=checks,
                  device="cpu", test_data_read=False)
    write_json(campaign / "smoke_validation.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, type=Path)
    run(p.parse_args().campaign)
