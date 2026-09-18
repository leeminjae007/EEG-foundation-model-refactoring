"""14개 dataset의 세 split 로딩과 head, 세 task 유형의 학습 gradient를 확인한다."""

import gc
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch
from torch.utils.data import default_collate
import yaml

from scripts.convert_checkpoint import encoder_key
from src.data.datasets.registry import get_dataset_spec
from src.modules.loss import downstream_loss
from src.training.engine import build_finetune
from src.training.runtime import ROOT, set_paths

set_paths()
torch.set_num_threads(2)


@pytest.fixture(scope="module", autouse=True)
def export():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(ROOT / "tests/reference/export_downstream.py")],
                   cwd=ROOT / "tests/reference", env=env, check=True)


def test_datasets_and_models():
    paths = []
    for path in sorted((ROOT / "configs/downstream").glob("gr9-1_*_seed42.yaml")):
        if "lr5e6" not in path.name:
            paths.append(path)
    report = {}
    assert len(paths) == 13
    for path in paths:
        config = yaml.safe_load(path.read_text())
        name = config["data"]["dataset"]
        spec = get_dataset_spec(name)
        oracle = torch.load(ROOT / "outputs" / ("oracle-" + name + ".pth"), map_location="cpu")
        for split in ("train", "val", "test"):
            dataset = spec.dataset_class(config["data"]["dataset_dir"], split)
            dataset.enable_coordinate_only_channels()
            expected = oracle["splits"][split]
            assert len(dataset) == expected["count"]
            samples = [dataset[0], dataset[len(dataset) - 1]]
            for actual, reference in zip(samples, expected["samples"]):
                assert set(actual) == set(reference)
                for key, value in actual.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, reference[key], rtol=0, atol=0, msg=name + "/" + key)
                    else:
                        assert value == reference[key]
            del dataset
            gc.collect()
        samples = oracle["splits"]["train"]["samples"]
        if name == "isruc":
            samples = samples[:1]
        batch = default_collate(samples)
        torch.manual_seed(721)
        model = build_finetune(config)
        for key, value in model.head.state_dict().items():
            digest = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            assert digest == oracle["head_hashes"][key], name + "/" + key
        model.eval()
        with torch.no_grad():
            logits = model(batch["x"], batch["channel_coordinates"], batch["channel_validity"])
        torch.testing.assert_close(logits, oracle["logits"], rtol=1e-5, atol=2e-6, msg=name)
        item = {"eval_max_abs": float((logits - oracle["logits"]).abs().max()),
                "input_shape": list(batch["x"].shape), "head_initialization_exact": True,
                "all_three_splits_exact": True}
        if name in ("seed-v", "stress"):
            model.train()
            torch.set_rng_state(oracle["rng_before_train"])
            logits = model(batch["x"], batch["channel_coordinates"], batch["channel_validity"])
            loss = downstream_loss(spec.task, logits, batch["label"], config["optimization"])
            loss.backward()
            torch.testing.assert_close(logits, oracle["train_logits"], rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(loss, oracle["loss"], rtol=1e-6, atol=1e-7)
            assert torch.equal(torch.get_rng_state(), oracle["rng_after_train"])
            parameters = dict(model.named_parameters())
            maximum = 0.0
            for old, expected in oracle["gradients"].items():
                if old.startswith("classifier.encoder."):
                    key = "backbone." + encoder_key(old[len("classifier.encoder."):])
                else:
                    key = old.replace("classifier.head.", "head.", 1)
                actual = parameters[key].grad
                maximum = max(maximum, float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6, msg=name + "/" + key)
            item["gradient_max_abs"] = maximum
            item["loss_abs"] = float((loss - oracle["loss"]).abs())
            item["dropout_final_rng_equal"] = True
        report[name] = item
        del model, oracle
        gc.collect()
    (ROOT / "outputs/downstream-equivalence.json").write_text(json.dumps(report, indent=2))
