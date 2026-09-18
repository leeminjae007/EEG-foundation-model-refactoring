"""실제 학습 진입점 산출물과 소량 validation 평가 경로를 확인한다."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.datasets.registry import get_dataset_spec
from src.model import PretrainModel
from src.training.engine import build_finetune, evaluate
from src.training.runtime import ROOT

torch.set_num_threads(2)


def test_smoke_checkpoints_load_strictly():
    report = {}
    for name in ("pretrain", "seed-v", "stress"):
        path = ROOT / "outputs/smoke" / name / "last.pth"
        checkpoint = torch.load(path, map_location="cpu")
        if name == "pretrain":
            model = PretrainModel(checkpoint["config"], torch.device("cpu"))
        else:
            model = build_finetune(checkpoint["config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        assert checkpoint["extra"]["step"] == 1
        assert checkpoint["extra"]["partial_epoch_smoke"]
        assert len(checkpoint["optimizer"]["state"]) > 0
        for tensor in model.state_dict().values():
            assert torch.isfinite(tensor).all()
        rows = (path.parent / "metrics.jsonl").read_text().splitlines()
        values = json.loads(rows[-1])
        assert values["loss"] >= 0
        report[name] = {"loss": values["loss"], "pre_clip_norm": values["pre_clip_norm"],
                        "lr_used": values["lr_used"], "strict_load": True,
                        "optimizer_state_present": True, "batch_size": 2, "updates": 1}
    (ROOT / "outputs/smoke-summary.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("task_name", ["seedv", "mentalarithmetic"])
def test_small_validation_path(task_name):
    path = ROOT / "configs/downstream" / ("gr9-1_" + task_name + "_seed42.yaml")
    config = yaml.safe_load(path.read_text())
    name = config["data"]["dataset"]
    spec = get_dataset_spec(name)
    dataset = spec.dataset_class(config["data"]["dataset_dir"], "val")
    dataset.enable_coordinate_only_channels()
    indices = list(range(4)) + list(range(len(dataset) - 4, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=2, num_workers=0)
    model = build_finetune(config)
    result = evaluate(model, loader, spec.task, name, torch.device("cpu"), 1)
    assert result["diagnostics"]["probe"]["samples"] == 8
    assert "subject" in result["diagnostics"]["probe"]
    if spec.task == "regression":
        assert "r2" in result
    else:
        assert "balanced_accuracy" in result
        assert result["diagnostics"]["probe"]["person_class_stratified"]
    # 소량 검증 수치는 과학적 test 결과와 분리한다.
    (ROOT / "outputs" / ("validation-smoke-" + name + ".json")).write_text(json.dumps(result, indent=2))
