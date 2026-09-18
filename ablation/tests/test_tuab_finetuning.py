"""Verify optimizer state, DDP freezing, resume and unchanged baseline behavior."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest
import torch
from torch import nn
import yaml

from ablation.tuab.campaign import ARMS, make_config
from ablation.tuab.finetune import validate_resume
from ablation.tuab.policy import FineTunePolicy, PerGroupCosineScheduler
from src.training import engine
from src.training.scheduler import GroupCosineScheduler


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = nn.Linear(4, 4)
        self.position = nn.Linear(4, 4)
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Dropout(.2))

    def forward(self, x, coordinates=None, validity=None):
        return self.encoder(self.position(self.tokenizer(x))).reshape(-1, 1, 1, 4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(.1), nn.Linear(4, 1))

    def forward(self, x, coordinates=None, validity=None):
        return self.head(self.backbone(x, coordinates, validity)).flatten()


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, directory, split):
        self.x = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 20

    def __len__(self):
        return len(self.x)

    def enable_coordinate_only_channels(self):
        pass

    def __getitem__(self, index):
        return {"x": self.x[index], "label": torch.tensor(index % 2),
                "channel_coordinates": torch.zeros(1, 3), "channel_validity": torch.ones(1, dtype=torch.bool),
                "subject_id": "subject%d" % (index // 4)}


def optimizer(model):
    return torch.optim.AdamW(engine.optimizer_groups(model, {
        "tokenizer_learning_rate": 1e-6, "encoder_learning_rate": 1e-6, "head_learning_rate": 1e-5}),
        weight_decay=5e-5)


def test_group_floors_keep_tenfold_ratio_and_resume():
    opt = optimizer(TinyModel())
    floors = {"tokenizer": 1e-7, "encoder": 1e-7, "head": 1e-6}
    schedule = PerGroupCosineScheduler(opt, 20, 1e-6, floors)
    sequence = []
    for index in range(20):
        sequence.append(schedule.step())
        if index == 7:
            saved = deepcopy(schedule.state_dict())
    assert sequence[0] == [1e-6, 1e-6, 1e-5]
    assert sequence[-1] == [1e-7, 1e-7, 1e-6]
    assert all(row[0] == row[1] and row[2] / row[0] == pytest.approx(10) for row in sequence)
    resumed = PerGroupCosineScheduler(optimizer(TinyModel()), 20, 1e-6, floors)
    resumed.load_state_dict(saved)
    assert [resumed.step() for _ in range(12)] == sequence[8:]
    wrong = PerGroupCosineScheduler(optimizer(TinyModel()), 21, 1e-6, floors)
    with pytest.raises(ValueError, match="total_steps"):
        wrong.load_state_dict(saved)


def test_default_policy_delegates_original_scheduler():
    opt = optimizer(TinyModel())
    schedule = FineTunePolicy({}).make_scheduler(opt, 60, 1e-6)
    assert type(schedule) is GroupCosineScheduler


@pytest.mark.parametrize("ddp", [False, True])
def test_head_first_freezes_weights_and_adam_state_then_unfreezes(tmp_path, ddp):
    store_path = Path("outputs").resolve() / ("tuab-ddp-" + uuid.uuid4().hex)
    if ddp:
        store_path.parent.mkdir(exist_ok=True)
        torch.distributed.init_process_group("gloo", store=torch.distributed.FileStore(str(store_path), 1),
                                            rank=0, world_size=1)
    try:
        torch.manual_seed(10)
        model = TinyModel()
        opt = optimizer(model)
        trainer = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) if ddp else model
        policy = FineTunePolicy({"head_first_epochs": 2})
        original = deepcopy(model.state_dict())
        for epoch in range(4):
            trainer.train()
            policy.on_epoch_start(model, epoch)
            assert model.backbone.training == (epoch >= 2)
            assert model.head.training
            # Two updates per epoch also check that the next DDP reduction can start.
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                trainer(torch.ones(2, 4)).square().mean().backward()
                opt.step()
            if epoch < 2:
                assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items() if k.startswith("backbone."))
                assert all(p not in opt.state for p in model.backbone.parameters())
            else:
                assert any(not torch.equal(v, original[k]) for k, v in model.state_dict().items() if k.startswith("backbone."))
                assert all(p in opt.state for p in model.backbone.parameters())
        assert all(int(opt.state[p]["step"]) == 8 for p in model.head.parameters())
        assert all(int(opt.state[p]["step"]) == 4 for p in model.backbone.parameters())
    finally:
        if ddp:
            torch.distributed.destroy_process_group()
            store_path.unlink(missing_ok=True)


def setup_engine(monkeypatch, output, arm):
    monkeypatch.setattr(engine, "build_finetune", lambda config: TinyModel())
    monkeypatch.setattr(engine, "get_dataset_spec", lambda name: SimpleNamespace(task="binary", dataset_class=TinyDataset))
    base = yaml.safe_load(Path("configs/downstream/gr9-1_tuab_seed42.yaml").read_text())
    config = make_config(base, arm, Path("unused.pth"), output)
    config["data"]["num_workers"] = 0
    config["optimization"].update(epochs=4, batch_size_per_gpu=2, gradient_accumulation_steps=2)
    config["runtime"]["log_every_steps"] = 1
    args = SimpleNamespace(device="cpu", distributed=False, smoke=False, resume=None)
    return config, args


@pytest.mark.parametrize("stop_epoch", [1, 2, 3])
def test_real_engine_resume_across_freeze_boundary(tmp_path, monkeypatch, stop_epoch):
    config, args = setup_engine(monkeypatch, tmp_path / "full", ARMS[2])
    engine.run_finetune(config, args, FineTunePolicy(config["finetune_ablation"]))
    expected = torch.load(tmp_path / "full/last.pth", weights_only=False)
    config["runtime"]["output"] = str(tmp_path / "resume")
    original_save = engine.save_checkpoint

    def interrupt(*values, **kwargs):
        original_save(*values, **kwargs)
        if values[4] == stop_epoch:
            raise RuntimeError("interruption")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "save_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            engine.run_finetune(config, args, FineTunePolicy(config["finetune_ablation"]))
    args.resume = str(tmp_path / "resume/last.pth")
    validate_resume(config, torch.load(args.resume, weights_only=False))
    engine.run_finetune(config, args, FineTunePolicy(config["finetune_ablation"]))
    actual = torch.load(args.resume, weights_only=False)
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["extra"] == expected["extra"]
    for name, value in actual["model"].items():
        torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)
    assert json.loads((tmp_path / "resume/result.json").read_text()) == json.loads((tmp_path / "full/result.json").read_text())
    assert set(json.loads((tmp_path / "full/result.json").read_text())) == {"balanced_accuracy", "auroc"}


def test_default_policy_is_bit_identical(tmp_path, monkeypatch):
    config, args = setup_engine(monkeypatch, tmp_path / "default", "baseline")
    engine.run_finetune(config, args)
    for name, policy in (("policy", FineTunePolicy({})),):
        config["runtime"]["output"] = str(tmp_path / name)
        engine.run_finetune(config, args, policy)
        original = torch.load(tmp_path / "default/last.pth", weights_only=False)
        actual = torch.load(tmp_path / name / "last.pth", weights_only=False)
        assert actual["extra"] == original["extra"]
        for key, value in actual["model"].items():
            torch.testing.assert_close(value, original["model"][key], rtol=0, atol=0)
        assert (tmp_path / name / "metrics.jsonl").read_bytes() == (tmp_path / "default/metrics.jsonl").read_bytes()
        assert (tmp_path / name / "result.json").read_bytes() == (tmp_path / "default/result.json").read_bytes()


def test_arms_change_only_declared_scientific_settings(tmp_path):
    base = yaml.safe_load(Path("configs/downstream/gr9-1_tuab_seed42.yaml").read_text())
    for arm in ARMS:
        config = make_config(base, arm, "weight.pth", tmp_path / arm)
        restored = deepcopy(config)
        del restored["finetune_ablation"]
        restored["model"]["checkpoint"] = base["model"]["checkpoint"]
        restored["runtime"]["output"] = base["runtime"]["output"]
        if arm in ARMS[1:3]:
            for key in ("tokenizer_learning_rate", "encoder_learning_rate"):
                restored["optimization"][key] = base["optimization"][key]
        if arm == "head_h4":
            restored["model"]["head_hidden_tokens"] = None
        assert restored == base
