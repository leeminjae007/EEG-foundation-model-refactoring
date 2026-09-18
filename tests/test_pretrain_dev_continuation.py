"""Continuation of four-hour pretraining allocations, without GPU submissions."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import gr2_pretrain_campaign, mjde12_campaign


@pytest.fixture(params=[gr2_pretrain_campaign, mjde12_campaign])
def campaign(request, tmp_path, monkeypatch):
    module = request.param
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda *args: None))
    config = {"seed": 42, "optimization": {"epochs": 40}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "training"
    output.mkdir()
    entry = dict(index=0, kind="pretrain", arm="mjde_lite", seed=42,
                 config=str(config_path), result_dir=str(output), job="100", retries=4,
                 last_resume_epoch=4, max_timeout_resumes=40,
                 gpu_partitions="a100_dev,a100_short,a100_long", time_limit="04:00:00",
                 timeout_continuation=True)
    manifest = dict(root=str(tmp_path), pretrain_entries=[entry], submitted_entries=[],
                    downstream_submitted=True, pretrain_prepared=True,
                    results_dir=str(tmp_path / "results"))
    module.write_json(tmp_path / "manifest.json", manifest)
    return module, tmp_path, config, output


@pytest.mark.parametrize("state", ["TIMEOUT", "PREEMPTED", "COMPLETED"])
def test_incomplete_pretrain_continues_after_four_restarts(campaign, monkeypatch, state):
    module, folder, config, output = campaign
    torch.save(dict(config=config, epoch=8, model={}, optimizer={}, scheduler={},
                    rng_states=[{}] * 4), output / "last.pth")
    monkeypatch.setattr(module, "states", lambda entries: {"100": state, "200": "PENDING"})
    calls = []

    def submit(folder, manifest, kind, indices):
        calls.append((kind, indices))
        manifest["pretrain_entries"][0]["job"] = "200"
        module.write_json(folder / "manifest.json", manifest)

    monkeypatch.setattr(module, "submit", submit)
    module.monitor_step(folder)
    module.monitor_step(folder)
    saved = json.loads((folder / "manifest.json").read_text())["pretrain_entries"][0]
    assert calls == [("pretrain", [0])]
    assert saved["retries"] == 5 and saved["last_resume_epoch"] == 8
    assert saved["gpu_partitions"] == "a100_dev,a100_short,a100_long"
    assert saved["time_limit"] == "04:00:00"


@pytest.mark.parametrize("state", ["CANCELLED", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "PENDING"])
def test_no_blind_restart_or_duplicate_for_other_states(campaign, monkeypatch, state):
    module, folder, config, output = campaign
    torch.save(dict(config=config, epoch=8, model={}, optimizer={}, scheduler={},
                    rng_states=[{}] * 4), output / "last.pth")
    monkeypatch.setattr(module, "states", lambda entries: {"100": state})
    monkeypatch.setattr(module, "submit", lambda *args: pytest.fail("Unexpected GPU submission"))
    module.monitor_step(folder)


@pytest.mark.parametrize("epoch,match", [(4, "No completed-epoch progress"), (40, "Target epoch reached")])
def test_stalled_and_finished_checkpoints_are_not_resubmitted(campaign, epoch, match):
    module, folder, config, output = campaign
    entry = json.loads((folder / "manifest.json").read_text())["pretrain_entries"][0]
    saved = dict(config=config, epoch=epoch, model={}, optimizer={}, scheduler={}, rng_states=[{}] * 4)
    with pytest.raises(ValueError, match=match):
        module.timeout_resume_epoch(entry, config, saved)


def test_callback_waits_for_accounting_then_continues(campaign, monkeypatch):
    module, folder, _, _ = campaign
    states = iter(["UNKNOWN", "COMPLETING", "TIMEOUT"])
    monkeypatch.setattr(module, "states", lambda entries: {"100": next(states)})
    sleeps, calls = [], []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module, "monitor_step", lambda folder: calls.append(folder) or {"resumed": True})
    assert module.continue_after_job(folder, "100") == {"resumed": True}
    assert sleeps == [10, 10] and calls == [folder]


def test_old_callback_cannot_submit_again(campaign, monkeypatch):
    module, folder, _, _ = campaign
    monkeypatch.setattr(module, "monitor_step", lambda *args: pytest.fail("Superseded callback"))
    assert module.continue_after_job(folder, "old") == {"superseded_job": "old"}


def test_four_hour_submission_chains_cpu_callback(campaign, monkeypatch):
    module, folder, _, _ = campaign
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest.update(python="/venv/python", pretrain_launcher="slurm_flexible")
    commands = []
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda cmd, **kwargs: commands.append(cmd) or str(200 + len(commands)))
    module.submit(folder, manifest, "pretrain", [0])
    assert "--time=04:00:00" in commands[0]
    assert "--partition=a100_dev,a100_short,a100_long" in commands[0]
    assert "--nodes=1-4" in commands[0]
    assert "--dependency=afterany:201" in commands[1]
    assert not any("--gpu" in arg or "--gres" in arg for arg in commands[1])
    assert "continue --folder" in commands[1][-1] and "--job 201" in commands[1][-1]
    module.attach_continuation(folder, manifest, manifest["pretrain_entries"][0])
    assert len(commands) == 2
