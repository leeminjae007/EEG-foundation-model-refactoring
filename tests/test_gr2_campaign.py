import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import gr2_pretrain_campaign as campaign


def setup(tmp_path, monkeypatch, state):
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda *args: None))
    config = {"seed": 42, "optimization": {"epochs": 40}}
    cp = tmp_path / "config.yaml"
    cp.write_text(yaml.safe_dump(config))
    out = tmp_path / "pretrain"
    out.mkdir()
    entry = dict(index=0, kind="pretrain", arm="mjde-geometry", config=str(cp),
                 result_dir=str(out), job="100", retries=0)
    manifest = dict(pretrain_entries=[entry], results_dir=str(tmp_path / "results"))
    campaign.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": state})
    return config, out


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PENDING"])
def test_failure_or_pending_does_not_submit_gpu(tmp_path, monkeypatch, state):
    setup(tmp_path, monkeypatch, state)
    monkeypatch.setattr(campaign, "submit", lambda *args: pytest.fail("Must not resubmit this state"))
    result = campaign.monitor_step(tmp_path)
    assert not result["complete"] and not result["training_started"]


def test_timeout_resumes_progress_once(tmp_path, monkeypatch):
    config, out = setup(tmp_path, monkeypatch, "TIMEOUT")
    torch.save(dict(config=config, epoch=2, model={}, optimizer={}, scheduler={}, rng_states=[{}]*4), out / "last.pth")
    calls = []
    def submit(folder, manifest, kind, indices):
        calls.append((kind, indices))
        manifest["pretrain_entries"][0]["job"] = "200"
    monkeypatch.setattr(campaign, "submit", submit)
    assert campaign.monitor_step(tmp_path)["state"] == "RESUBMITTED"
    campaign.monitor_step(tmp_path)
    assert calls == [("pretrain", [0])]


def test_training_needs_probes_independent_rng_and_new_update(tmp_path, monkeypatch):
    _, out = setup(tmp_path, monkeypatch, "RUNNING")
    for rank in range(4):
        campaign.write_json(out / ("gpu-health-100-rank%d.json" % rank), dict(cuda_probe="passed"))
        campaign.write_json(out / ("rng-start-100-rank%d.json" % rank), dict(cuda_sha256=str(rank), start_step=100))
    (out / "metrics.jsonl").write_text(json.dumps(dict(step=100, epoch=2, loss=.5)) + "\n")
    assert not campaign.monitor_step(tmp_path)["training_started"]
    (out / "metrics.jsonl").write_text(json.dumps(dict(step=120, epoch=3, loss=.4)) + "\n")
    assert campaign.monitor_step(tmp_path)["training_started"]


def test_four_gpus_flexible_nodes_and_no_downstream(tmp_path, monkeypatch):
    manifest = dict(pretrain_launcher="slurm_flexible", pretrain_entries=[dict(arm="mjde-geometry")])
    commands = []
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda cmd, **kw: commands.append(cmd) or "100\n")
    campaign.submit(tmp_path, manifest, "pretrain", [0])
    for flag in ("--nodes=1-4", "--ntasks=4", "--gpus-per-task=a100:1", "--cpus-per-task=4", "--mem-per-gpu=32G"):
        assert flag in commands[0]
    with pytest.raises(ValueError, match="only submits pretraining"):
        campaign.submit(tmp_path, manifest, "downstream", [0])
