import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import mjde12_campaign as campaign


@pytest.fixture
def setup_campaign(tmp_path, monkeypatch):
    # File locking is independently supplied by Linux on the actual controller.
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda *args: None))
    pre = tmp_path / "pretrain/mjde"
    down = tmp_path / "downstream/chb_seed42"
    pre.mkdir(parents=True)
    down.mkdir(parents=True)
    cfg = {"seed": 42, "optimization": {"epochs": 40}, "ablation": {"protocol": "context_blocks_v1"}}
    pc = tmp_path / "pretrain.yaml"
    dc = tmp_path / "downstream.yaml"
    pc.write_text(yaml.safe_dump(cfg))
    dc.write_text(yaml.safe_dump(cfg))
    entries = [dict(index=0, kind="downstream", slug="chb", seed=42, config=str(dc),
                    result_dir=str(down), retries=0)]
    pretrains = [dict(index=0, kind="pretrain", arm="mjde", seed=42, config=str(pc),
                      result_dir=str(pre), job="100", retries=0)]
    manifest = dict(root=str(tmp_path), submitted_entries=entries, reused_entries=[],
                    pretrain_entries=pretrains, pretrain_prepared=True, downstream_submitted=False,
                    checkpoint=str(pre / "checkpoint-epoch-0040.pth"), checkpoint_sha256=None)
    campaign.write_json(tmp_path / "manifest.json", manifest)
    calls = []

    def submit(folder, state, kind, indices=None):
        calls.append((kind, indices))
        for entry in state["submitted_entries" if kind == "downstream" else "pretrain_entries"]:
            entry["job"] = "200_0" if kind == "downstream" else "201"
        if kind == "downstream":
            state["downstream_submitted"] = True
        campaign.write_json(folder / "manifest.json", state)
        return "200"

    monkeypatch.setattr(campaign, "submit", submit)
    return tmp_path, manifest, cfg, calls


def test_no_downstream_before_new_pretrain_completes(setup_campaign, monkeypatch):
    folder, manifest, cfg, calls = setup_campaign
    torch.save({"epoch": 40, "config": cfg}, manifest["checkpoint"])
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "RUNNING"})
    state = campaign.monitor_step(folder)
    assert calls == []
    assert state["rows"][0]["state"] == "WAITING_PRETRAIN"
    assert not state["complete"]


@pytest.mark.parametrize("bad", ["epoch", "config", "smoke"])
def test_completed_job_needs_verified_checkpoint(setup_campaign, monkeypatch, bad):
    folder, manifest, cfg, calls = setup_campaign
    saved = {"epoch": 40, "config": cfg}
    if bad == "epoch":
        saved["epoch"] = 39
    elif bad == "config":
        saved["config"] = {"wrong": True}
    else:
        saved["extra"] = {"partial_epoch_smoke": True}
    torch.save(saved, manifest["checkpoint"])
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "COMPLETED"})
    with pytest.raises(ValueError, match="downstream gate"):
        campaign.monitor_step(folder)
    assert calls == []


def test_verified_new_checkpoint_submits_downstream_once(setup_campaign, monkeypatch):
    folder, manifest, cfg, calls = setup_campaign
    torch.save({"epoch": 40, "config": cfg}, manifest["checkpoint"])
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "COMPLETED", "200_0": "PENDING"})
    campaign.monitor_step(folder)
    campaign.monitor_step(folder)
    assert calls == [("downstream", None)]
    saved = json.loads((folder / "manifest.json").read_text())
    assert saved["checkpoint_sha256"] == campaign.digest(manifest["checkpoint"])
    assert saved["downstream_submitted"]


def test_timeout_resume_requires_identical_config(setup_campaign, monkeypatch):
    folder, manifest, cfg, calls = setup_campaign
    manifest["downstream_submitted"] = True
    manifest["submitted_entries"][0]["job"] = "300_0"
    campaign.write_json(folder / "manifest.json", manifest)
    path = folder / "downstream/chb_seed42/last.pth"
    torch.save({"epoch": 2, "config": {"wrong": 1}}, path)
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "RUNNING", "300_0": "TIMEOUT"})
    with pytest.raises(ValueError, match="Unsafe timeout resume"):
        campaign.monitor_step(folder)
    assert not calls
    torch.save({"epoch": 2, "config": cfg}, path)
    campaign.monitor_step(folder)
    assert calls == [("downstream", [0])]


def reference_manifest(folder, manifest, cfg):
    cfg["ablation"]["encoder"] = "mjde"
    path = folder / "reference_full.yaml"
    path.write_text(yaml.safe_dump(cfg))
    torch.save({"epoch": 40, "config": cfg}, manifest["checkpoint"])
    manifest.update(reuse_mjde_reference=True, reference_checkpoint=manifest["checkpoint"],
                    reference_sha256=campaign.digest(manifest["checkpoint"]),
                    reference_full_config=str(path), downstream_runner="base")
    manifest["pretrain_entries"][0]["arm"] = "mjde_lite"
    campaign.write_json(folder / "manifest.json", manifest)


def test_reference_full_submits_without_waiting_for_lite(setup_campaign, monkeypatch):
    folder, manifest, cfg, calls = setup_campaign
    reference_manifest(folder, manifest, cfg)
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "PENDING", "200_0": "PENDING"})
    campaign.monitor_step(folder)
    campaign.monitor_step(folder)
    assert calls == [("downstream", None)]
    saved = json.loads((folder / "manifest.json").read_text())
    assert saved["checkpoint_sha256"] == manifest["reference_sha256"]


@pytest.mark.parametrize("bad", ["hash", "configuration", "epoch", "lite"])
def test_reference_reuse_rejects_changed_pretrain(setup_campaign, monkeypatch, bad):
    folder, manifest, cfg, calls = setup_campaign
    reference_manifest(folder, manifest, cfg)
    saved = {"epoch": 40, "config": cfg}
    if bad == "epoch":
        saved["epoch"] = 39
    elif bad == "configuration":
        saved["config"]["optimization"]["epochs"] = 41
    elif bad == "lite":
        saved["config"]["ablation"]["encoder"] = "mjde_lite"
    torch.save(saved, manifest["checkpoint"])
    manifest["reference_sha256"] = "changed" if bad == "hash" else campaign.digest(manifest["checkpoint"])
    campaign.write_json(folder / "manifest.json", manifest)
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "PENDING"})
    with pytest.raises(ValueError, match="Reference"):
        campaign.monitor_step(folder)
    assert not calls


@pytest.mark.parametrize("local_ids", [(0, 1, 2, 3), (0, 1, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)])
def test_flexible_ranks_ignore_node_local_gpu_indices(local_ids):
    for rank, local_id in enumerate(local_ids):
        env = dict(SLURM_PROCID=str(rank), SLURM_LOCALID=str(local_id), SLURM_NTASKS="4",
                   MASTER_ADDR="gpu-a", MASTER_PORT="12345", CUDA_VISIBLE_DEVICES=str(local_id + 1))
        assert campaign.rank_environment(env) == dict(RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK="0")


@pytest.mark.parametrize("visible", ["", "0,1", "-1"])
def test_flexible_rank_rejects_unbound_gpu(visible):
    with pytest.raises(ValueError, match="one bound GPU"):
        campaign.rank_environment(dict(CUDA_VISIBLE_DEVICES=visible))


def test_flexible_pretrain_requests_four_total_gpus(tmp_path, monkeypatch):
    manifest = dict(pretrain_launcher="slurm_flexible", pretrain_entries=[dict(arm="mjde_lite")])
    calls = []
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda cmd, **kw: calls.append(cmd) or "100\n")
    campaign.submit(tmp_path, manifest, "pretrain", [0])
    command = calls[0]
    for value in ("--nodes=1-4", "--ntasks=4", "--gpus-per-task=a100:1", "--cpus-per-task=4", "--mem-per-gpu=32G"):
        assert value in command
    assert not any(value.startswith("--gres=") or value == "--nodes=1" for value in command)


def test_flexible_worker_uses_srun_and_shared_rendezvous(tmp_path, monkeypatch):
    config = tmp_path / "pretrain.yaml"
    config.write_text("seed: 42\n")
    entry = dict(config=str(config), config_sha256=campaign.digest(config), source=str(tmp_path),
                 result_dir=str(tmp_path / "out"))
    manifest = dict(pretrain_entries=[entry], pretrain_launcher="slurm_flexible", python="/venv/python")
    env = dict(MJ12_KIND="pretrain", MJ12_INDEX="0", SLURM_JOB_NODELIST="gpu-[a-c]",
               SLURM_JOB_NUM_NODES="3", SLURM_JOB_ID="1234")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(campaign.os, "chdir", lambda path: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "gpu-a\ngpu-b\ngpu-c\n")
    commands = []
    monkeypatch.setattr(campaign.os, "execvp", lambda program, cmd: commands.append(cmd))
    campaign.worker(tmp_path, manifest)
    assert campaign.os.environ["MASTER_ADDR"] == "gpu-a"
    assert commands[0][0] == "srun"
    assert "--nodes=3" in commands[0] and "--ntasks=4" in commands[0]
    assert "rank-worker" in commands[0] and "torch.distributed.run" not in commands[0]


def test_allocated_gpu_uuid_replaces_ambiguous_ordinal():
    assert campaign.bound_gpu_environment("GPU-aabbcc\n") == dict(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="GPU-aabbcc")


@pytest.mark.parametrize("output", ["", "0", "GPU-a\nGPU-b\n", "GPU-a,GPU-b"])
def test_gpu_binding_refuses_multiple_or_unidentified_devices(output):
    with pytest.raises(ValueError, match="refusing ambiguous binding"):
        campaign.bound_gpu_environment(output)


@pytest.mark.parametrize("action", ["None", "Reset", "Reboot", "Drain P2P", "Drain and Reset"])
def test_gpu_recovery_status_is_read_from_driver_report(action):
    report = "GPU 0000:65:00.0\n    GPU Recovery Action                  : " + action + "\n"
    assert campaign.gpu_recovery_action(report) == action


def test_missing_recovery_status_is_not_reported_healthy():
    assert campaign.gpu_recovery_action("nvidia-smi failed") == "unavailable"


def test_reset_required_gpu_saves_kernel_evidence_before_training(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n")
    entry = dict(config=str(config), config_sha256=campaign.digest(config),
                 result_dir=str(tmp_path), source=str(tmp_path))
    env = dict(SLURM_PROCID="2", SLURM_NTASKS="4", MASTER_ADDR="gpu-a", MASTER_PORT="12345",
               CUDA_VISIBLE_DEVICES="0", SLURMD_NODENAME="gpu-b", SLURM_JOB_ID="123", MJ12_INDEX="0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(campaign.os, "chdir", lambda path: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "GPU-test\n")
    def run(command, **kwargs):
        output = "GPU Recovery Action : Reset\n" if command[0] == "nvidia-smi" else "NVRM: Xid: 74 NVLink fatal\n"
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(campaign.subprocess, "run", run)
    monkeypatch.setattr(campaign.os, "execv", lambda *a: pytest.fail("Training must not start on a reset-required GPU"))
    with pytest.raises(RuntimeError, match="requires recovery: Reset"):
        campaign.rank_worker(tmp_path, dict(pretrain_entries=[entry]))
    report = json.loads((tmp_path / "gpu-health-123-rank2.json").read_text())
    assert report["cuda_probe"] == "failed" and report["node"] == "gpu-b"
    assert report["gpu_uuid"] == "GPU-test"
    assert "Xid: 74" in report["kernel_nvrm"][0]


def test_dev_submission_keeps_all_partitions_and_adds_cpu_continuation(tmp_path, monkeypatch):
    entry = dict(arm="cbramod", gpu_partitions="a100_dev,a100_short,a100_long", time_limit="04:00:00",
                 timeout_continuation=True)
    manifest = dict(python="/venv/python", pretrain_launcher="slurm_flexible", pretrain_entries=[entry])
    calls = []
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda cmd, **kw: calls.append(cmd) or str(100 + len(calls)))
    campaign.submit(tmp_path, manifest, "pretrain", [0])
    assert "--partition=a100_dev,a100_short,a100_long" in calls[0]
    assert "--time=04:00:00" in calls[0]
    assert "--dependency=afterany:101" in calls[1]
    assert "--partition=cpu_short,cpu_long" in calls[1]
    assert not any("--gpu" in arg or "--gres" in arg for arg in calls[1])
    assert entry["continuation_for_job"] == "101"
    campaign.attach_continuation(tmp_path, manifest, entry)
    assert len(calls) == 2


def test_pretrain_timeout_restores_all_state_and_requires_epoch_progress():
    entry = dict(kind="pretrain", result_dir="run", last_resume_epoch=3)
    config = dict(optimization=dict(epochs=40))
    saved = dict(config=config, epoch=7, model={}, optimizer={}, scheduler={}, rng_states=[{}, {}, {}, {}])
    assert campaign.timeout_resume_epoch(entry, config, saved) == 7
    saved["epoch"] = 3
    with pytest.raises(ValueError, match="No completed-epoch progress"):
        campaign.timeout_resume_epoch(entry, config, saved)
    saved["epoch"] = 7
    saved["rng_states"] = [{}]
    with pytest.raises(ValueError, match="four RNG"):
        campaign.timeout_resume_epoch(entry, config, saved)


def test_dev_timeout_can_continue_beyond_two_resumes(setup_campaign, monkeypatch):
    folder, manifest, cfg, calls = setup_campaign
    manifest["downstream_submitted"] = True
    entry = manifest["pretrain_entries"][0]
    entry.update(retries=3, max_timeout_resumes=40, last_resume_epoch=8)
    torch.save(dict(config=cfg, epoch=12, model={}, optimizer={}, scheduler={}, rng_states=[{}, {}, {}, {}]),
               folder / "pretrain/mjde/last.pth")
    campaign.write_json(folder / "manifest.json", manifest)
    monkeypatch.setattr(campaign, "states", lambda entries: {"100": "TIMEOUT"})
    campaign.monitor_step(folder)
    assert calls == [("pretrain", [0])]
    state = json.loads((folder / "manifest.json").read_text())
    assert state["pretrain_entries"][0]["retries"] == 4
    assert state["pretrain_entries"][0]["last_resume_epoch"] == 12
