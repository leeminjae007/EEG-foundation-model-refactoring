"""Guard against duplicate GPU submissions and changed experiment snapshots."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from ablation.tuab import campaign as module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    checkpoint = root / module.CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake fixture checkpoint, never used for training")
    monkeypatch.setattr(module, "CHECKPOINT_SHA256", module.digest(checkpoint))
    (root / "src/data/datasets").mkdir(parents=True)
    (root / "src/data/datasets/registry.py").write_text("# fixture\n")
    (root / "src/training").mkdir()
    (root / "src/training/engine.py").write_text(Path("src/training/engine.py").read_text(encoding="utf-8"), encoding="utf-8")
    (root / "ablation/tuab").mkdir(parents=True)
    (root / "ablation/__init__.py").write_text("")
    (root / "ablation/bootstrap.py").write_text("# fixture\n")
    (root / "ablation/tuab/__init__.py").write_text("")
    (root / "configs/downstream").mkdir(parents=True)
    base = yaml.safe_load(Path("configs/downstream/gr9-1_tuab_seed42.yaml").read_text())
    for seed in module.SEEDS:
        canonical = deepcopy(base)
        canonical["seed"] = seed
        (root / "configs/downstream" / ("gr9-1_tuab_seed%d.yaml" % seed)).write_text(yaml.safe_dump(canonical))
        actual = deepcopy(canonical)
        actual["model"]["checkpoint"] = str(checkpoint)
        path = root / module.BASELINE / ("tuab_seed" + str(seed))
        path.mkdir(parents=True)
        (path / "resolved_config.yaml").write_text(yaml.safe_dump(actual))
        (path / "result.json").write_text(json.dumps({"balanced_accuracy": {}}))
    destination = root / module.DEFAULT_CAMPAIGN
    module.prepare(destination)
    # pytest's Windows cleanup needs permission to remove the fixture snapshot.
    for p in (destination / "source").rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda *args: None))
    module.write_json(destination / "smoke_validation.json", {
        "passed": True, "manifest_sha256": module.digest(destination / "manifest.json")})
    # Windows has no Path.owner; this fixture supplies the Slurm user for the mocked scheduler.
    monkeypatch.setattr(type(root), "owner", lambda self: "fixture-user")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "")
    return destination


def test_prepare_keeps_fifteen_unique_runs_and_reuses_snapshot(prepared):
    manifest = module.verify_source(prepared)
    entries = manifest["entries"]
    assert len(entries) == 15
    assert len({e["output"] for e in entries}) == 15
    assert {(e["arm"], e["seed"]) for e in entries} == {(a, s) for a in module.RUN_ARMS for s in module.SEEDS}
    original = (prepared / "manifest.json").read_bytes()
    module.prepare(prepared)
    assert (prepared / "manifest.json").read_bytes() == original


def test_successful_submission_is_idempotent(prepared, monkeypatch):
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        assert (prepared / "submission_intent.json").exists()
        return SimpleNamespace(returncode=0, stdout="123456;fixture-cluster\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", sbatch)
    module.submit(prepared)
    module.submit(prepared)
    assert len(calls) == 2
    record = json.loads((prepared / "submission.json").read_text())
    assert record["job_id"] == "123456" and record["tasks"] == 15


@pytest.mark.parametrize("response", [SimpleNamespace(returncode=1, stdout="", stderr="QOS limit"),
                                      SimpleNamespace(returncode=0, stdout="ambiguous reply", stderr="")])
def test_ambiguous_or_rejected_submission_never_retries_blindly(prepared, monkeypatch, response):
    calls = []

    def sbatch(*args, **kwargs):
        calls.append(args)
        return response

    monkeypatch.setattr(module.subprocess, "run", sbatch)
    with pytest.raises(RuntimeError):
        module.submit(prepared)
    with pytest.raises(RuntimeError, match="reconcile"):
        module.submit(prepared)
    assert len(calls) == 1
    assert not (prepared / "submission.json").exists()


def test_changed_snapshot_blocks_submission(prepared, monkeypatch):
    manifest = module.verify_source(prepared)
    Path(manifest["entries"][0]["config"]).write_text("changed")
    with pytest.raises(RuntimeError, match="Snapshot changed"):
        module.submit(prepared)
    assert not (prepared / "submission_intent.json").exists()
