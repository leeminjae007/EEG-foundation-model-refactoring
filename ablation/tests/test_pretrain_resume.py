from copy import deepcopy
import json

import pytest

from ablation.pretrain_resume import validate_resume, trim_partial_metrics


def fixture():
    cfg = {key: {} for key in ("ablation", "encoder", "position", "patch_encoder", "mae", "masking", "data")}
    cfg.update(seed=42, optimization={"epochs": 40})
    saved = dict(config=deepcopy(cfg), epoch=7, extra={"step": 700}, rng_states=[{}, {}, {}, {}])
    return cfg, saved


def test_recovery_preserves_world_size_schedule_and_architecture():
    cfg, saved = fixture()
    validate_resume(saved, cfg, 4)
    with pytest.raises(ValueError, match="world size"):
        validate_resume(saved, cfg, 2)
    for field in ("ablation", "masking", "optimization", "data"):
        changed = deepcopy(cfg)
        changed[field]["changed"] = True
        with pytest.raises(ValueError, match="config differs"):
            validate_resume(saved, changed, 4)
    saved["extra"]["partial_epoch_smoke"] = True
    with pytest.raises(ValueError, match="partial smoke"):
        validate_resume(saved, cfg, 4)


def test_interrupted_epoch_log_is_backed_up_and_trimmed(tmp_path):
    _, saved = fixture()
    path = tmp_path / "metrics.jsonl"
    committed = json.dumps(dict(epoch=7, step=700)) + "\n"
    uncommitted = json.dumps(dict(epoch=8, step=720)) + "\n" + '{"epoch":8'
    path.write_text(committed + uncommitted)
    assert trim_partial_metrics(tmp_path, saved, "job-1") == 2
    assert path.read_text() == committed
    assert (tmp_path / "interrupted-metrics-job-1.jsonl").read_text() == uncommitted
    assert trim_partial_metrics(tmp_path, saved, "job-1") == 0


def test_interior_corruption_does_not_silently_erase_training_history(tmp_path):
    _, saved = fixture()
    path = tmp_path / "metrics.jsonl"
    original = 'broken\n' + json.dumps(dict(epoch=7, step=700)) + '\n'
    path.write_text(original)
    with pytest.raises(ValueError, match="interior"):
        trim_partial_metrics(tmp_path, saved, "job-1")
    assert path.read_text() == original


def test_controller_cannot_change_other_jobs_or_gpu_types():
    from ablation.pretrain_queue import check_owned_pretrain
    fields = dict(JobId="27512241", UserId="ml10266(1234)", NumTasks="4",
                  ReqTRES="cpu=32,gres/gpu:a100=4", Partition="a100_short")
    check_owned_pretrain("27512241", fields)
    for change in [dict(JobId="27512901"), dict(Partition="gl40s_short"), dict(NumTasks="2")]:
        with pytest.raises(ValueError):
            check_owned_pretrain("27512241", dict(fields, **change))


def test_controller_requeues_only_timeout_or_near_limit():
    from datetime import datetime
    from ablation.pretrain_queue import continuation_needed
    clock = datetime(2026, 9, 16, 23, 57)
    job = dict(JobState="RUNNING", EndTime="2026-09-17T00:00:00")
    assert continuation_needed(job, clock)
    assert not continuation_needed(dict(job, EndTime="2026-09-17T01:00:00"), clock)
    for state in ["PENDING", "COMPLETED", "FAILED", "CANCELLED"]:
        assert not continuation_needed(dict(job, JobState=state), clock)
    assert continuation_needed(dict(job, JobState="TIMEOUT"), clock)
