import pytest

from ablation.optuna_search import campaign as c


def control(tmp_path):
    value = dict(action="fix_existing_trial", trial=0, hp=dict(lr=1e-4, wd=.01, dropout=.2),
                 reason="User stopped this dataset", reporting_note="Historical test-informed result")
    c.write(tmp_path / "dataset_controls.json", dict(version=1, datasets={"mentalarithmetic": value}))
    return value


@pytest.mark.parametrize("indices", [None, [0, 2]])
def test_fixed_dataset_blocks_both_initial_and_retry_submission(tmp_path, monkeypatch, indices):
    control(tmp_path)
    monkeypatch.setattr(c, "submit_once", lambda *args: pytest.fail("Must not submit a GPU job"))
    with pytest.raises(RuntimeError, match="disabled new trials and retries"):
        c.submit_trial(tmp_path, {}, "mentalarithmetic", {}, indices)


def test_fixed_trial_validates_five_seeds_without_opening_study(tmp_path, monkeypatch):
    fixed = control(tmp_path)
    directory = tmp_path / "datasets/mentalarithmetic/trial-000"
    rows = [dict(seed=s, test={"balanced_accuracy": .7}) for s in c.SEEDS]
    plan = dict(number=0, slug="mentalarithmetic", hp=fixed["hp"], entries=rows)
    c.write(directory / "plan.json", plan)
    c.write(directory / "summary.json", dict(rows=rows, metrics=c.five_seed_summary(rows)))
    monkeypatch.setattr(c, "validate_seed", lambda e: e)
    monkeypatch.setattr(c, "study_for", lambda *args: pytest.fail("Must not ask or retry a trial"))
    state = c.advance_dataset(tmp_path, {}, "mentalarithmetic")
    assert state["status"] == "fixed" and state["completed_seeds"] == 5
    assert "winner" not in state and state["independent_test_evaluation"] is False
    plan["entries"] = rows[:4]
    c.write(directory / "plan.json", plan)
    with pytest.raises(ValueError):
        c.advance_dataset(tmp_path, {}, "mentalarithmetic")


@pytest.mark.parametrize("slug", ["tuev", "tuab", "isruc", "hmc"])
def test_other_datasets_can_still_submit(tmp_path, monkeypatch, slug):
    control(tmp_path)
    commands = []
    monkeypatch.setattr(c, "submit_once", lambda path, command: commands.append(command) or "123")
    manifest = dict(python="python", gpu_partitions="gl40s_short", cpus=2,
                    memory_gib=20, time_limit="04:00:00", concurrency_per_dataset=2)
    plan = dict(directory=str(tmp_path / slug), number=1)
    assert c.submit_trial(tmp_path, manifest, slug, plan) == "123"
    assert len(commands) == 1


def test_fixed_dataset_does_not_enable_final_aggregate(tmp_path):
    from types import SimpleNamespace
    from scripts import monitor_optuna_search as monitor
    datasets = {slug: dict(status="complete") for slug in c.NAMES}
    datasets["mentalarithmetic"] = dict(status="fixed")
    assert c.campaign_status(datasets) == "finished_with_fixed_settings"
    with pytest.raises(ValueError, match="All five"):
        monitor.publish(SimpleNamespace(), dict(status=c.campaign_status(datasets), datasets=datasets))
    datasets["hmc"] = dict(status="running")
    assert c.campaign_status(datasets) == "running"
