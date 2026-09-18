"""Validate adaptive beam expansion, seed completeness and submission recovery."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ablation.bciciv2a import campaign as beam


def test_beam_expands_one_axis_without_repeating_candidates():
    initial = [beam.ANCHOR] + beam.neighbors([beam.ANCHOR], set())
    assert len(initial) == 9
    seen = {beam.candidate_id(hp) for hp in initial}
    parents = [initial[1], initial[-1]]
    expanded = beam.neighbors(parents, seen)
    assert expanded
    assert len({beam.candidate_id(hp) for hp in expanded}) == len(expanded)
    assert not seen.intersection(beam.candidate_id(hp) for hp in expanded)
    for hp in expanded:
        assert any(sum(hp[k] != parent[k] for k in beam.SPACE) == 1 for parent in parents)


def test_ranking_uses_validation_not_test():
    rows = [dict(candidate="a", validation_mean=.6, validation_sd=.02, test=.99),
            dict(candidate="b", validation_mean=.7, validation_sd=.03, test=.1),
            dict(candidate="c", validation_mean=.7, validation_sd=.01, test=.01)]
    assert [r["candidate"] for r in beam.rank_scores(rows)] == ["c", "b", "a"]


@pytest.fixture
def stage(tmp_path):
    source = tmp_path / "source/baseline_configs"
    source.mkdir(parents=True)
    base = yaml.safe_load(Path("configs/downstream/gr9-1_bciciv2a_seed42.yaml").read_text())
    for seed in beam.SEEDS:
        config = deepcopy(base)
        config["seed"] = seed
        (source / (str(seed) + ".yaml")).write_text(yaml.safe_dump(config))
    (tmp_path / "manifest.json").write_text("{}")
    plan = beam.plan_stage(tmp_path, "round_00", [beam.ANCHOR])
    yield tmp_path, plan
    for p in tmp_path.rglob("*"):
        if p.is_file():
            p.chmod(0o644)


def test_search_and_final_keep_test_separate(stage):
    path, plan = stage
    assert [e["seed"] for e in plan["entries"]] == list(beam.SEARCH_SEEDS)
    for e in plan["entries"]:
        cfg = yaml.safe_load(Path(e["config"]).read_text())
        assert "warmup_epochs" not in cfg["optimization"]
        assert cfg["runtime"]["evaluate_test"] is False
    final = beam.plan_stage(path, "final", [beam.ANCHOR])
    assert [e["seed"] for e in final["entries"]] == list(beam.SEEDS)
    assert all(yaml.safe_load(Path(e["config"]).read_text())["runtime"]["evaluate_test"] for e in final["entries"])


def test_partial_seeds_are_not_ranked_and_test_leak_is_rejected(stage):
    path, plan = stage
    for e in plan["entries"][:2]:
        out = Path(e["output"])
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_bytes(Path(e["config"]).read_bytes())
        (out / "validation.jsonl").write_text("\n".join(json.dumps({"epoch": i, "balanced_accuracy": .6}) for i in range(1, 51)))
        (out / "result.json").write_text(json.dumps({"balanced_accuracy": {"selection": {"score": .6}, "test": None}}))
    assert beam.read_scores(path, "round_00")["scores"] == []
    out = Path(plan["entries"][0]["output"])
    (out / "result.json").write_text(json.dumps({"balanced_accuracy": {"selection": {"score": .6}, "test": {"balanced_accuracy": .99}}}))
    report = beam.read_scores(path, "round_00")
    assert any("accessed test" in r["error"] for r in report["failures"])


def test_submission_records_intent_and_does_not_duplicate(tmp_path, monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert (tmp_path / "submission_intent.json").exists()
        return SimpleNamespace(returncode=0, stdout="123456;cluster\n", stderr="")
    monkeypatch.setattr(beam.subprocess, "run", fake)
    for _ in range(2):
        assert beam.submit_once(tmp_path / "submission.json", ["sbatch"]) == "123456"
    assert len(calls) == 1


def test_ambiguous_submission_is_not_retried(tmp_path, monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="ambiguous", stderr="")
    monkeypatch.setattr(beam.subprocess, "run", fake)
    with pytest.raises(RuntimeError):
        beam.submit_once(tmp_path / "submission.json", ["sbatch"])
    with pytest.raises(RuntimeError, match="reconcile"):
        beam.submit_once(tmp_path / "submission.json", ["sbatch"])
    assert len(calls) == 1


@pytest.mark.parametrize("round_limit,expected_stage", [(3, "round_01"), (1, "final")])
def test_controller_expands_or_fixes_winner_before_final_submission(stage, monkeypatch, round_limit, expected_stage):
    path, plan = stage
    score = {"candidate": beam.candidate_id(beam.ANCHOR), "hp": beam.ANCHOR,
             "validation_mean": .6, "validation_sd": .01, "rows": [], "stage": "round_00"}
    monkeypatch.setattr(beam, "MAX_ROUNDS", round_limit)
    monkeypatch.setattr(beam, "verify_source", lambda campaign: {})
    monkeypatch.setattr(beam, "read_scores", lambda *args: {"scores": [score], "failures": []})
    submitted = []
    def submit(campaign, next_stage):
        submitted.append(next_stage)
        next_plan = beam.load_plan(campaign, next_stage)
        if next_stage == "final":
            assert json.loads((campaign / "winner.json").read_text())["selected_before_test"]
            assert len(next_plan["entries"]) == 5
        else:
            assert len(next_plan["candidates"]) == 8
            assert len(next_plan["entries"]) == 24
    monkeypatch.setattr(beam, "submit_stage", submit)
    beam.advance(path, "round_00")
    assert submitted == [expected_stage]
