"""Scientific constraints and staged promotion for the small knn37 search."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ablation.local_search import campaign as search


@pytest.mark.parametrize("slug", list(search.NAMES))
def test_requested_ranges_and_one_seed_candidates(slug):
    cfg = yaml.safe_load(Path(f"configs/downstream/gr9-1_{slug}_seed42.yaml").read_text())
    # The actual TUAB campaign uses the completed server baseline below.
    if slug == "tuab":
        for key in search.LR_KEYS:
            cfg["optimization"][key] = 1e-5
        cfg["model"]["head_dropout"] = 0.1
        cfg["optimization"]["weight_decay"] = 5e-5
    base = search.anchor(cfg)
    variants = search.first_candidates(slug, base)
    assert len(variants) == (6 if slug == "tuab" else 7)
    assert len({search.cid(hp) for hp in variants}) == len(variants)
    for hp in variants:
        assert sum(hp[k] != base[k] for k in base) == 1
        new = search.config_for(cfg, hp, "/tmp/candidate")
        assert len({new["optimization"][k] for k in search.LR_KEYS}) == 1
        assert 0.1 <= new["model"]["head_dropout"] <= 0.3
        assert new["optimization"]["epochs"] == cfg["optimization"]["epochs"]
        assert new["model"]["head_hidden_tokens"] == cfg["model"]["head_hidden_tokens"]
        assert not new["runtime"]["evaluate_test"]
        assert new["runtime"]["early_stopping"]["patience"] > 0
    if slug != "tuab":
        assert search.space(slug)["lr"] == [5e-5, 1e-4, 1.5e-4, 5e-4]


def dataset():
    return dict(anchor=dict(lr=1e-4, wd=.01, dropout=.1), start_seed=696,
                confirmation_seeds=[696, 42, 1234],
                baseline=[dict(seed=s, validation=.7) for s in search.SEEDS])


def test_only_improving_axes_generate_combinations():
    d = dataset()
    def row(hp, score, test):
        return dict(hp=hp, score=score, candidate=search.cid(hp), test=test)
    rows = [row(dict(d["anchor"], lr=5e-5), .71, .1),
            row(dict(d["anchor"], dropout=.2), .72, .1),
            row(dict(d["anchor"], wd=.02), .69, .99)]
    assert search.combinations(d, rows) == [dict(d["anchor"], lr=5e-5, dropout=.2)]
    assert search.combinations(d, rows[:1]) == []


def test_start_seed_recovery_alone_cannot_promote_to_five():
    d = dataset()
    hp = dict(d["anchor"], lr=5e-5)
    rows = [dict(seed=s, score=score, hp=hp, candidate=search.cid(hp))
            for s, score in [(696, .8), (42, .69), (1234, .69)]]
    assert search.confirmed_winner(d, rows) == []
    rows[1]["score"] = .72
    rows[2]["score"] = .71
    assert len(search.confirmed_winner(d, rows)) == 1
    assert search.confirmed_winner(d, rows[:2]) == []


def test_early_stop_gate_is_local_and_keeps_scheduler():
    source = Path("src/training/engine.py").read_text(encoding="utf-8")
    connected = search.connected_engine(source)
    compile(connected, "engine.py", "exec")
    assert 'config["runtime"].get("early_stopping")' in connected
    assert 'optimization["epochs"] * updates' in connected
    assert 'config["runtime"].get("early_stopping")' not in source


def test_early_stopped_results_are_complete_only_with_valid_patience(tmp_path):
    cfg = yaml.safe_load(Path("configs/downstream/gr9-1_hmc_seed42.yaml").read_text())
    hp = search.anchor(cfg)
    cfg = search.config_for(cfg, hp, tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    (tmp_path / "resolved_config.yaml").write_bytes(config.read_bytes())
    (tmp_path / "best-balanced_accuracy.pth").write_bytes(b"fixture")
    (tmp_path / "validation.jsonl").write_text("\n".join(json.dumps(dict(epoch=i, balanced_accuracy=.7 if i == 1 else .6)) for i in range(1, 13)))
    search.write_json(tmp_path / "result.json", {"balanced_accuracy": {"selection": {"score": .7, "epoch": 1}, "test": None}})
    entry = dict(output=str(tmp_path), config=str(config), config_sha256=search.digest(config), seed=42, hp=hp)
    with pytest.raises(FileNotFoundError):
        search.check_result(entry)
    search.write_json(tmp_path / "early_stop.json", dict(epoch=12))
    assert search.check_result(entry)["score"] == .7
    r = search.read(tmp_path / "result.json")
    r["balanced_accuracy"]["test"] = {"balanced_accuracy": .99}
    search.write_json(tmp_path / "result.json", r)
    with pytest.raises(ValueError, match="must not evaluate test"):
        search.check_result(entry)


def test_submission_is_idempotent_and_ambiguous_responses_are_not_retried(tmp_path, monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="12345;cluster\n", stderr="")
    monkeypatch.setattr(search.subprocess, "run", fake)
    path = tmp_path / "submit.json"
    assert search.submit_once(path, ["sbatch"]) == "12345"
    assert search.submit_once(path, ["sbatch"]) == "12345"
    assert len(calls) == 1
    search.write_json(tmp_path / "bad_intent.json", {})
    with pytest.raises(RuntimeError, match="Ambiguous"):
        search.submit_once(tmp_path / "bad.json", ["sbatch"])


def test_publisher_refuses_incomplete_campaign_before_writes(tmp_path, monkeypatch):
    from scripts import monitor_local_search as monitor
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    (tmp_path / "configs/results").mkdir(parents=True)
    (tmp_path / "configs/results/literature_baselines.json").write_text('{"model_order": [], "datasets": {}}')
    args = SimpleNamespace(alias="knn37-hp", stamp="09161157")
    status = {"datasets": {slug: {} for slug in search.NAMES}}
    with pytest.raises(ValueError, match="All requested"):
        monitor.publish(args, status, {})
    assert not (tmp_path / "outputs").exists()


def test_stage_plan_never_runs_five_initial_seeds(tmp_path):
    bases = []
    for seed in search.SEEDS:
        cfg = yaml.safe_load(Path("configs/downstream/gr9-1_hmc_seed42.yaml").read_text())
        cfg["seed"] = seed
        path = tmp_path / f"base-{seed}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        bases.append(dict(seed=seed, config=str(path)))
    search.write_json(tmp_path / "manifest.json", {"datasets": {"hmc": {"baseline": bases}}})
    hp = search.anchor(cfg)
    plan = search.plan_stage(tmp_path, "hmc", "screen", search.first_candidates("hmc", hp), [1001])
    assert len(plan["entries"]) == 7
    assert {e["seed"] for e in plan["entries"]} == {1001}
    assert search.load_plan(tmp_path, "hmc", "screen") == plan
