import json
from pathlib import Path

import pytest
import yaml

from ablation.optuna_search import campaign as c


def test_five_seed_objective_rejects_partial_duplicate_and_nonfinite():
    rows = [dict(seed=s, test={"balanced_accuracy": v}) for s, v in zip(c.SEEDS, [.5, .6, .7, .8, .9])]
    assert c.five_seed_summary(rows)["balanced_accuracy"]["mean"] == .7
    assert c.five_seed_summary(rows)["balanced_accuracy"]["population_sd"] == pytest.approx(2 ** .5 / 10)
    for wrong in [rows[:4], rows[:4] + [rows[0]], rows[:4] + [dict(seed=3407, test={"balanced_accuracy": float("nan")})]]:
        with pytest.raises(ValueError):
            c.five_seed_summary(wrong)


@pytest.mark.parametrize("slug", c.NAMES)
def test_search_constraints_and_joint_candidates(slug):
    base = yaml.safe_load(Path(f"configs/downstream/gr9-1_{slug}_seed42.yaml").read_text())
    hp = dict(lr=1e-5 if slug == "tuab" else 1e-4,
              wd=5e-5 if slug == "tuab" else .05 if slug == "hmc" else .01,
              dropout=.1)
    for candidate in c.initial_candidates(slug, hp):
        cfg = c.config_for(base, slug, candidate, "/output")
        assert [cfg["optimization"][k] for k in c.LR_KEYS] == [candidate["lr"]] * 3
        assert cfg["optimization"]["epochs"] == base["optimization"]["epochs"]
        assert .1 <= cfg["model"]["head_dropout"] <= .3
        assert cfg["runtime"]["early_stopping"]["min_epochs"] >= 5
        assert cfg["optimization"]["label_smoothing"] == (0 if slug in ("tuab", "mentalarithmetic") else .1)
    assert any(sum(h[k] != hp[k] for k in hp) >= 2 for h in c.initial_candidates(slug, hp))


def test_early_stop_minimum_and_patience():
    p = dict(min_epochs=5, patience=3)
    assert not c.should_stop(4, {"balanced_accuracy": dict(epoch=1)}, p)
    assert c.should_stop(5, {"balanced_accuracy": dict(epoch=1)}, p)
    assert not c.should_stop(5, {"balanced_accuracy": dict(epoch=4)}, p)
    assert c.should_stop(7, {"balanced_accuracy": dict(epoch=4)}, p)


def test_engine_patch_selects_validation_without_changing_training_or_cosine():
    source = Path("src/training/engine.py").read_text(encoding="utf-8")
    modified = c.connected_engine(source)
    compile(modified, "engine.py", "exec")
    assert 'score = validation[selector]' in modified
    assert 'score = test_metrics[selector]' not in modified
    assert 'optimization["epochs"] * updates' in modified
    assert 'loss = downstream_loss(spec.task, logits, labels, optimization)' in modified
    assert modified.partition("def run_finetune(")[0] == source.partition("def run_finetune(")[0]


def test_submission_does_not_duplicate_after_ambiguous_response(tmp_path, monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(args)
        raise TimeoutError("unknown whether sbatch accepted")
    monkeypatch.setattr(c.subprocess, "run", timeout)
    with pytest.raises(TimeoutError):
        c.submit_once(tmp_path / "submission.json", ["sbatch"])
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit_once(tmp_path / "submission.json", ["sbatch"])
    assert len(calls) == 1


def test_completed_seed_requires_validation_selected_checkpoint_and_minimum(tmp_path):
    cfg = dict(seed=42, data={"dataset": "tuab"}, optimization={"epochs": 20},
               runtime={"early_stopping": {"min_epochs": 5, "patience": 3}})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "best-balanced_accuracy.pth").write_bytes(b"fixture")
    (tmp_path / "validation.jsonl").write_text("\n".join(json.dumps(dict(epoch=e, balanced_accuracy=.8 if e == 1 else .7)) for e in range(1, 6)))
    result = {"balanced_accuracy": {"selection": dict(score=.8, epoch=1),
                                  "test": dict(balanced_accuracy=.6, auroc=.9, auprc=.9)}}
    c.write(tmp_path / "result.json", result)
    c.write(tmp_path / "early_stop.json", dict(epoch=5))
    e = dict(seed=42, output=str(tmp_path), config=str(config), config_sha256=c.digest(config))
    assert c.validate_seed(e)["test"]["balanced_accuracy"] == .6
    result["balanced_accuracy"]["selection"]["score"] = .9
    c.write(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="Invalid validation"):
        c.validate_seed(e)


def test_publication_gates_all_datasets_and_labels_test_tuning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import monitor_optuna_search as monitor
    reference = tmp_path / "configs/results/literature_baselines.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"model_order": ["CBraMod", "CSBrain", "REVE"], "datasets": {}}))
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    published = []
    monkeypatch.setattr(monitor, "update_global", lambda sections, stamp, alias: published.extend(sections))
    args = SimpleNamespace(stamp="09162040", alias="knn37-optuna-test")
    rows = [dict(seed=s, test={"balanced_accuracy": .7}, selection={"epoch": 1, "split": "validation"}, source_result="fixture") for s in c.SEEDS]
    winner = dict(rows=rows, metrics=c.five_seed_summary(rows), hp=dict(lr=1e-4, wd=.05, dropout=.2), trial=0)
    status = dict(status="complete", datasets={slug: dict(status="complete", winner=winner) for slug in c.NAMES})
    partial = dict(status, datasets=dict(list(status["datasets"].items())[:4]))
    with pytest.raises(ValueError, match="All five"):
        monitor.publish(args, partial)
    assert not (tmp_path / "outputs/results").exists()
    monitor.publish(args, status)
    assert len(published) == 5
    folder = tmp_path / "outputs/results/09162040-tuab-knn37-optuna-test"
    assert "validation BAcc" in (folder / "results.md").read_text(encoding="utf-8")
    assert (folder / "results.csv").read_text(encoding="utf-8-sig").startswith("Metric,우리 (knn37-optuna-test),CBraMod")
    assert "ours_rank" in (folder / "results.csv").read_text(encoding="utf-8-sig")
    assert len((folder / "seed_results.csv").read_text(encoding="utf-8-sig").splitlines()) == 6
