import csv
import json

from scripts.experiment_results import update


def write_result(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"balanced_accuracy": {"test": metrics}}), encoding="utf-8")


def test_binary_seed_rows_are_incrementally_updated(tmp_path):
    first = tmp_path / "seed42" / "result.json"
    second = tmp_path / "seed696" / "result.json"
    write_result(first, {"balanced_accuracy": .7, "auroc": .8, "auprc": .6})
    write_result(second, {"balanced_accuracy": .9, "auroc": .6, "auprc": .8})
    update(tmp_path, "tuab", 42, first)
    target = update(tmp_path, "tuab", 696, second)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["Metric"] for row in rows] == ["B-ACC", "AUROC", "AUPRC"]
    assert rows[0]["seed_42"] == "0.7" and rows[0]["seed_696"] == "0.9"
    assert rows[0]["n_complete"] == "2" and rows[0]["status"] == "running"


def test_multiclass_schema_is_selected(tmp_path):
    result = tmp_path / "result.json"
    write_result(result, {"balanced_accuracy": .7, "weighted_f1": .8, "kappa": .6})
    target = update(tmp_path, "tuev", 42, result)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        assert [row["Metric"] for row in csv.DictReader(stream)] == ["B-ACC", "Weighted-F1", "Kappa"]
