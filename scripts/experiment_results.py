"""Atomically update one experiment's per-dataset seed-result CSV."""
import argparse
import csv
import json
from pathlib import Path
import statistics

SEEDS = (42, 696, 1001, 1234, 3407)
BINARY = (("B-ACC", "balanced_accuracy"), ("AUROC", "auroc"), ("AUPRC", "auprc"))
MULTI = (("B-ACC", "balanced_accuracy"), ("Weighted-F1", "weighted_f1"), ("Kappa", "kappa"))

def metrics(payload):
    test = payload.get("balanced_accuracy", {}).get("test", {})
    return test if isinstance(test, dict) else {}

def update(experiment, dataset, seed, result):
    experiment, result = Path(experiment), Path(result)
    values = metrics(json.loads(result.read_text(encoding="utf-8")))
    schema = BINARY if {"auroc", "auprc"} <= set(values) else MULTI
    if not all(key in values for _, key in schema):
        raise ValueError("result.json lacks the required binary or multiclass metric triplet")
    target = experiment / ("results_" + dataset + ".csv")
    old = {}
    if target.exists():
        with target.open(encoding="utf-8-sig", newline="") as stream:
            old = {row["Metric"]: row for row in csv.DictReader(stream)}
    columns = ["Metric", *["seed_" + str(item) for item in SEEDS], "mean", "population_sd", "n_complete", "status"]
    rows = []
    for label, key in schema:
        row = {name: "" for name in columns}
        row.update(old.get(label, {})); row["Metric"] = label; row["seed_" + str(seed)] = f"{float(values[key]):.10g}"
        complete = [float(row["seed_" + str(item)]) for item in SEEDS if row.get("seed_" + str(item), "") != ""]
        if complete:
            row["mean"] = f"{statistics.fmean(complete):.10g}"; row["population_sd"] = f"{statistics.pstdev(complete):.10g}"
        row["n_complete"] = str(len(complete)); row["status"] = "complete" if len(complete) == len(SEEDS) else "running"
        rows.append(row)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    temporary.replace(target)
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True); parser.add_argument("--dataset", required=True); parser.add_argument("--seed", required=True, type=int); parser.add_argument("--result", required=True)
    args = parser.parse_args(); print(update(args.experiment, args.dataset, args.seed, args.result))
