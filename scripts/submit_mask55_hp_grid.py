"""Launch the complete five-seed A100 HP grid for the final mask-55 model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERIMENT = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55")
SEEDS = (42, 696, 1001, 1234, 3407)
GRIDS = {
    "mentalarithmetic": {
        "learning_rate": (5e-5, 1e-4, 2e-4, 5e-4),
        "weight_decay": (.005, .01, .02),
        "dropout": (.1, .2, .3),
    },
    "physionet_mi": {
        "learning_rate": (2.5e-5, 5e-5, 1e-4, 2e-4),
        "weight_decay": (.005, .01, .02),
        "dropout": (.1, .2, .3),
    },
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def candidate_id(hp):
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:12]


def candidates(dataset):
    grid = GRIDS[dataset]
    return [dict(learning_rate=lr, weight_decay=wd, dropout=dropout)
            for lr, wd, dropout in itertools.product(
                grid["learning_rate"], grid["weight_decay"], grid["dropout"])]


def make_config(base, hp, checkpoint, output):
    config = json.loads(json.dumps(base))
    opt = config["optimization"]
    if any("warmup" in str(key).lower() for key in opt):
        raise ValueError("Downstream warmup is forbidden")
    for key in ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate"):
        opt[key] = hp["learning_rate"]
    opt["weight_decay"] = hp["weight_decay"]
    config["model"]["head_dropout"] = hp["dropout"]
    config["model"]["checkpoint"] = str(checkpoint)
    config["data"]["num_workers"] = 2
    config["runtime"]["output"] = str(output)
    return config


def prepare(output_root):
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%y%m%d-%H%M")
    campaign = output_root.resolve() / (stamp + "-gr2-d2-mask55-hp-grid")
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        ".git", ".venv*", "outputs", "results", "__pycache__", "*.pyc", "*.pth"))
    verified = json.loads((BASE_EXPERIMENT / "pretrain/verified.json").read_text())
    checkpoint = Path(verified["checkpoint"])
    if not checkpoint.is_file() or digest(checkpoint) != verified["sha256"]:
        raise ValueError("The mask-55 pretrained checkpoint is missing or changed")
    (campaign / "pretrain").mkdir()
    write_json(campaign / "pretrain/verified.json", verified)
    config_dir = campaign / "configs"
    entries = []
    for dataset in GRIDS:
        for hp in candidates(dataset):
            cid = candidate_id(hp)
            for seed in SEEDS:
                base_path = BASE_EXPERIMENT / "configs/downstream" / (f"{dataset}_seed{seed}.yaml")
                base = yaml.safe_load(base_path.read_text())
                output = campaign / "runs" / dataset / cid / f"seed-{seed}"
                config = make_config(base, hp, checkpoint, output)
                path = config_dir / dataset / f"{cid}_seed{seed}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                entries.append(dict(index=len(entries), dataset=dataset, seed=seed, candidate=cid,
                                    hp=hp, config=str(path), config_sha256=digest(path), output=str(output)))
    if len(entries) != 360 or any(len(candidates(dataset)) != 36 for dataset in GRIDS):
        raise AssertionError("Expected 36 candidates and five seeds for each of two datasets")
    write_json(campaign / "downstream_entries.json", entries)
    cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
    manifest = dict(created_at_new_york=stamp, preset="gr2-d2-patch-dimension-mask55",
                    base_experiment=str(BASE_EXPERIMENT), checkpoint=str(checkpoint),
                    checkpoint_sha256=verified["sha256"], python=sys.executable,
                    datasets=list(GRIDS), grids=GRIDS, seeds=list(SEEDS), entries=len(entries),
                    selection="Per-run best validation balanced accuracy; final HP by five-seed validation mean",
                    test_policy="Every grid run evaluates test only from its validation-selected checkpoint",
                    resources=dict(gpu="a100", partitions="a100_dev,a100_short,a100_long",
                                   cpus=2, memory="32G", time="04:00:00", concurrency_per_dataset=10,
                                   excluded_nodes=cluster["pretrain"].get("excluded_nodes", [])),
                    jobs=[], status="prepared")
    write_json(campaign / "manifest.json", manifest)
    return campaign


def submit(campaign):
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["jobs"] or manifest.get("aggregate_job"):
        raise ValueError("Grid already submitted")
    entries = json.loads((campaign / "downstream_entries.json").read_text())
    logs = campaign / "logs"
    logs.mkdir(exist_ok=True)
    resources = manifest["resources"]
    dependencies = []
    for dataset in GRIDS:
        indices = [e["index"] for e in entries if e["dataset"] == dataset]
        command = ["sbatch", "--parsable", "--account=system", "--job-name=mask55-grid-" + dataset,
                   "--partition=" + resources["partitions"], "--nodes=1", "--ntasks=1",
                   "--gpus-per-task=a100:1", "--cpus-per-task=2", "--mem=32G", "--time=04:00:00",
                   "--array=" + ",".join(map(str, indices)) + "%10",
                   "--exclude=" + ",".join(resources["excluded_nodes"]),
                   "--chdir=" + str(campaign / "source"),
                   "--output=" + str(logs / (dataset + "-%A_%a.out")),
                   "--error=" + str(logs / (dataset + "-%A_%a.err")),
                   "--wrap=exec " + shlex.join(["srun", "--ntasks=1", "--gpus-per-task=a100:1",
                       "--gpu-bind=single:1", "--kill-on-bad-exit=1", manifest["python"],
                       str(campaign / "source/scripts/downstream_experiment_worker.py"),
                       "--experiment", str(campaign), "--source", str(campaign / "source")])]
        job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
        manifest["jobs"].append(dict(dataset=dataset, job=job, indices=indices, command=command))
        write_json(manifest_path, manifest)
        dependencies.append(job)
    aggregate = subprocess.check_output([
        "sbatch", "--parsable", "--account=system", "--job-name=mask55-grid-results",
        "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        "--mem=4G", "--time=00:30:00", "--dependency=afterany:" + ":".join(dependencies),
        "--output=" + str(logs / "aggregate-%j.out"), "--error=" + str(logs / "aggregate-%j.err"),
        "--wrap=exec " + shlex.join([manifest["python"], str(campaign / "source/scripts/submit_mask55_hp_grid.py"),
                                      "aggregate", "--campaign", str(campaign)])
    ], text=True).strip().split(";", 1)[0]
    manifest.update(aggregate_job=aggregate, status="submitted")
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), jobs=manifest["jobs"], aggregate_job=aggregate), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / "downstream_entries.json").read_text())
    missing, rows = [], []
    for entry in entries:
        try:
            if digest(entry["config"]) != entry["config_sha256"]:
                raise ValueError("Config changed")
            result = json.loads((Path(entry["output"]) / "result.json").read_text())
            selected = result["balanced_accuracy"]
            validation = float(selected["selection"]["score"])
            test = {k: float(v) for k, v in selected["test"].items() if isinstance(v, (int, float))}
            if not math.isfinite(validation) or not all(math.isfinite(v) for v in test.values()):
                raise ValueError("Non-finite score")
            rows.append(dict(**entry, validation_bacc=validation, test=test,
                             selected_epoch=selected["selection"]["epoch"]))
        except Exception as exc:
            missing.append(dict(entry=entry, error=repr(exc)))
    summaries = []
    for dataset in GRIDS:
        for hp in candidates(dataset):
            cid = candidate_id(hp)
            group = [r for r in rows if r["dataset"] == dataset and r["candidate"] == cid]
            if len(group) != 5 or {r["seed"] for r in group} != set(SEEDS):
                continue
            metrics = sorted(set.intersection(*(set(r["test"]) for r in group)))
            summaries.append(dict(dataset=dataset, candidate=cid, hp=hp,
                validation_bacc_mean=statistics.mean(r["validation_bacc"] for r in group),
                validation_bacc_sd=statistics.pstdev(r["validation_bacc"] for r in group),
                test={metric: dict(mean=statistics.mean(r["test"][metric] for r in group),
                                   sd=statistics.pstdev(r["test"][metric] for r in group)) for metric in metrics},
                seeds=[dict(seed=r["seed"], validation_bacc=r["validation_bacc"],
                            selected_epoch=r["selected_epoch"], test=r["test"]) for r in group]))
    winners = {}
    for dataset in GRIDS:
        complete = [r for r in summaries if r["dataset"] == dataset]
        if complete:
            winners[dataset] = min(complete, key=lambda r: (-r["validation_bacc_mean"],
                                                             r["validation_bacc_sd"], r["candidate"]))
    status = "complete" if not missing and len(summaries) == 72 and len(winners) == 2 else "incomplete"
    report = dict(status=status, expected_runs=360, readable_runs=len(rows), missing=missing,
                  candidate_summaries=summaries, winner_by_validation=winners)
    write_json(campaign / "grid_summary.json", report)
    with (campaign / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "candidate", "learning_rate", "weight_decay", "dropout",
                  "validation_bacc_mean", "validation_bacc_sd", "balanced_accuracy_mean",
                  "balanced_accuracy_sd", "secondary_1", "secondary_1_mean", "secondary_1_sd",
                  "secondary_2", "secondary_2_mean", "secondary_2_sd"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            secondary = [k for k in row["test"] if k != "balanced_accuracy"]
            values = dict(dataset=row["dataset"], candidate=row["candidate"], **row["hp"],
                          validation_bacc_mean=row["validation_bacc_mean"],
                          validation_bacc_sd=row["validation_bacc_sd"],
                          balanced_accuracy_mean=row["test"]["balanced_accuracy"]["mean"],
                          balanced_accuracy_sd=row["test"]["balanced_accuracy"]["sd"])
            for i, metric in enumerate(secondary[:2], 1):
                values[f"secondary_{i}"] = metric
                values[f"secondary_{i}_mean"] = row["test"][metric]["mean"]
                values[f"secondary_{i}_sd"] = row["test"][metric]["sd"]
            writer.writerow(values)
    lines = ["# Mask-55 five-seed HP grid", "", f"Status: **{status}**", ""]
    for dataset, winner in winners.items():
        lines += [f"## {dataset}", "", f"Validation-selected HP: `{winner['hp']}`",
                  f"Validation BAcc: {winner['validation_bacc_mean']:.6f} ± {winner['validation_bacc_sd']:.6f}", ""]
        for metric, value in winner["test"].items():
            lines.append(f"- Test {metric}: {value['mean']:.6f} ± {value['sd']:.6f}")
        lines.append("")
    (campaign / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = status
    write_json(manifest_path, manifest)
    print(json.dumps({"status": status, "readable_runs": len(rows), "missing": len(missing),
                      "winners": winners}, indent=2))
    return 0 if status == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "aggregate"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results"))
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.action in ("prepare", "launch"):
        campaign = prepare(args.output_root)
        print(campaign)
        if args.action == "launch":
            submit(campaign)
    else:
        if args.campaign is None:
            parser.error("aggregate requires --campaign")
        return aggregate(args.campaign.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
