"""Submit and audit the mask-55 ISRUC/HMC/Siena five-seed A100 grid."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import candidate_id, digest, write_json

BASE = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55")
EXPECTED_SHA256 = "a5ab908bde519241b199efcc701ea1352b3fcd9da068f600549845bef65babca"
SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ("isruc", "hmc", "siena")
LEARNING_RATES = (2.5e-5, 5e-5, 1e-4, 2e-4)
WEIGHT_DECAYS = (.005, .01, .05)
DROPOUTS = (.1, .2, .3)
TIME_LIMITS = {"isruc": "12:00:00", "hmc": "12:00:00", "siena": "04:00:00"}


def candidates():
    return [dict(learning_rate=lr, weight_decay=wd, dropout=dropout)
            for lr, wd, dropout in itertools.product(LEARNING_RATES, WEIGHT_DECAYS, DROPOUTS)]


def make_config(base, hp, checkpoint, output):
    config = json.loads(json.dumps(base))
    optimization = config["optimization"]
    if any("warmup" in str(key).lower() for key in optimization):
        raise ValueError("Downstream warmup is forbidden")
    for key in ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate"):
        optimization[key] = hp["learning_rate"]
    optimization["weight_decay"] = hp["weight_decay"]
    optimization["early_stopping"] = dict(monitor="balanced_accuracy", patience=10,
                                           min_epochs=15, min_delta=0.0)
    config["model"]["head_dropout"] = hp["dropout"]
    config["model"]["checkpoint"] = str(checkpoint)
    config["runtime"]["output"] = str(output)
    return config


def existing_campaigns(output_root):
    found = []
    for manifest_path in output_root.glob("*-gr2-d2-mask55-isruc-hmc-siena-grid/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            found.append(dict(path=str(manifest_path.parent), status=manifest.get("status"),
                              jobs=manifest.get("jobs", [])))
        except Exception as exc:
            found.append(dict(path=str(manifest_path.parent), error=repr(exc)))
    return found


def prepare(output_root):
    output_root = output_root.resolve()
    duplicate = existing_campaigns(output_root)
    if duplicate:
        raise ValueError("Existing matching campaign(s), refusing duplicate submission: " + json.dumps(duplicate))
    verified = json.loads((BASE / "pretrain/verified.json").read_text())
    checkpoint = Path(verified["checkpoint"])
    actual_sha = digest(checkpoint)
    if verified.get("sha256") != EXPECTED_SHA256 or actual_sha != EXPECTED_SHA256:
        raise ValueError(f"Mask-55 checkpoint SHA mismatch: manifest={verified.get('sha256')} actual={actual_sha}")
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%y%m%d-%H%M")
    campaign = output_root / (stamp + "-gr2-d2-mask55-isruc-hmc-siena-grid")
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        ".git", ".venv*", "outputs", "results", "__pycache__", "*.pyc", "*.pth"))
    (campaign / "pretrain").mkdir()
    write_json(campaign / "pretrain/verified.json", verified)
    entries = []
    for dataset in DATASETS:
        for hp in candidates():
            cid = candidate_id(hp)
            for seed in SEEDS:
                base_path = BASE / "configs/downstream" / f"{dataset}_seed{seed}.yaml"
                base = yaml.safe_load(base_path.read_text())
                output = campaign / "runs" / dataset / cid / f"seed-{seed}"
                config = make_config(base, hp, checkpoint, output)
                path = campaign / "configs" / dataset / f"{cid}_seed{seed}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                entries.append(dict(index=len(entries), dataset=dataset, seed=seed, candidate=cid,
                                    hp=hp, config=str(path), config_sha256=digest(path), output=str(output)))
    if len(entries) != 540 or len(candidates()) != 36:
        raise AssertionError("Expected 36 candidates x five seeds x three datasets")
    write_json(campaign / "downstream_entries.json", entries)
    cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
    manifest = dict(created_at_new_york=stamp, preset="gr2-d2-patch-dimension-mask55",
                    base_experiment=str(BASE), checkpoint=str(checkpoint), checkpoint_sha256=actual_sha,
                    datasets=list(DATASETS), learning_rates=list(LEARNING_RATES),
                    weight_decays=list(WEIGHT_DECAYS), dropouts=list(DROPOUTS), seeds=list(SEEDS),
                    candidates_per_dataset=36, runs_per_dataset=180, total_runs=540,
                    selection="Final HP uses five-seed mean validation balanced_accuracy only",
                    test_policy="Test metrics are reporting-only from each seed's validation-selected checkpoint",
                    publication_policy="Do not publish RESULTS.md until the validation winner has five audited seeds",
                    resources=dict(gpu="a100", partitions="a100_dev,a100_short,a100_long", cpus=2,
                                   memory="32G", concurrency_per_dataset=10,
                                   excluded_nodes=cluster["pretrain"].get("excluded_nodes", []),
                                   time_limits=TIME_LIMITS), jobs=[], status="prepared")
    write_json(campaign / "manifest.json", manifest)
    return campaign


def submit(campaign):
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = json.loads((campaign / "downstream_entries.json").read_text())
    logs = campaign / "logs"
    logs.mkdir(exist_ok=True)
    dependencies = []
    for dataset in DATASETS:
        indices = [entry["index"] for entry in entries if entry["dataset"] == dataset]
        resources = manifest["resources"]
        command = ["sbatch", "--parsable", "--account=system", "--job-name=mask55-grid-" + dataset,
                   "--partition=" + resources["partitions"], "--nodes=1", "--ntasks=1",
                   "--gpus-per-task=a100:1", "--cpus-per-task=2", "--mem=32G",
                   "--time=" + resources["time_limits"][dataset],
                   "--array=" + ",".join(map(str, indices)) + "%10",
                   "--exclude=" + ",".join(resources["excluded_nodes"]),
                   "--chdir=" + str(campaign / "source"),
                   "--output=" + str(logs / (dataset + "-%A_%a.out")),
                   "--error=" + str(logs / (dataset + "-%A_%a.err")),
                   "--wrap=exec " + shlex.join(["srun", "--ntasks=1", "--gpus-per-task=a100:1",
                       "--gpu-bind=single:1", "--kill-on-bad-exit=1", sys.executable,
                       str(campaign / "source/scripts/downstream_experiment_worker.py"),
                       "--experiment", str(campaign), "--source", str(campaign / "source")])]
        job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
        manifest["jobs"].append(dict(dataset=dataset, job=job, indices=indices,
                                     time_limit=resources["time_limits"][dataset]))
        write_json(manifest_path, manifest)
        dependencies.append(job)
    aggregate = subprocess.check_output([
        "sbatch", "--parsable", "--account=system", "--job-name=mask55-clinical-grid-results",
        "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        "--mem=4G", "--time=00:30:00", "--dependency=afterany:" + ":".join(dependencies),
        "--output=" + str(logs / "aggregate-%j.out"), "--error=" + str(logs / "aggregate-%j.err"),
        "--wrap=exec " + shlex.join([sys.executable,
            str(campaign / "source/scripts/submit_mask55_clinical_hp_grid.py"),
            "aggregate", "--campaign", str(campaign)])], text=True).strip().split(";", 1)[0]
    manifest.update(aggregate_job=aggregate, status="submitted")
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), jobs=manifest["jobs"], aggregate_job=aggregate), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / "downstream_entries.json").read_text())
    readable, missing = [], []
    for entry in entries:
        try:
            if digest(entry["config"]) != entry["config_sha256"]:
                raise ValueError("Config changed")
            payload = json.loads((Path(entry["output"]) / "result.json").read_text())
            selected = payload["balanced_accuracy"]
            validation = float(selected["selection"]["score"])
            epoch = int(selected["selection"]["epoch"])
            test = {key: float(value) for key, value in selected["test"].items()
                    if key in ("balanced_accuracy", "weighted_f1", "kappa")}
            if set(test) != {"balanced_accuracy", "weighted_f1", "kappa"}:
                raise ValueError("Missing multiclass test metric triplet")
            if not math.isfinite(validation) or not all(math.isfinite(value) for value in test.values()):
                raise ValueError("Non-finite score")
            readable.append(dict(**entry, validation_bacc=validation, selected_epoch=epoch, test=test))
        except Exception as exc:
            missing.append(dict(dataset=entry["dataset"], candidate=entry["candidate"],
                                seed=entry["seed"], error=repr(exc)))
    summaries = []
    for dataset in DATASETS:
        for hp in candidates():
            cid = candidate_id(hp)
            group = [row for row in readable if row["dataset"] == dataset and row["candidate"] == cid]
            if len(group) != 5 or {row["seed"] for row in group} != set(SEEDS):
                continue
            summaries.append(dict(dataset=dataset, candidate=cid, hp=hp,
                validation_bacc_mean=statistics.mean(row["validation_bacc"] for row in group),
                validation_bacc_sd=statistics.pstdev(row["validation_bacc"] for row in group),
                test={metric: dict(mean=statistics.mean(row["test"][metric] for row in group),
                                   sd=statistics.pstdev(row["test"][metric] for row in group))
                      for metric in ("balanced_accuracy", "weighted_f1", "kappa")}))
    winners = {}
    for dataset in DATASETS:
        complete = [row for row in summaries if row["dataset"] == dataset]
        if complete:
            winners[dataset] = min(complete, key=lambda row: (-row["validation_bacc_mean"],
                                                               row["validation_bacc_sd"], row["candidate"]))
    status = "complete" if not missing and len(readable) == 540 and len(summaries) == 108 else "incomplete"
    write_json(campaign / "grid_summary.json", dict(status=status, expected_runs=540,
        readable_runs=len(readable), missing=missing, candidate_summaries=summaries,
        winner_by_validation=winners, test_used_for_selection=False))
    with (campaign / "seed_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "candidate", "seed", "learning_rate", "weight_decay", "dropout",
                  "selected_epoch", "validation_bacc", "test_balanced_accuracy", "test_weighted_f1",
                  "test_kappa", "config", "output"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in readable:
            writer.writerow(dict(dataset=row["dataset"], candidate=row["candidate"], seed=row["seed"],
                **row["hp"], selected_epoch=row["selected_epoch"], validation_bacc=row["validation_bacc"],
                test_balanced_accuracy=row["test"]["balanced_accuracy"],
                test_weighted_f1=row["test"]["weighted_f1"], test_kappa=row["test"]["kappa"],
                config=row["config"], output=row["output"]))
    with (campaign / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "candidate", "learning_rate", "weight_decay", "dropout",
                  "validation_bacc_mean", "validation_bacc_sd", "test_bacc_mean", "test_bacc_sd",
                  "test_weighted_f1_mean", "test_weighted_f1_sd", "test_kappa_mean", "test_kappa_sd"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in summaries:
            writer.writerow(dict(dataset=row["dataset"], candidate=row["candidate"], **row["hp"],
                validation_bacc_mean=row["validation_bacc_mean"], validation_bacc_sd=row["validation_bacc_sd"],
                test_bacc_mean=row["test"]["balanced_accuracy"]["mean"],
                test_bacc_sd=row["test"]["balanced_accuracy"]["sd"],
                test_weighted_f1_mean=row["test"]["weighted_f1"]["mean"],
                test_weighted_f1_sd=row["test"]["weighted_f1"]["sd"],
                test_kappa_mean=row["test"]["kappa"]["mean"], test_kappa_sd=row["test"]["kappa"]["sd"]))
    lines = ["# Mask-55 ISRUC/HMC/Siena HP grid", "", f"Status: **{status}**", "",
             "Selection uses five-seed mean validation BAcc only; test metrics are reporting-only.", ""]
    for dataset in DATASETS:
        winner = winners.get(dataset)
        if winner:
            lines += [f"## {dataset}", "", f"Validation-selected HP: `{winner['hp']}`",
                      f"Validation BAcc: {winner['validation_bacc_mean']:.6f} ± {winner['validation_bacc_sd']:.6f}", ""]
            for metric, value in winner["test"].items():
                lines.append(f"- Test {metric}: {value['mean']:.6f} ± {value['sd']:.6f}")
            lines.append("")
    (campaign / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["status"] = status
    write_json(manifest_path, manifest)
    print(json.dumps(dict(status=status, readable_runs=len(readable), missing=len(missing),
                          winners=winners), indent=2))
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
        if args.action == "launch":
            submit(campaign)
        else:
            print(campaign)
        return 0
    if args.campaign is None:
        parser.error("aggregate requires --campaign")
    return aggregate(args.campaign.resolve())


if __name__ == "__main__":
    sys.exit(main())
