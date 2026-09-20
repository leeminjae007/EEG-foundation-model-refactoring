"""Submit only failed/missing downstream seeds for a completed experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from submit_experiment_downstream import DATASETS, SEEDS, TEMPLATE_PREFIX, warmup

ROOT = Path(__file__).resolve().parents[1]


def require_submission_runtime():
    """Fail before sbatch when the launcher is not the project environment."""
    missing = [name for name in ("torch", "lmdb", "yaml") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "The submitting Python lacks required packages (%s). "
            "Activate eeg-foundation-model-cu118, or invoke this script with "
            "conda run -n eeg-foundation-model-cu118 python." % ", ".join(missing)
        )


def readable_result(path):
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError):
        return False


def main():
    require_submission_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--account", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); experiment = args.experiment.resolve(); checkpoint = args.checkpoint.resolve()
    if not (experiment / "manifest.json").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("existing experiment manifest and verified checkpoint are required")
    # Never collide with an existing array writing these same seed folders.
    active_names = [experiment.name + "-ds", experiment.name + "-repair-ds"]
    active = subprocess.run(
        ["squeue", "-h", "-n", ",".join(active_names), "-o", "%i"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    if active:
        raise RuntimeError("an existing downstream array is still active; wait for it to finish before repair submission: " + active)
    repair = experiment / "downstream_repair"; source = repair / "source"
    if source.exists(): shutil.rmtree(source)
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(".git", "outputs", "results", "__pycache__", "*.pyc", "*.pth"))
    subprocess.run([sys.executable, str(source / "scripts/visualize_fusion_gates.py"), "--checkpoint", str(checkpoint), "--output-dir", str(repair / "gate_visualization")], check=True)
    config_dir = experiment / "configs/downstream"; config_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            output = experiment / "downstream" / dataset / ("seed-%d" % seed)
            if readable_result(output / "result.json"):
                continue
            template = source / ("configs/downstream/%s_%s_seed%d.yaml" % (TEMPLATE_PREFIX.get(dataset, "gr9-1"), dataset, seed))
            config = yaml.safe_load(template.read_text())
            if warmup(config): raise ValueError("downstream warmup is forbidden: " + str(template))
            config["model"]["checkpoint"] = str(checkpoint); config["runtime"]["output"] = str(output); config["data"]["num_workers"] = 2
            path = config_dir / ("%s_seed%d.yaml" % (dataset, seed)); path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            entries.append({"dataset": dataset, "seed": seed, "config": str(path), "output": str(output)})
    (experiment / "downstream_entries.json").write_text(json.dumps(entries, indent=2) + "\n")
    print("skipped complete seeds:", len(DATASETS) * len(SEEDS) - len(entries), "submitted candidates:", len(entries))
    if args.dry_run or not entries: return
    cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
    default_policy = cluster["downstream"]
    logs = experiment / "downstream/logs"; logs.mkdir(parents=True, exist_ok=True)
    # TUAB approaches the four-hour budget, while the other downstream tasks
    # do not.  Give repair seeds one extra hour on a100_short; this avoids
    # reserving a long-partition allocation for work expected to finish near
    # the current four-hour boundary.
    groups = (("standard", [i for i, e in enumerate(entries) if e["dataset"] != "tuab"], dict(default_policy)),
              ("tuab", [i for i, e in enumerate(entries) if e["dataset"] == "tuab"],
               dict(default_policy, partitions="a100_short", time="05:00:00")))
    submissions = []
    for label, indices, policy in groups:
        if not indices:
            continue
        command = ["sbatch", "--parsable", "--account=" + (args.account or cluster["account"]),
                   "--job-name=" + experiment.name + "-repair-" + label,
                   "--partition=" + policy["partitions"], "--nodes=1", "--ntasks=1",
                   "--gpus-per-task=" + policy["gpu"] + ":1", "--cpus-per-task=" + str(policy["cpus_per_task"]),
                   "--mem=" + policy["memory"], "--time=" + policy["time"],
                   "--array=" + ",".join(map(str, indices)) + "%" + str(policy["array_parallelism"]),
                   "--output=" + str(logs / ("repair_" + label + "_%A_%a.out")),
                   "--error=" + str(logs / ("repair_" + label + "_%A_%a.err")),
                   "--wrap=exec " + sys.executable + " " + str(source / "scripts/downstream_experiment_worker.py") + " --experiment " + str(experiment) + " --source " + str(source)]
        excluded = sorted(set(cluster["pretrain"].get("excluded_nodes", [])) | set(policy.get("excluded_nodes", [])))
        if excluded:
            command.append("--exclude=" + ",".join(excluded))
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        submissions.append({"job": job, "group": label, "indices": indices, "resources": policy})
        print(job, label)
    (repair / "submission.json").write_text(json.dumps({"jobs": submissions, "checkpoint": str(checkpoint), "entries": entries}, indent=2) + "\n")


if __name__ == "__main__": main()
