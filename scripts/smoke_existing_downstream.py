"""Submit a real-data A100 smoke run for one existing downstream experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from submit_experiment_downstream import DATASETS, SEEDS, TEMPLATE_PREFIX, warmup

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", choices=DATASETS, default="tusz")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=42)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    missing = [name for name in ("torch", "yaml", "lmdb") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Activate eeg-foundation-model-cu118 (missing: %s)" % ", ".join(missing))

    experiment = args.experiment.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (experiment / "manifest.json").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("existing experiment manifest and checkpoint are required")

    smoke = experiment / "downstream_smoke"
    source = smoke / "source"
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(".git", "outputs", "results", "__pycache__", "*.pyc", "*.pth"))
    prefix = TEMPLATE_PREFIX.get(args.dataset, "gr9-1")
    template = source / "configs" / "downstream" / ("%s_%s_seed%d.yaml" % (prefix, args.dataset, args.seed))
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    if warmup(config):
        raise ValueError("downstream warmup is forbidden: " + str(template))
    output = smoke / "output" / args.dataset / ("seed-%d" % args.seed)
    config["model"]["checkpoint"] = str(checkpoint)
    config["runtime"]["output"] = str(output)
    config["data"]["num_workers"] = 2
    config_path = smoke / "configs" / ("%s_seed%d.yaml" % (args.dataset, args.seed))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    entries_path = smoke / "downstream_entries.json"
    entries_path.write_text(json.dumps([{"dataset": args.dataset, "seed": args.seed,
                                          "config": str(config_path), "output": str(output)}], indent=2) + "\n")

    cluster = yaml.safe_load((source / "configs" / "cluster" / "bigpurple_a100.yaml").read_text())["slurm"]
    policy = cluster["downstream"]
    logs = smoke / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = [sys.executable, str(source / "scripts" / "downstream_experiment_worker.py"),
              "--experiment", str(smoke), "--source", str(source), "--smoke-batches", str(args.batches)]
    command = ["sbatch", "--parsable", "--account=" + cluster["account"],
               "--job-name=" + experiment.name + "-downstream-smoke",
               "--partition=" + policy["partitions"], "--nodes=1", "--ntasks=1",
               "--gpus-per-task=" + policy["gpu"] + ":1", "--cpus-per-task=" + str(policy["cpus_per_task"]),
               "--mem=" + policy["memory"], "--time=00:20:00", "--output=" + str(logs / "%j.out"),
               "--error=" + str(logs / "%j.err"), "--exclude=" + ",".join(policy["excluded_nodes"]),
               "--wrap=exec " + shlex.join(["srun", "--ntasks=1", "--gpus-per-task=" + policy["gpu"] + ":1",
                                              "--gpu-bind=single:1", "--kill-on-bad-exit=1"] + worker)]
    job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    (smoke / "submission.json").write_text(json.dumps({"job": job, "dataset": args.dataset,
        "seed": args.seed, "batches": args.batches, "checkpoint": str(checkpoint),
        "excluded_nodes": policy["excluded_nodes"]}, indent=2) + "\n")
    print(job)


if __name__ == "__main__":
    main()
