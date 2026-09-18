"""Prepare and submit a five-seed downstream campaign for one verified pretrain."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml

SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ("chb", "siena", "physionet_mi", "tuev", "tuab", "faced", "seedv", "mentalarithmetic", "isruc", "hmc")

def warmup(value):
    if isinstance(value, dict): return any("warmup" in str(k).lower() or warmup(v) for k, v in value.items())
    return any(warmup(v) for v in value) if isinstance(value, list) else False

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path); parser.add_argument("--checkpoint", type=Path); parser.add_argument("--account", default=None); parser.add_argument("--hold", action="store_true", help="Submit the array held until the pretrain verifier releases it."); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); experiment = args.experiment.resolve(); source = experiment / "source"; checkpoint = (args.checkpoint or experiment / "pretrain/checkpoint-epoch-0040.pth").resolve()
    if not source.is_dir(): raise FileNotFoundError("experiment/source is required")
    if not args.hold and not checkpoint.is_file(): raise FileNotFoundError("verified checkpoint is required unless --hold is used")
    cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]; policy = cluster["downstream"]
    config_dir = experiment / "configs/downstream"; config_dir.mkdir(parents=True, exist_ok=True); entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            template = source / ("configs/downstream/gr9-1_%s_seed%d.yaml" % (dataset, seed)); config = yaml.safe_load(template.read_text())
            if warmup(config): raise ValueError("downstream warmup is forbidden: " + str(template))
            output = experiment / "downstream" / dataset / ("seed-%d" % seed); config["model"]["checkpoint"] = str(checkpoint); config["runtime"]["output"] = str(output); config["data"]["num_workers"] = 2
            path = config_dir / ("%s_seed%d.yaml" % (dataset, seed)); path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            entries.append({"dataset": dataset, "seed": seed, "config": str(path), "output": str(output)})
    (experiment / "downstream_entries.json").write_text(json.dumps(entries, indent=2) + "\n")
    if args.prepare_only: return
    account = args.account or cluster["account"]; logs = experiment / "downstream/logs"; logs.mkdir(parents=True, exist_ok=True)
    command = ["sbatch", "--parsable", "--account=" + account, "--job-name=" + experiment.name + "-ds", "--partition=" + policy["partitions"], "--nodes=1", "--ntasks=1", "--gpus-per-task=" + policy["gpu"] + ":1", "--cpus-per-task=" + str(policy["cpus_per_task"]), "--mem=" + policy["memory"], "--time=" + policy["time"], "--array=0-%d%%%d" % (len(entries)-1, policy["array_parallelism"]), "--output=" + str(logs / "%A_%a.out"), "--error=" + str(logs / "%A_%a.err"), "--wrap=exec " + sys.executable + " " + str(source / "scripts/downstream_experiment_worker.py") + " --experiment " + str(experiment)]
    if args.hold:
        command.append("--hold")
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    manifest_path = experiment / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    manifest["downstream_job"] = job; manifest["downstream_state"] = "held" if args.hold else "submitted"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(job)
if __name__ == "__main__": main()
