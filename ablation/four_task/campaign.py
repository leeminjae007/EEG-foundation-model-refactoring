"""Snapshot the completed baseline and submit only the four requested tasks."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from ablation.tuab.campaign import CHECKPOINT, CHECKPOINT_SHA256, SEEDS, digest, write_json, verify_source

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = "outputs/four_task_recipe_20260915"
BASELINE = "outputs/nearest3_7_downstream_20260913"
NAME = "four-task-hp-20260915"
RECIPES = {
    "seed-v": {"name": "SEED-V", "slug": "seedv", "weight_decay": .01,
               "head_dropout": .1, "head_hidden_tokens": 4, "batch": 64, "accumulation": 1},
    "stress": {"name": "MentalArithmetic", "slug": "mentalarithmetic", "weight_decay": .01,
               "head_dropout": .2, "head_hidden_tokens": None, "batch": 64, "accumulation": 1},
    "isruc": {"name": "ISRUC", "slug": "isruc", "weight_decay": .01,
              "head_dropout": .1, "head_hidden_tokens": None, "batch": 2, "accumulation": 32},
    "hmc": {"name": "HMC", "slug": "hmc", "weight_decay": .05,
            "head_dropout": .1, "head_hidden_tokens": 4, "batch": 64, "accumulation": 1},
}


def validate_recipe(config):
    recipe = RECIPES[config["data"]["dataset"]]
    opt, model = config["optimization"], config["model"]
    expected = {"epochs": 50, "weight_decay": recipe["weight_decay"],
                "batch_size_per_gpu": recipe["batch"], "gradient_accumulation_steps": recipe["accumulation"],
                "tokenizer_learning_rate": 1e-4, "encoder_learning_rate": 1e-4, "head_learning_rate": 1e-4,
                "min_learning_rate": 1e-6, "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
                "label_smoothing": .1, "gradient_clip_norm": 1.0}
    for key, value in expected.items():
        if opt[key] != value:
            raise ValueError("Unexpected recipe value: " + key)
    for key in ("head_dropout", "head_hidden_tokens"):
        if model[key] != recipe[key]:
            raise ValueError("Unexpected head setting: " + key)
    if config["data"]["dataset"] == "stress" and opt["class_counts"] != [1007, 336]:
        raise ValueError("MentalArithmetic must retain its weighted binary loss")
    if config["seed"] not in SEEDS:
        raise ValueError("Unexpected seed")


def make_config(base, output):
    config = deepcopy(base)
    config["optimization"].pop("warmup_epochs", None)
    config["optimization"].pop("warmup_ratio", None)
    if config["data"]["dataset"] == "hmc":
        config["model"]["head_hidden_tokens"] = 4
    config["runtime"]["output"] = str(output)
    validate_recipe(config)
    return config


def prepare(campaign):
    if (campaign / "manifest.json").exists():
        verify_source(campaign)
        return
    baseline = ROOT / BASELINE
    old_manifest = json.loads((baseline / "manifest.json").read_text())
    if digest(baseline / "array.json") != old_manifest["array_sha256"]:
        raise ValueError("Baseline array changed")
    for item in old_manifest["source_files"]:
        if digest(baseline / "source" / item["path"]) != item["sha256"]:
            raise ValueError("Baseline source changed: " + item["path"])
    checkpoint = ROOT / CHECKPOINT
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Wrong pretrained checkpoint")
    array = json.loads((baseline / "array.json").read_text())
    planned = []
    for dataset, recipe in RECIPES.items():
        for seed in SEEDS:
            matches = [e for e in array if e["dataset"] == dataset and e["seed"] == seed]
            if len(matches) != 1:
                raise ValueError("Expected exactly one completed baseline")
            entry = matches[0]
            path = Path(entry["config"])
            if digest(path) != entry["config_sha256"]:
                raise ValueError("Baseline config changed")
            base = yaml.safe_load(path.read_text())
            if Path(base["model"]["checkpoint"]).resolve() != checkpoint.resolve():
                raise ValueError("Baseline used a different checkpoint")
            if base["seed"] != seed or base["data"]["dataset"] != dataset:
                raise ValueError("Baseline index/config mismatch")
            if "balanced_accuracy" not in json.loads((Path(entry["result_dir"]) / "result.json").read_text()):
                raise ValueError("Baseline has not completed")
            output = campaign / "runs" / (recipe["slug"] + "_seed" + str(seed))
            config = make_config(base, output)
            planned.append((entry, config, output))
    source = campaign / "source"
    source.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    # The historical training engine, data loaders and evaluation are copied byte-for-byte.
    shutil.copytree(baseline / "source/src", source / "src", ignore=ignore)
    (source / "ablation").mkdir()
    shutil.copy2(ROOT / "ablation/__init__.py", source / "ablation/__init__.py")
    for package in ("tuab", "four_task"):
        shutil.copytree(ROOT / "ablation" / package, source / "ablation" / package, ignore=ignore)
    (source / "configs").mkdir()
    entries = []
    for previous, config, output in planned:
        path = source / "configs" / (output.name + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entries.append({"index": len(entries), "dataset": config["data"]["dataset"],
                        "name": RECIPES[config["data"]["dataset"]]["name"], "seed": config["seed"],
                        "config": str(path), "output": str(output), "config_sha256": digest(path),
                        "baseline_config": previous["config"], "baseline_config_sha256": previous["config_sha256"],
                        "baseline_result_dir": previous["result_dir"]})
    files = [{"path": p.relative_to(source).as_posix(), "sha256": digest(p)}
             for p in sorted(source.rglob("*")) if p.is_file()]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
                "entries": entries, "source_files": files, "recipes": RECIPES,
                "source_origin": str(baseline / "source/src"), "core_source_changed": False,
                "common": {"learning_rate": 1e-4, "epochs": 50, "effective_batch": 64},
                "selection": "All-epoch validation BAcc; same-run AUROC/Kappa comparison retained",
                "requested_scope": "Four task hyperparameter recipes on the current nearest3-7 weight",
                "time_limit": "04:00:00", "max_concurrent": 10}
    write_json(campaign / "manifest.json", manifest)
    for item in files:
        (source / item["path"]).chmod(0o444)
    print(json.dumps({"prepared": len(entries), "datasets": list(RECIPES), "campaign": str(campaign)}), flush=True)


def run_entry(campaign):
    manifest = verify_source(campaign)
    entry = manifest["entries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = Path(entry["output"])
    if (output / "result.json").exists():
        print("Already completed: " + str(output), flush=True)
        return
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
               "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=four-hp-%s-%s" % (os.environ["SLURM_JOB_ID"], entry["index"]),
               "--module", "ablation.four_task.finetune", "--config", entry["config"]]
    if (output / "last.pth").exists():
        command += ["--resume", str(output / "last.pth")]
    print(json.dumps({"entry": entry, "command": command}), flush=True)
    subprocess.run(command, cwd=campaign / "source", check=True)


def submit(campaign):
    import fcntl
    with (campaign / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = campaign / "submission.json"
        if ledger.exists():
            print(ledger.read_text())
            return
        intent = campaign / "submission_intent.json"
        if intent.exists():
            raise RuntimeError("Reconcile prior submission intent with sacct before retrying")
        manifest = verify_source(campaign)
        smoke = json.loads((campaign / "smoke_validation.json").read_text())
        if not smoke["passed"] or smoke["manifest_sha256"] != digest(campaign / "manifest.json"):
            raise RuntimeError("The exact snapshot must pass smoke validation")
        if {r["dataset"] for r in smoke["checks"] if r["passed"]} != set(RECIPES):
            raise RuntimeError("All four datasets must pass validation")
        active = subprocess.check_output(["squeue", "-h", "-u", ROOT.owner(), "--name", NAME, "-o", "%i"], text=True).strip()
        if active:
            raise RuntimeError("Matching jobs already exist: " + active)
        runner = [str(ROOT / ".venv/bin/python"), "-m", "ablation.four_task.campaign", "run-entry", "--campaign", str(campaign)]
        wrapper = ('unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
                   'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; '
                   'export TMPDIR="/tmp/four-hp-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"; '
                   'exec srun --ntasks=1 --gpus-per-task=l40s:1 --cpu-bind=none --kill-on-bad-exit=1 ' + shlex.join(runner))
        command = ["sbatch", "--parsable", "--account=system", "--job-name=" + NAME,
                   "--partition=gl40s_dev,gl40s_short,gl40s_long", "--array=0-19%10",
                   "--nodes=1", "--ntasks=1", "--gpus-per-task=l40s:1", "--cpus-per-task=8",
                   "--mem=32G", "--time=" + manifest["time_limit"], "--nice=0",
                   "--chdir=" + str(campaign / "source"), "--output=" + str(campaign / "slurm-%A_%a.log"), "--wrap", wrapper]
        write_json(intent, {"command": command, "created_utc": datetime.now(timezone.utc).isoformat()})
        result = subprocess.run(command, capture_output=True, text=True)
        write_json(campaign / "submission_response.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Ambiguous sbatch response; reconcile before retrying")
        record = {"job_id": job, "tasks": 20, "array": "0-19%10", "datasets": list(RECIPES),
                  "seeds": SEEDS, "command": command, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": digest(campaign / "manifest.json")}
        write_json(ledger, record)
        print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "submit", "run-entry", "all"))
    parser.add_argument("--campaign", type=Path, default=ROOT / DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action in ("prepare", "all"):
        prepare(campaign)
    if args.action in ("smoke", "all") and not (campaign / "submission.json").exists():
        subprocess.run([sys.executable, "-m", "ablation.four_task.smoke", "--campaign", str(campaign)],
                       cwd=campaign / "source", check=True)
    if args.action in ("submit", "all"):
        submit(campaign)
    if args.action == "run-entry":
        run_entry(campaign)


if __name__ == "__main__":
    main()
