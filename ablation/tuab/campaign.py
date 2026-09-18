"""Prepare, smoke-test, submit and inspect the four-arm TUAB campaign on BigPurple."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

import yaml
from ablation.tuab.connection import connected_source

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = "outputs/tuab_recovery_retry2_20260915"
BASELINE = "outputs/nearest3_7_downstream_20260913/downstream"
CHECKPOINT = "outputs/geometry50_nearest3_7_t2_15_20260913/training/checkpoint-epoch-0040.pth"
CHECKPOINT_SHA256 = "689f67c47ee8c4b9d3e1d9aaf52d6d3c36ab930784011b100668ae4040c3304b"
SEEDS = (42, 1234, 696, 1001, 3407)
ARMS = ("baseline", "backbone_lr_x0p1", "head_first2_backbone_lr_x0p1", "head_h4")
RUN_ARMS = ARMS[1:]


def without_removed_seedvig(original):
    """Keep the copied historical registry importable after SEED-VIG removal."""
    for line in (
        "from src.data.datasets.seedvig_dataset import SEEDVIGDataset\n",
        '    "seed-vig": DatasetSpec(SEEDVIGDataset, "regression", 1, 17, 1600),\n',
    ):
        if original.count(line) != 1:
            raise ValueError("Inspect changed dataset registry before snapshotting")
        original = original.replace(line, "")
    return original


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_config(base, arm, checkpoint, output):
    if arm not in ARMS:
        raise ValueError("Unknown arm: " + arm)
    config = deepcopy(base)
    config["model"]["checkpoint"] = str(checkpoint)
    config["runtime"]["output"] = str(output)
    config["optimization"].pop("warmup_epochs", None)
    config["optimization"].pop("warmup_ratio", None)
    policy = {"arm": arm, "head_first_epochs": 0}
    if arm in ARMS[1:3]:
        for key in ("tokenizer_learning_rate", "encoder_learning_rate"):
            config["optimization"][key] = 1e-6
        policy["min_learning_rates"] = {"tokenizer": 1e-7, "encoder": 1e-7, "head": 1e-6}
    if arm == ARMS[2]:
        policy["head_first_epochs"] = 2
    if arm == "head_h4":
        config["model"]["head_hidden_tokens"] = 4
    config["finetune_ablation"] = policy
    return config


def verify_source(campaign):
    manifest = json.loads((campaign / "manifest.json").read_text())
    source = campaign / "source"
    for item in manifest["source_files"]:
        if digest(source / item["path"]) != item["sha256"]:
            raise RuntimeError("Snapshot changed: " + item["path"])
    if digest(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Pretrained checkpoint changed")
    return manifest


def prepare(campaign):
    if (campaign / "manifest.json").exists():
        verify_source(campaign)
        return
    checkpoint = ROOT / CHECKPOINT
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Wrong pretrained checkpoint; expected nearest3-7 epoch40")
    baseline_campaign = ROOT / BASELINE
    baseline_source = baseline_campaign.parent / "source"
    array_path = baseline_campaign.parent / "array.json"
    array = json.loads(array_path.read_text()) if array_path.exists() else []
    baseline_manifest_path = baseline_campaign.parent / "manifest.json"
    baseline_source_files = []
    if baseline_manifest_path.exists():
        baseline_manifest = json.loads(baseline_manifest_path.read_text())
        if digest(array_path) != baseline_manifest["array_sha256"]:
            raise ValueError("Completed baseline array manifest changed")
        baseline_source_files = baseline_manifest["source_files"]
        removed = {"src/data/datasets/seedvig_dataset.py",
                   "src/data/preprocessing/preprocessing_seedvig.py"}
        removed.update("configs/nearest3_7_seedvig_seed%d.yaml" % seed for seed in SEEDS)
        for item in baseline_source_files:
            path = baseline_source / item["path"]
            if not path.exists() and item["path"] in removed:
                continue
            if digest(path) != item["sha256"]:
                raise ValueError("Completed baseline snapshot changed: " + item["path"])
    bases = []
    for seed in SEEDS:
        run_dir = ROOT / BASELINE / ("tuab_seed" + str(seed))
        matches = [e for e in array if e["dataset"] == "tuab" and e["seed"] == seed]
        if array:
            if len(matches) != 1 or Path(matches[0]["result_dir"]) != run_dir:
                raise ValueError("Cannot identify the completed TUAB baseline")
            path = Path(matches[0]["config"])
            if digest(path) != matches[0]["config_sha256"]:
                raise ValueError("Completed baseline config changed")
        else:
            path = run_dir / "resolved_config.yaml"
        base = yaml.safe_load(path.read_text())
        expected = yaml.safe_load((ROOT / "configs/downstream" / ("gr9-1_tuab_seed%d.yaml" % seed)).read_text())
        if base["data"] != expected["data"]:
            raise ValueError("Completed baseline differs from canonical TUAB settings: data")
        normalized_optimization = deepcopy(base["optimization"])
        normalized_optimization.pop("warmup_epochs", None)
        normalized_optimization.pop("warmup_ratio", None)
        if normalized_optimization != expected["optimization"]:
            raise ValueError("Completed baseline differs from canonical TUAB settings: optimization")
        if base["seed"] != seed or base["model"]["head_dropout"] != .1 or base["model"]["head_hidden_tokens"] is not None:
            raise ValueError("Unexpected completed baseline")
        if Path(base["model"]["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("Completed baseline used a different weight")
        result = json.loads((run_dir / "result.json").read_text())
        if "balanced_accuracy" not in result:
            raise ValueError("Baseline must have completed validation-BAcc selection")
        bases.append((base, path))
    # Refuse to reuse a half-prepared directory; no existing scientific outputs are overwritten.
    source = campaign / "source"
    source.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    source_tree = baseline_source / "src" if baseline_source_files else ROOT / "src"
    shutil.copytree(source_tree, source / "src", ignore=ignore)
    if not (source / "src/data/datasets/registry.py").is_file():
        raise FileNotFoundError("Prepare on the server with its real src/data package")
    registry_path = source / "src/data/datasets/registry.py"
    registry = registry_path.read_text(encoding="utf-8")
    if "seedvig_dataset" in registry:
        registry_path.chmod(stat.S_IMODE(registry_path.stat().st_mode) | stat.S_IWUSR)
        registry_path.write_text(without_removed_seedvig(registry), encoding="utf-8")
    engine_path = source / "src/training/engine.py"
    engine_before = digest(engine_path)
    original = engine_path.read_text(encoding="utf-8")
    engine_path.chmod(stat.S_IMODE(engine_path.stat().st_mode) | stat.S_IWUSR)
    engine_path.write_text(connected_source(original), encoding="utf-8")
    (source / "ablation").mkdir()
    for name in ("__init__.py", "bootstrap.py"):
        shutil.copy2(ROOT / "ablation" / name, source / "ablation" / name)
    shutil.copytree(ROOT / "ablation/tuab", source / "ablation/tuab", ignore=ignore)
    (source / "configs").mkdir()
    entries = []
    for arm in RUN_ARMS:
        for base, baseline_path in bases:
            seed = base["seed"]
            output = campaign / "runs" / arm / ("tuab_seed" + str(seed))
            config = make_config(base, arm, checkpoint, output)
            path = source / "configs" / (arm + "_seed%d.yaml" % seed)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            entries.append({"index": len(entries), "arm": arm, "seed": seed,
                            "config": str(path), "config_sha256": digest(path),
                            "output": str(output), "baseline_config": str(baseline_path),
                            "baseline_result_dir": str(ROOT / BASELINE / ("tuab_seed" + str(seed))),
                            "baseline_sha256": digest(baseline_path)})
    files = [{"path": str(p.relative_to(source)), "sha256": digest(p)}
             for p in sorted(source.rglob("*")) if p.is_file()]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
                "entries": entries, "source_files": files, "arms": RUN_ARMS,
                "baseline_source_tree": str(source_tree),
                "baseline_engine_sha256": engine_before,
                "policy_engine_sha256": digest(engine_path),
                "selection": "All-epoch validation BAcc; AUROC selector retained for comparison",
                "control": "Completed nearest3-7 historical control, same seeds, no repeated baseline jobs",
                "evaluation": "Test evaluated only after training; no test-based checkpoint or arm selection",
                "head_first_schedule": "2 head-only + 18 full epochs; shared 20-epoch cosine, no restart",
                "time_limit": "04:00:00", "max_concurrent": 10}
    write_json(campaign / "manifest.json", manifest)
    for item in files:
        (source / item["path"]).chmod(0o444)
    print(json.dumps({"prepared": len(entries), "campaign": str(campaign)}), flush=True)


def run_entry(campaign):
    manifest = verify_source(campaign)
    entry = manifest["entries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = Path(entry["output"])
    if (output / "result.json").exists():
        print("Already completed: " + str(output), flush=True)
        return
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
               "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=tuab-%s-%s" % (os.environ["SLURM_JOB_ID"], entry["index"]),
               "--module", "ablation.tuab.finetune", "--config", entry["config"], "--distributed"]
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
            record = json.loads(ledger.read_text())
            submit_monitor(campaign, record["job_id"])
            print(json.dumps(record, indent=2))
            return
        intent = campaign / "submission_intent.json"
        if intent.exists():
            raise RuntimeError("Submission intent exists; reconcile sacct/squeue before retrying")
        manifest = verify_source(campaign)
        smoke = json.loads((campaign / "smoke_validation.json").read_text())
        if smoke["manifest_sha256"] != digest(campaign / "manifest.json") or not smoke["passed"]:
            raise RuntimeError("This exact snapshot must pass the real-data smoke check")
        name = "tuab-retry-20260915"
        active = subprocess.check_output(["squeue", "-h", "-u", ROOT.owner(), "--name", name, "-o", "%i"], text=True).strip()
        if active:
            raise RuntimeError("Matching jobs already exist: " + active)
        runner = [str(ROOT / ".venv/bin/python"), "-m", "ablation.tuab.campaign", "run-entry", "--campaign", str(campaign)]
        wrapper = ('unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
                   'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; '
                   'export TMPDIR="/tmp/tuab-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"; '
                   'exec srun --ntasks=1 --gpus-per-task=l40s:1 --cpu-bind=none --kill-on-bad-exit=1 ' + shlex.join(runner))
        command = ["sbatch", "--parsable", "--account=system", "--job-name=" + name,
                   "--partition=gl40s_dev,gl40s_short,gl40s_long",
                   "--array=0-%d%%10" % (len(manifest["entries"]) - 1),
                   "--nodes=1", "--ntasks=1", "--gpus-per-task=l40s:1", "--cpus-per-task=8",
                   "--mem=32G", "--time=" + manifest["time_limit"], "--nice=0",
                   "--chdir=" + str(campaign / "source"),
                   "--output=" + str(campaign / "slurm-%A_%a.log"), "--wrap", wrapper]
        write_json(intent, {"command": command, "created_utc": datetime.now(timezone.utc).isoformat()})
        result = subprocess.run(command, capture_output=True, text=True)
        write_json(campaign / "submission_response.json", {"returncode": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Ambiguous sbatch response; inspect submission_response.json before retrying")
        array = "0-%d%%10" % (len(manifest["entries"]) - 1)
        record = {"job_id": job, "tasks": len(manifest["entries"]), "array": array, "arms": RUN_ARMS,
                  "seeds": SEEDS, "command": command,
                  "submitted_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": digest(campaign / "manifest.json")}
        write_json(ledger, record)
        submit_monitor(campaign, job)
        print(json.dumps(record, indent=2), flush=True)


def submit_monitor(campaign, gpu_job):
    ledger = campaign / "monitor_submission.json"
    if ledger.exists():
        return json.loads(ledger.read_text())["job_id"]
    intent = campaign / "monitor_submission_intent.json"
    if intent.exists():
        raise RuntimeError("Monitor submission intent exists; reconcile before retrying")
    runner = [str(ROOT / ".venv/bin/python"), "-m", "ablation.tuab.campaign", "finalize",
              "--campaign", str(campaign)]
    wrapper = ("unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; exec " +
               shlex.join(runner))
    command = ["sbatch", "--parsable", "--account=system", "--job-name=tuab-retry-check",
               "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
               "--mem=4G", "--time=00:15:00", "--dependency=afterany:" + str(gpu_job),
               "--kill-on-invalid-dep=yes", "--chdir=" + str(campaign / "source"),
               "--output=" + str(campaign / "monitor-%j.log"), "--wrap", wrapper]
    write_json(intent, {"command": command, "created_utc": datetime.now(timezone.utc).isoformat()})
    result = subprocess.run(command, capture_output=True, text=True)
    write_json(campaign / "monitor_submission_response.json",
               {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    job = result.stdout.strip().split(";")[0]
    if result.returncode or not job.isdigit():
        raise RuntimeError("Monitor submission failed/ambiguous: " + result.stderr)
    write_json(ledger, {"job_id": job, "gpu_job_id": str(gpu_job), "command": command,
                        "submitted_utc": datetime.now(timezone.utc).isoformat()})
    return job


def finalize(campaign):
    manifest = verify_source(campaign)
    missing, unreadable = [], []
    for entry in manifest["entries"]:
        path = Path(entry["output"]) / "result.json"
        if not path.exists():
            missing.append({"arm": entry["arm"], "seed": entry["seed"]})
            continue
        try:
            json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            unreadable.append({"arm": entry["arm"], "seed": entry["seed"]})
    complete = not missing and not unreadable and len(manifest["entries"]) == len(RUN_ARMS) * len(SEEDS)
    write_json(campaign / "status.json", {"complete": complete, "expected": len(manifest["entries"]),
               "missing": missing, "unreadable": unreadable,
               "checked_utc": datetime.now(timezone.utc).isoformat()})
    if not complete:
        raise RuntimeError("TUAB retry incomplete; inspect status.json")
    subprocess.run([sys.executable, "-m", "ablation.tuab.report", "--campaign", str(campaign)],
                   cwd=campaign / "source", check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "smoke", "submit", "run-entry", "finalize", "all"))
    parser.add_argument("--campaign", type=Path, default=ROOT / DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action in ("prepare", "all"):
        prepare(campaign)
    if args.action in ("verify",):
        verify_source(campaign)
        print("Source and checkpoint hashes verified")
    if args.action in ("smoke", "all") and not (campaign / "submission.json").exists():
        subprocess.run([sys.executable, "-m", "ablation.tuab.smoke", "--campaign", str(campaign)],
                       cwd=campaign / "source", check=True)
    if args.action in ("submit", "all"):
        submit(campaign)
    if args.action == "run-entry":
        run_entry(campaign)
    if args.action == "finalize":
        finalize(campaign)


if __name__ == "__main__":
    main()
