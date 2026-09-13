"""Retry geometry-only GR9-1 with a short node-local multiprocessing path."""

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry1"
RETRY_OF = "27393348"
JOB_NAME = "eeg-geo50-r3-retry1"
NUM_WORKERS = None
RUNTIME_FIX = "short TMPDIR plus torch multiprocessing file_system sharing strategy"
EXCLUDE = "a100-4004,a100-4007,a100-4011,a100-4012,a100-4021,a100-4023,a100-4028,a100-4032,a100-4033,a100-4037,a100-4042,a100-4044,a100-4045"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit():
    if ROOT.name != "EEG-founation-model":
        raise RuntimeError(f"refusing to launch outside the uppercase refactor repository: {ROOT}")
    CAMPAIGN.mkdir(exist_ok=True)
    with (CAMPAIGN / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = CAMPAIGN / "submission.json"
        if ledger.exists():
            print(ledger.read_text())
            return
        if (CAMPAIGN / "submission_intent.json").exists():
            raise RuntimeError("prior submission intent exists; reconcile scheduler before retrying")
        source = CAMPAIGN / "source"
        if source.exists():
            raise RuntimeError("source snapshot exists without submission ledger; inspect manually")
        source.mkdir()
        shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pretrain.py", source / "pretrain.py")
        (source / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/check_pretrain_initialization.py",
                     source / "scripts/check_pretrain_initialization.py")
        shutil.copy2(ROOT / "scripts/check_pretrain_loader.py",
                     source / "scripts/check_pretrain_loader.py")
        (source / "configs").mkdir()
        shutil.copy2(ROOT / "configs/gr9_1.yaml", source / "configs/gr9_1.yaml")

        config = yaml.safe_load((ROOT / "configs/pretrain.yaml").read_text())
        expected_masking = {"policy": "geometry_tubelet", "mask_ratio": 0.5,
                            "distance_metric": "euclidean_m", "radius_m": 0.03,
                            "min_time_patches": 2, "max_time_patches": 15}
        assert config["masking"] == expected_masking
        assert config["optimization"]["epochs"] == 40
        assert config["optimization"]["batch_size_per_gpu"] == 128
        assert config["seed"] == 42
        if NUM_WORKERS is not None:
            config["data"]["num_workers"] = NUM_WORKERS
        training = CAMPAIGN / "training"
        training.mkdir()
        config["runtime"]["output"] = str(training)
        config_path = source / "configs/pretrain.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        python = ROOT / ".venv/bin/python"
        launch = source / "launch.sh"
        launch.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
cd "{source}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NCCL_DEBUG=WARN
# BigPurple can otherwise derive an AF_UNIX socket below the long campaign path.
export TMPDIR="/tmp/eeggeo-${{SLURM_JOB_ID}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
"{python}" -c 'import tempfile; p=tempfile.gettempdir(); assert len(p) < 40, p; print("TMPDIR", p, flush=True)'
"{python}" scripts/check_pretrain_loader.py --config configs/pretrain.yaml
"{python}" scripts/check_pretrain_initialization.py --config configs/pretrain.yaml --baseline configs/gr9_1.yaml --output "{CAMPAIGN / 'initialization_gpu.json'}"
exec "{python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 pretrain.py --config configs/pretrain.yaml --distributed
''')
        files = [path for path in source.rglob("*") if path.is_file()]
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "retry_of_job": RETRY_OF,
            "repository_root": str(ROOT),
            "scientific_change_from_failed_attempt": "none",
            "runtime_fix": RUNTIME_FIX,
            "loader_preflight": "one real LMDB batch with configured DataLoader workers before GPU initialization",
            "config": config,
            "global_batch": 512,
            "source_files": [{"path": str(path.relative_to(source)), "sha256": sha256(path)}
                             for path in sorted(files)],
        }
        atomic_json(CAMPAIGN / "manifest.json", manifest)
        for path in files:
            path.chmod(0o444)

        job_name = JOB_NAME
        active = subprocess.check_output(
            ["squeue", "-h", "-u", str(ROOT.owner()), "--name", job_name, "-o", "%i"], text=True).strip()
        if active:
            raise RuntimeError("matching retry already exists: " + active)
        wrapper = (f'srun --ntasks=1 --gpus-per-task=a100:4 --cpu-bind=none '
                   f'--kill-on-bad-exit=1 bash {launch}')
        command = ["sbatch", "--parsable", "--job-name", job_name, "--account", "system",
                   "--partition", "a100_short,a100_long", "--nodes", "1", "--ntasks", "1",
                   "--gpus-per-task", "a100:4", "--cpus-per-task", "32", "--mem", "128G",
                   "--time", "1-00:00:00", "--nice=0", "--exclude", EXCLUDE,
                   "--chdir", str(source), "--output", str(CAMPAIGN / "slurm-%j.log"),
                   "--wrap", wrapper]
        atomic_json(CAMPAIGN / "submission_intent.json", {"command": command})
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            atomic_json(CAMPAIGN / "submission_rejection.json", {
                "command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError(result.stderr)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError("unexpected sbatch output: " + result.stdout)
        atomic_json(ledger, {"job_id": job_id, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                            "retry_of_job": RETRY_OF, "command": command,
                            "manifest": str(CAMPAIGN / "manifest.json"),
                            "training_output": str(training)})
        print(json.dumps({"job_id": job_id}, indent=2))


if __name__ == "__main__":
    submit()
