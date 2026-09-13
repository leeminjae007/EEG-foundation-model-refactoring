"""Verify geometry epoch 40, then submit its immutable TUAB five-seed array."""

import fcntl
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path("/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model")
CAMPAIGN = Path(os.environ.get(
    "GEOMETRY_TUAB_CAMPAIGN", ROOT / "outputs/geometry9_1_tuab_dropout01_20260912"))
PRETRAIN = Path(os.environ.get(
    "GEOMETRY_PRETRAIN_ROOT", ROOT / "outputs/geometry50_reve3cm_t2_15_20260912"))
CHECKPOINT = PRETRAIN / "training/checkpoint-epoch-0040.pth"
PRETRAIN_JOB = os.environ.get("GEOMETRY_PRETRAIN_JOB", "27393348")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parent_state():
    output = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", PRETRAIN_JOB, "--format=JobIDRaw,State"], text=True)
    for line in output.splitlines():
        job_id, state = line.split("|", 1)
        if job_id == PRETRAIN_JOB:
            return state.split()[0].split("+")[0]
    raise RuntimeError(f"cannot resolve pretrain job {PRETRAIN_JOB}")


def verify_checkpoint():
    state = parent_state()
    if state != "COMPLETED":
        raise RuntimeError(f"pretrain {PRETRAIN_JOB} ended as {state}; downstream not submitted")
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"missing epoch-40 checkpoint: {CHECKPOINT}")
    payload = torch.load(CHECKPOINT, map_location="cpu")
    expected = {
        "policy": "geometry_tubelet", "mask_ratio": 0.5, "distance_metric": "euclidean_m",
        "radius_m": 0.03, "min_time_patches": 2, "max_time_patches": 15,
    }
    if payload.get("epoch") != 40:
        raise RuntimeError(f"checkpoint epoch is {payload.get('epoch')}, expected 40")
    if payload["config"]["masking"] != expected:
        raise RuntimeError("checkpoint masking is not the approved geometry-only GR9-1 setting")
    if payload.get("extra", {}).get("partial_epoch_smoke"):
        raise RuntimeError("checkpoint is marked as a partial smoke epoch")
    state_dict = payload["model"]
    if not state_dict or not all(key.startswith(("backbone.", "decoder.")) for key in state_dict):
        raise RuntimeError("checkpoint state_dict does not match the pretrain model schema")
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "pretrain_job_id": PRETRAIN_JOB,
              "job_state": state, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
              "epoch": 40, "masking": expected, "partial_epoch_smoke": False}
    atomic_json(CAMPAIGN / "checkpoint_verification.json", report)
    return report


def submit_array(verification):
    ledger = CAMPAIGN / "submission.json"
    with (CAMPAIGN / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger.exists():
            print(ledger.read_text())
            return
        if (CAMPAIGN / "submission_intent.json").exists():
            raise RuntimeError("prior submission intent exists; reconcile scheduler before retrying")
        job_name = "geo91-tuab-d01"
        active = subprocess.check_output(
            ["squeue", "-h", "-u", str(ROOT.owner()), "--name", job_name, "-o", "%i"], text=True).strip()
        if active:
            raise RuntimeError("matching downstream job already exists: " + active)
        source = CAMPAIGN / "source"
        manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
        for item in manifest["source_files"]:
            if sha256(source / item["path"]) != item["sha256"]:
                raise RuntimeError("immutable source checksum mismatch: " + item["path"])
        python = ROOT / ".venv/bin/python"
        wrapper = ("unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 "
                   "OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; "
                   f'exec srun --ntasks=1 --gpus-per-task=l40s:1 --cpu-bind=none --kill-on-bad-exit=1 '
                   f'{python} {source / "run_array.py"}')
        command = ["sbatch", "--parsable", "--job-name", job_name, "--account", "system",
                   "--partition", "gl40s_dev,gl40s_short,gl40s_long", "--array", "0-4%3",
                   "--nodes", "1", "--ntasks", "1", "--gpus-per-task", "l40s:1",
                   "--cpus-per-task", "8", "--mem", "32G", "--time", "1-00:00:00", "--nice=0",
                   "--chdir", str(source), "--output", str(CAMPAIGN / "slurm-%A_%a.log"),
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
        atomic_json(ledger, {"job_id": job_id, "array": "0-4%3", "tasks": 5,
                            "submitted_utc": datetime.now(timezone.utc).isoformat(),
                            "checkpoint_verification": verification, "command": command})
        print(json.dumps({"downstream_job_id": job_id}, indent=2))


if __name__ == "__main__":
    try:
        submit_array(verify_checkpoint())
    except Exception as error:
        atomic_json(CAMPAIGN / "controller_failure.json", {
            "failed_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error)})
        raise
