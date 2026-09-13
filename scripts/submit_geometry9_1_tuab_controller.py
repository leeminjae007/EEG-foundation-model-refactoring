"""Submit the idempotent geometry9-1 -> TUAB verification handoff."""

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/geometry9_1_tuab_dropout01_20260912"
PRETRAIN = ROOT / "outputs/geometry50_reve3cm_t2_15_20260912"
PRETRAIN_JOB = "27393348"
CONTROLLER_JOB_NAME = "geo91-tuab-handoff"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def submit():
    with (CAMPAIGN / ".controller_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = CAMPAIGN / "controller_submission.json"
        if ledger.exists():
            print(ledger.read_text())
            return
        intent = CAMPAIGN / "controller_submission_intent.json"
        if intent.exists():
            raise RuntimeError("prior controller intent exists; reconcile scheduler before retrying")
        controller = ROOT / "scripts/geometry9_1_tuab_controller.py"
        snapshot = CAMPAIGN / "controller.py"
        snapshot.write_bytes(controller.read_bytes())
        snapshot.chmod(0o444)
        job_name = CONTROLLER_JOB_NAME
        active = subprocess.check_output(
            ["squeue", "-h", "-u", str(ROOT.owner()), "--name", job_name, "-o", "%i"], text=True).strip()
        if active:
            raise RuntimeError("matching controller already exists: " + active)
        python = ROOT / ".venv/bin/python"
        environment = (f"GEOMETRY_TUAB_CAMPAIGN={CAMPAIGN} "
                       f"GEOMETRY_PRETRAIN_ROOT={PRETRAIN} "
                       f"GEOMETRY_PRETRAIN_JOB={PRETRAIN_JOB}")
        command = ["sbatch", "--parsable", "--job-name", job_name, "--account", "system",
                   "--partition", "cpu_short,cpu_long", "--dependency", f"afterany:{PRETRAIN_JOB}",
                   "--cpus-per-task", "2", "--mem", "8G", "--time", "00:30:00", "--nice=0",
                   "--chdir", str(CAMPAIGN), "--output", str(CAMPAIGN / "controller-%j.out"),
                   "--error", str(CAMPAIGN / "controller-%j.err"),
                   "--wrap", f"unset PYTHONPATH; export PYTHONNOUSERSITE=1; {environment} {python} {snapshot}"]
        atomic_json(intent, {"command": command})
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            atomic_json(CAMPAIGN / "controller_submission_rejection.json", {
                "command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError(result.stderr)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError("unexpected sbatch output: " + result.stdout)
        atomic_json(ledger, {"controller_job_id": job_id, "dependency": f"afterany:{PRETRAIN_JOB}",
                            "submitted_utc": datetime.now(timezone.utc).isoformat(),
                            "controller_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                            "command": command})
        print(json.dumps({"controller_job_id": job_id}, indent=2))


if __name__ == "__main__":
    submit()
