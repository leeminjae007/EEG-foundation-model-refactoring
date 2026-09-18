"""Advance an existing Optuna campaign on a CPU node, independent of SSH."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    command = [manifest["controller_python"], "-m", "ablation.optuna_search.campaign",
               "step", "--campaign", str(campaign)]
    with (campaign / "server_controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another server controller already owns this campaign", flush=True)
            return
        while not (campaign / "STOP_SERVER_CONTROLLER").exists():
            record = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                          job_id=os.environ.get("SLURM_JOB_ID"), pid=os.getpid())
            try:
                result = subprocess.run(command, cwd=campaign / "source", text=True,
                                        capture_output=True, timeout=600, check=True)
                status = json.loads(next(line for line in reversed(result.stdout.splitlines())
                                         if line.startswith("{")))
                record.update(status=status["status"], datasets={slug: {
                    key: value for key, value in ds.items()
                    if key in ("status", "trial", "job_id", "completed_seeds", "error")}
                    for slug, ds in status["datasets"].items()})
                if status["status"] == "complete":
                    # The publisher checks exactly five seeds for every dataset.
                    sys.path.insert(0, manifest["root"])
                    from scripts.monitor_optuna_search import publish
                    publish(args, status)
                    record["published"] = True
                write(campaign / "server_controller_status.json", record)
                print(json.dumps(record), flush=True)
                if status["status"] in ("complete", "finished_with_fixed_settings"):
                    return
            except Exception as exc:
                record.update(status="controller_error", error=repr(exc))
                if isinstance(exc, subprocess.CalledProcessError):
                    record["stderr"] = exc.stderr[-4000:]
                write(campaign / "server_controller_status.json", record)
                print(json.dumps(record), flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
