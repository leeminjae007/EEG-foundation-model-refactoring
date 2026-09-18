"""CPU-only monitor: release a held downstream array after strict pretrain verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def slurm_state(job_id):
    """Return the current accounting state for one parent pretrain job."""
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", str(job_id), "--format=State"],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    for line in result.stdout.splitlines():
        state = line.strip().split("|", 1)[0].split()[0] if line.strip() else ""
        if state:
            return state
    return "UNKNOWN"


TERMINAL_FAILURES = {"CANCELLED", "FAILED", "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--downstream-job", required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    manifest_path = experiment / "manifest.json"
    controller = experiment / "controller/gr2_pretrain_campaign.py"
    if not controller.is_file():
        raise FileNotFoundError("missing isolated pretrain controller: " + str(controller))

    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pretrain_job = manifest.get("pretrain_job") or manifest["pretrain_entries"][0].get("job")
        state = slurm_state(pretrain_job)
        if state in TERMINAL_FAILURES:
            # A held downstream array has no valid checkpoint to wait for once
            # pretraining is terminally unsuccessful.  Cancel it explicitly
            # rather than retaining an inert scheduler entry.
            subprocess.run(["scancel", args.downstream_job], check=False)
            manifest["downstream_state"] = "cancelled"
            manifest["downstream_cancel_reason"] = "pretrain_" + state.lower()
            manifest["downstream_cancelled_for_pretrain_job"] = str(pretrain_job)
            write_manifest(manifest_path, manifest)
            print("cancelled downstream job %s after pretrain %s (%s)" %
                  (args.downstream_job, pretrain_job, state), flush=True)
            return
        # The controller is the sole authority for strict checkpoint loading,
        # config matching, optimizer/scheduler and four-rank RNG validation.
        result = subprocess.run([sys.executable, str(controller), "step", "--folder", str(experiment)])
        if result.returncode == 0:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["pretrain_entries"][0]
            if entry.get("verified_complete"):
                subprocess.run(["scontrol", "release", args.downstream_job], check=True)
                manifest["downstream_state"] = "released"
                manifest["downstream_released_after_checkpoint"] = entry["checkpoint_sha256"]
                write_manifest(manifest_path, manifest)
                print("released downstream job " + args.downstream_job, flush=True)
                return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
