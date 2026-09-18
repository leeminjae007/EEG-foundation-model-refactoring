"""Release only the recorded user holds after the designated Slurm job starts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def fields(job):
    text = subprocess.check_output(["scontrol", "show", "job", "-o", str(job)], text=True)
    values = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9/:]*)=(\S*)", text))
    if values.get("JobId") != str(job) or values.get("UserId", "").split("(")[0] != "ml10266":
        raise ValueError("Unexpected job identity: " + str(job))
    return values


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def release(gate_path):
    import fcntl
    with gate_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = json.loads(gate_path.read_text())
        if gate.get("complete"):
            return gate
        target = fields(gate["target_job"])
        gate["checked_utc"] = datetime.now(timezone.utc).isoformat()
        gate["target_state"] = target["JobState"]
        # Slurm after: also fires on cancellation. A cancelled/failed target must
        # not accidentally release these holds without a successful start check.
        if target["JobState"] not in ("RUNNING", "COMPLETING", "COMPLETED"):
            gate["attention"] = "Target has not entered an eligible started state; holds retained"
            save(gate_path, gate)
            raise RuntimeError(gate["attention"])
        for entry in gate["held_entries"]:
            if entry.get("released") or entry.get("skipped"):
                continue
            current = fields(entry["job"])
            if current.get("Command") != entry["before"]["Command"]:
                raise ValueError("Job command changed: " + entry["job"])
            if current["JobState"] != "PENDING":
                entry["skipped"] = "Current state: " + current["JobState"]
            elif current.get("Reason") == "JobHeldUser" and current.get("Priority") == "0":
                subprocess.run(["scontrol", "release", entry["job"]], check=True)
                entry["released"] = True
                entry["after"] = fields(entry["job"])
            else:
                entry["skipped"] = "No longer held by user; no mutation needed"
            save(gate_path, gate)
        gate["complete"] = True
        gate["released_after_target_started"] = True
        save(gate_path, gate)
        return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(release(args.gate.resolve()), indent=2))
