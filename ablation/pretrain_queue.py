"""CPU-only backfill/requeue controller restricted to the nine A100 pretrains."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

JOBS = {
    "27508289": "encoder_mjde_mix1only", "27512241": "pe_none", "27512242": "pe_channel_id",
    "27512243": "pe_acpe", "27512244": "encoder_mjde_s2t6", "27512245": "encoder_mjde_t2s6",
    "27512246": "encoder_mjde_average", "27512247": "encoder_csbrain", "27512248": "encoder_mjde_lite",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def call(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=45, check=True).stdout


def job_info(job, original=None):
    result = subprocess.run(["scontrol", "show", "job", job, "-o"], capture_output=True, text=True, timeout=45)
    if result.returncode == 0 and result.stdout.strip():
        return dict(re.findall(r"(\w+)=(\S+)", result.stdout))
    if original is not None:
        text = call(["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State,Elapsed,Start,End,Timelimit"])
        rows = [line.split("|") for line in text.splitlines() if line.split("|")[0] == job]
        if rows:
            row = rows[-1]
            return dict(original, JobState=row[1].split()[0], RunTime=row[2], StartTime=row[3],
                        EndTime=row[4], TimeLimit=row[5], Reason="accounting_record")
    raise ValueError("Job no longer available to scontrol: " + job)


def check_owned_pretrain(job, fields):
    if job not in JOBS or fields["JobId"] != job or not fields["UserId"].startswith("ml10266("):
        raise ValueError("Outside the authorized pretrain jobs")
    if fields["NumTasks"] != "4" or "gres/gpu:a100=4" not in fields["ReqTRES"]:
        raise ValueError("Keep four A100 ranks unchanged")
    if not set(fields["Partition"].split(",")) <= {"a100_short", "a100_long"}:
        raise ValueError("Unexpected partition change")


def configure(folder):
    if (folder / "manifest.json").exists():
        raise FileExistsError("Scheduling profile already exists")
    initial = {job: job_info(job) for job in JOBS}
    for job, fields in initial.items():
        check_owned_pretrain(job, fields)
    write(folder / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=JOBS,
          time_min="04:00:00", time_limit="1-00:00:00", max_continuations=10, original=initial,
          scope="Only the listed A100 pretrains; no downstream or Optuna changes"))
    changes = []
    for job in JOBS:
        current = job_info(job)
        check_owned_pretrain(job, current)
        if current["JobState"] == "PENDING":
            call(["scontrol", "update", "JobId=" + job, "TimeMin=04:00:00"])
            after = job_info(job)
            if after["TimeMin"] != "04:00:00" or after["EligibleTime"] != current["EligibleTime"]:
                raise ValueError("TimeMin/queue age differs from requested change")
            changes.append(dict(job=job, before=current, after=after))
    write(folder / "scheduling_changes.json", changes)
    return dict(configured=len(changes), job_ids=list(JOBS))


def checkpoint_info(path):
    import torch
    torch.set_num_threads(2)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("extra", {}).get("partial_epoch_smoke") or len(saved["rng_states"]) != 4:
        raise ValueError("Checkpoint is not a completed four-rank pretrain epoch")
    if not all(torch.isfinite(x).all().item() for x in saved["model"].values() if torch.is_floating_point(x)):
        raise ValueError("Nonfinite checkpoint weights")
    result = dict(epoch=saved["epoch"], target_epochs=saved["config"]["optimization"]["epochs"],
                  step=saved["extra"]["step"], arm=saved["config"]["ablation"]["name"],
                  seed=saved["config"]["seed"], world_size=len(saved["rng_states"]), finite_weights=True)
    if not 1 <= result["epoch"] <= result["target_epochs"]:
        raise ValueError("Checkpoint epoch is outside the training schedule")
    return result


def continuation_needed(fields, now=None):
    if fields["JobState"] == "TIMEOUT":
        return True
    if fields["JobState"] != "RUNNING" or fields.get("EndTime") in (None, "Unknown"):
        return False
    end = datetime.fromisoformat(fields["EndTime"])
    seconds = (end - (now or datetime.now())).total_seconds()
    return 0 < seconds <= 240


def advance(folder):
    import fcntl
    root = Path(__file__).resolve().parents[1]
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(folder / "manifest.json")
        histories = read(folder / "continuations.json") if (folder / "continuations.json").exists() else {}
        jobs = {}
        for job, arm in manifest["jobs"].items():
            output = root / "outputs/ablation" / arm / "seed42"
            try:
                fields = job_info(job, manifest["original"][job])
                check_owned_pretrain(job, fields)
                row = dict(arm=arm, state=fields["JobState"], reason=fields["Reason"],
                           start=fields["StartTime"], end=fields["EndTime"], runtime=fields["RunTime"],
                           time_limit=fields["TimeLimit"], time_min=fields["TimeMin"], nodes=fields.get("NodeList", ""),
                           continuations=len(histories.get(job, [])))
                final = output / "checkpoint-epoch-0040.pth"
                if fields["JobState"] == "COMPLETED":
                    verified = folder / ("verified-" + job + ".json")
                    if verified.exists() and read(verified)["mtime_ns"] == final.stat().st_mtime_ns:
                        info = read(verified)
                    else:
                        info = dict(checkpoint_info(final), mtime_ns=final.stat().st_mtime_ns)
                        if info["epoch"] != 40 or info["arm"] != arm or info["seed"] != 42:
                            raise ValueError("Final checkpoint identity mismatch")
                        write(verified, info)
                    row["verified"] = info
                elif continuation_needed(fields):
                    last = output / "last.pth"
                    info = checkpoint_info(last)
                    if info["arm"] != arm or info["seed"] != 42:
                        raise ValueError("Resume checkpoint identity mismatch")
                    row["checkpoint"] = info
                    if info["epoch"] < info["target_epochs"]:
                        history = histories.setdefault(job, [])
                        if len(history) >= manifest["max_continuations"]:
                            raise ValueError("Continuation budget exceeded")
                        if history and info["epoch"] <= history[-1]["epoch"]:
                            raise ValueError("No completed-epoch progress since previous continuation")
                        intent = folder / ("requeue-" + job + "-" + str(len(history)) + ".json")
                        if intent.exists():
                            raise ValueError("Ambiguous requeue; reconcile " + str(intent))
                        write(intent, dict(job=job, **info, utc=datetime.now(timezone.utc).isoformat()))
                        call(["scontrol", "requeue", job])
                        history.append(dict(epoch=info["epoch"], utc=datetime.now(timezone.utc).isoformat()))
                        write(folder / "continuations.json", histories)
                        current = job_info(job)
                        if current["JobState"] == "PENDING":
                            call(["scontrol", "update", "JobId=" + job, "TimeLimit=1-00:00:00", "TimeMin=04:00:00"])
                        # Reset the pretrain-only verification dependency after a timeout.
                        if job == "27508289":
                            dependent = job_info("27508310")
                            if dependent["JobState"] == "PENDING":
                                call(["scontrol", "update", "JobId=27508310", "Dependency=afterok:" + job])
                        row["state"] = "REQUEUED"
                        row["continuations"] = len(history)
                    elif fields["JobState"] == "TIMEOUT":
                        row["needs_attention"] = True
                        row["error"] = "Final epoch saved but job timed out; verify final checkpoint and dependent job before completion"
                elif fields["JobState"] not in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
                    row["needs_attention"] = True
                jobs[job] = row
            except Exception as exc:
                jobs[job] = dict(arm=arm, state="needs_attention", error=repr(exc))
        status = dict(checked_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
                      complete=all(j.get("verified") for j in jobs.values()))
        write(folder / "status.json", status)
        return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("configure", "step"))
    parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(configure(args.folder) if args.action == "configure" else advance(args.folder)))
