"""Independent three-arm, decoder-2 Full-MJDE L40S pretraining campaign."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = ROOT
if not (BASELINE_SOURCE / "configs/pretrain_gr2_geometry.yaml").is_file():
    BASELINE_SOURCE = ROOT / "outputs/mjde_gr2_geometry_20260917_171816/source"
ARMS = (
    ("static", "Static", "static_feature"),
    ("patch_scalar", "Patch scalar", "patch_scalar"),
    ("patch_dimension", "Patch dimension", "patch_feature"),
)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}
RESUMABLE = {"TIMEOUT", "PREEMPTED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(folder):
    return json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))


def copy_source(source, destination):
    ignored = shutil.ignore_patterns(".git", ".venv", "outputs", "__pycache__", ".pytest_cache", "Data")
    shutil.copytree(source, destination, ignore=ignored, dirs_exist_ok=False)


def validate_config(config, arm, mode):
    assert config["mae"]["decoder_dim"] == 100
    assert config["mae"]["decoder_depth"] == 2
    assert config["mae"]["decoder_heads"] == 4
    assert config["encoder"]["embed_dim"] == 200
    assert config["encoder"]["fusion_gate"] == mode
    assert config["optimization"]["epochs"] == 40
    assert config["optimization"]["batch_size_per_gpu"] == 128
    assert config["data"]["num_workers"] == 8
    assert config["runtime"]["pretrain_rank_rng"] == "independent"
    assert config["masking"]["policy"] == "geometry_tubelet"
    assert config["masking"]["mask_ratio"] == 0.5
    assert config["masking"]["min_radius_degrees"] == 35.0
    assert config["masking"]["max_radius_degrees"] == 75.0
    assert config["masking"]["min_time_patches"] == 2
    assert config["masking"]["max_time_patches"] == 10
    expected = {"static": "static_feature", "patch_scalar": "patch_scalar", "patch_dimension": "patch_feature"}
    assert expected[arm] == mode


def prepare(folder):
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError("Campaign folder already exists: " + str(folder))
    base_config = BASELINE_SOURCE / "configs/pretrain_gr2_geometry.yaml"
    if not base_config.is_file():
        raise FileNotFoundError("Missing frozen GR2 baseline config: " + str(base_config))
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    folder.mkdir(parents=True)
    copy_source(BASELINE_SOURCE, folder / "source")
    controller = folder / "controller"
    controller.mkdir()
    shutil.copy2(Path(__file__).resolve(), controller / "gr2_decoder2_l40s_campaign.py")
    entries = []
    for index, (arm, label, mode) in enumerate(ARMS):
        config = copy.deepcopy(base)
        config["encoder"]["fusion_gate"] = mode
        config["mae"]["decoder_depth"] = 2
        config["runtime"]["output"] = str(folder / "pretrain" / arm)
        config["runtime"]["pretrain_rank_rng"] = "independent"
        validate_config(config, arm, mode)
        config_path = folder / "configs" / (arm + ".yaml")
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entries.append({
            "index": index, "arm": arm, "label": label, "fusion_gate": mode,
            "config": str(config_path), "config_sha256": sha256(config_path),
            "source": str(folder / "source"), "result_dir": str(folder / "pretrain" / arm),
            "job": None, "job_history": [], "retries": 0, "last_resume_epoch": 0,
            "max_timeout_resumes": 40, "time_limit": "04:00:00",
            "gpu_partitions": "gl40s_dev,gl40s_short,gl40s_long",
        })
    worker = folder / "worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1\n"
        'export TMPDIR="${SLURM_TMPDIR:-/tmp}/gr2d2-${SLURM_JOB_ID}"\nmkdir -p "$TMPDIR"\n'
        + str(ROOT / ".venv/bin/python") + " " + str(controller / "gr2_decoder2_l40s_campaign.py")
        + " worker --folder " + str(folder) + "\n",
        encoding="utf-8",
    )
    worker.chmod(0o750)
    manifest = {
        "alias": "gr2-decoder2-l40s-gates", "created_utc": now(),
        "python": str(ROOT / ".venv/bin/python"), "pretrain_launcher": "slurm_flexible_l40s",
        "gpu_resource": "l40s", "partitions": "gl40s_dev,gl40s_short,gl40s_long",
        "nodes": "1-4", "tasks": 4, "cpus_per_task": 4, "memory_per_gpu": "32G",
        "time_limit": "04:00:00", "seed": 42, "target_epochs": 40,
        "pretrain_entries": entries, "monitor_job": None,
        "results_dir": str(ROOT / "outputs/results/pretrain" / folder.name),
        "downstream_submitted": False,
        "resource_policy": {"cpu_total": 16, "ram_total_gib": 128, "workers_per_gpu": 8,
                            "measured_not_inferred": True},
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def slurm_state(job):
    if not job:
        return "NOT_SUBMITTED"
    text = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", str(job),
                                    "--format=JobID,State"], text=True)
    for line in text.splitlines():
        fields = line.split("|")
        if fields and fields[0] == str(job):
            return fields[1].split()[0]
    return "UNKNOWN"


def submit(folder, manifest, index):
    entry = manifest["pretrain_entries"][index]
    logs = Path(folder) / "logs"
    logs.mkdir(exist_ok=True)
    cmd = ["sbatch", "--parsable", "--account=system", "--job-name=gr2d2-" + entry["arm"],
           "--partition=" + entry["gpu_partitions"], "--nodes=1-4", "--ntasks=4",
           "--gpus-per-task=l40s:1", "--cpus-per-task=4", "--mem-per-gpu=32G",
           "--time=" + entry["time_limit"], "--export=ALL,GR2D2_INDEX=" + str(index),
           "--output=" + str(logs / "%x-%j.out"), "--error=" + str(logs / "%x-%j.err"),
           str(Path(folder) / "worker.sh")]
    job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    entry["job"] = job
    entry.setdefault("job_history", []).append(job)
    entry.pop("continuation_for_job", None)
    entry.pop("continuation_job", None)
    write_json(Path(folder) / "manifest.json", manifest)
    attach_callback(folder, manifest, entry)
    return job


def attach_callback(folder, manifest, entry):
    job = entry["job"]
    command = [manifest["python"], str(Path(folder) / "controller/gr2_decoder2_l40s_campaign.py"),
               "continue", "--folder", str(folder), "--job", job]
    callback = subprocess.check_output([
        "sbatch", "--parsable", "--account=system", "--job-name=gr2d2-resume-" + entry["arm"],
        "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=4G",
        "--time=00:15:00", "--dependency=afterany:" + job,
        "--output=" + str(Path(folder) / "logs/continuation-%j.out"),
        "--error=" + str(Path(folder) / "logs/continuation-%j.err"),
        "--wrap=exec " + " ".join(command),
    ], text=True).strip().split(";")[0]
    entry["continuation_for_job"] = job
    entry["continuation_job"] = callback
    entry.setdefault("continuation_history", []).append(callback)
    write_json(Path(folder) / "manifest.json", manifest)
    return callback


def launch_audit(entry, rank):
    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    validate_config(config, entry["arm"], entry["fusion_gate"])
    expected_shape = {"static": [3, 200], "patch_scalar": [1], "patch_dimension": [200]}[entry["arm"]]
    return {"checked_utc": now(), "arm": entry["arm"], "config_sha256": sha256(entry["config"]),
            "decoder_depth": config["mae"]["decoder_depth"], "fusion_gate": entry["fusion_gate"],
            "expected_gate_shape": expected_shape, "rank": rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def rank_worker(folder, manifest):
    index = int(os.environ["GR2D2_INDEX"])
    entry = manifest["pretrain_entries"][index]
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if world != 4 or not visible or "," in visible or visible == "-1":
        raise RuntimeError("Expected four ranks with exactly one allocated L40S GPU per rank")
    if sha256(entry["config"]) != entry["config_sha256"]:
        raise RuntimeError("Frozen campaign config changed")
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    result = Path(entry["result_dir"])
    result.mkdir(parents=True, exist_ok=True)
    audit = launch_audit(entry, rank)
    probe = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
                            "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=20)
    audit.update(nvidia_smi=probe.stdout, nvidia_smi_exit_code=probe.returncode)
    import torch
    torch.cuda.set_device(0)
    torch.ones(1, device="cuda:0").sum().item()
    torch.cuda.synchronize(0)
    audit["cuda_probe"] = "passed"
    write_json(result / ("launch-audit-" + os.environ["SLURM_JOB_ID"] + "-rank" + str(rank) + ".json"), audit)
    os.chdir(entry["source"])
    command = [manifest["python"], "-m", "ablation.pretrain", "--config", entry["config"],
               "--distributed", "--output", entry["result_dir"]]
    last = result / "last.pth"
    if last.exists():
        command += ["--resume", str(last)]
    os.execv(command[0], command)


def worker(folder, manifest):
    index = int(os.environ["GR2D2_INDEX"])
    hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()
    os.environ.update(MASTER_ADDR=hosts[0], MASTER_PORT=str(15000 + int(os.environ["SLURM_JOB_ID"]) % 40000),
                      SLURM_EXPORT_ENV="ALL", NCCL_ASYNC_ERROR_HANDLING="1")
    command = ["srun", "--nodes=" + os.environ["SLURM_JOB_NUM_NODES"], "--ntasks=4", "--gpus-per-task=l40s:1",
               "--gpu-bind=single:1", "--distribution=block", "--kill-on-bad-exit=1", "--export=ALL,GR2D2_INDEX=" + str(index),
               manifest["python"], str(Path(__file__).resolve()), "rank-worker", "--folder", str(folder)]
    os.execvp(command[0], command)


def validate_final(entry):
    import torch
    sys.path.insert(0, entry["source"])
    from ablation.bootstrap import ensure_data_imports
    ensure_data_imports()
    from ablation.models import build_pretrain
    from ablation.pretrain_resume import validate_resume
    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    final = Path(entry["result_dir"]) / "checkpoint-epoch-0040.pth"
    saved = torch.load(final, map_location="cpu", weights_only=False)
    validate_resume(saved, config, 4)
    if saved["epoch"] != 40 or saved["config"] != config:
        raise ValueError("Final checkpoint/config mismatch")
    model = build_pretrain(config, torch.device("cpu"))
    model.load_state_dict(saved["model"], strict=True)
    if len(saved.get("rng_states", [])) != 4:
        raise ValueError("Missing four-rank RNG state")
    for key in ("torch", "cuda"):
        if len({state[key].numpy().tobytes() for state in saved["rng_states"]}) != 4:
            raise ValueError("Rank RNG states are not independent: " + key)
    return {"checkpoint": str(final), "checkpoint_sha256": sha256(final), "optimizer_steps": saved["extra"]["step"],
            "dataset_fingerprint": saved["extra"]["dataset_fingerprint"], "strict_model_load": True,
            "decoder_depth": config["mae"]["decoder_depth"], "fusion_gate": entry["fusion_gate"]}


def metrics_and_mask(entry):
    metrics = Path(entry["result_dir"]) / "metrics.jsonl"
    if not metrics.exists():
        return None, None
    with metrics.open("rb") as stream:
        stream.seek(max(0, metrics.stat().st_size - 500000))
        rows = []
        for line in stream.read().decode("utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    progress = rows[-1] if rows else None
    mask = next(({"target_tokens": row["masking/target_tokens"], "context_tokens": row["masking/context_tokens"]}
                 for row in rows if "masking/target_tokens" in row and "masking/context_tokens" in row), None)
    return progress, mask


def resource_record(job):
    try:
        text = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
                                        "--format=JobID,State,Elapsed,AllocTRES,MaxRSS,MaxVMSize"], text=True)
        return {"sacct": [line for line in text.splitlines() if line]}
    except subprocess.CalledProcessError as error:
        return {"sacct_error": repr(error)}


def publish_arm(folder, manifest, entry, final_report, mask):
    destination = Path(manifest["results_dir"]) / entry["arm"]
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry["config"], destination / "config.yaml")
    payload = {"alias": manifest["alias"], "label": entry["label"], "arm": entry["arm"],
               "jobs": entry["job_history"], "continuations": entry.get("continuation_history", []),
               "resource_policy": manifest["resource_policy"], "resource_measurements": entry.get("resource_measurements", []),
               "mask_audit": mask, "downstream_evaluated": False, **final_report}
    write_json(destination / "pretrain_result.json", payload)
    write_json(destination / "resource_measurements.json", payload["resource_measurements"])


def monitor_step(folder):
    folder = Path(folder)
    import fcntl
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_manifest(folder)
        rows = []
        for entry in manifest["pretrain_entries"]:
            state = slurm_state(entry["job"])
            progress, mask = metrics_and_mask(entry)
            if mask is not None:
                if mask["target_tokens"] != 285.0 or mask["context_tokens"] != 285.0:
                    raise ValueError("Unexpected mask count for " + entry["arm"] + ": " + repr(mask))
                write_json(Path(entry["result_dir"]) / "mask_audit.json", mask)
            entry.setdefault("resource_measurements", []).append({"checked_utc": now(), "job": entry["job"],
                                                                     **resource_record(entry["job"])})
            final = Path(entry["result_dir"]) / "checkpoint-epoch-0040.pth"
            complete = False
            error = None
            if final.is_file():
                try:
                    report = validate_final(entry)
                    publish_arm(folder, manifest, entry, report, mask)
                    entry["verified_complete"] = True
                    complete = True
                except Exception as exc:
                    error = repr(exc)
            elif state in RESUMABLE:
                try:
                    import torch
                    if entry["retries"] >= entry["max_timeout_resumes"]:
                        raise ValueError("Continuation limit reached")
                    saved = torch.load(Path(entry["result_dir"]) / "last.pth", map_location="cpu", weights_only=False)
                    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
                    if saved["config"] != config or saved.get("extra", {}).get("partial_epoch_smoke"):
                        raise ValueError("Unsafe resume checkpoint")
                    if saved["epoch"] <= entry.get("last_resume_epoch", 0) or saved["epoch"] >= 40:
                        raise ValueError("No committed epoch progress for resume")
                    if len(saved.get("rng_states", [])) != 4 or any(k not in saved for k in ("model", "optimizer", "scheduler")):
                        raise ValueError("Incomplete four-rank resume state")
                    entry["retries"] += 1
                    entry["last_resume_epoch"] = saved["epoch"]
                    submit(folder, manifest, entry["index"])
                    state = "RESUBMITTED"
                except Exception as exc:
                    error = repr(exc)
            elif state in {"FAILED", "CANCELLED"}:
                error = "Terminal state is not automatically retried: " + state
            elif state == "COMPLETED" and not final.is_file():
                error = "Completed without epoch-40 checkpoint; not retried"
            rows.append({"arm": entry["arm"], "job": entry["job"], "state": state, "complete": complete,
                         "progress": ({k: progress[k] for k in ("epoch", "step", "loss", "lr_used") if k in progress}
                                      if progress else None), "mask_audit": mask, "error": error})
        write_json(folder / "manifest.json", manifest)
        payload = {"checked_utc": now(), "rows": rows, "complete": all(row["complete"] for row in rows),
                   "downstream_submitted": False}
        write_json(folder / "status.json", payload)
        return payload


def continue_after(folder, job):
    for _ in range(30):
        manifest = read_manifest(folder)
        if job not in {entry.get("job") for entry in manifest["pretrain_entries"]}:
            return {"superseded_job": job}
        if slurm_state(job) not in ACTIVE:
            return monitor_step(folder)
        time.sleep(10)
    return {"deferred_to_hourly_monitor": job}


def submit_monitor(folder, manifest):
    command = [manifest["python"], str(Path(folder) / "controller/gr2_decoder2_l40s_campaign.py"),
               "monitor", "--folder", str(folder)]
    job = subprocess.check_output([
        "sbatch", "--parsable", "--account=system", "--job-name=gr2d2-monitor", "--partition=cpu_long",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=28-00:00:00",
        "--output=" + str(Path(folder) / "logs/controller-%j.out"),
        "--error=" + str(Path(folder) / "logs/controller-%j.err"),
        "--wrap=exec " + " ".join(command),
    ], text=True).strip().split(";")[0]
    manifest["monitor_job"] = job
    write_json(Path(folder) / "manifest.json", manifest)
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "worker", "rank-worker", "continue", "step", "monitor"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(folder), indent=2))
        return
    manifest = read_manifest(folder)
    if args.action == "submit":
        jobs = [submit(folder, manifest, entry["index"]) for entry in manifest["pretrain_entries"] if not entry["job"]]
        manifest = read_manifest(folder)
        monitor = submit_monitor(folder, manifest) if not manifest.get("monitor_job") else manifest["monitor_job"]
        print(json.dumps({"jobs": jobs, "monitor": monitor}))
    elif args.action == "worker":
        worker(folder, manifest)
    elif args.action == "rank-worker":
        rank_worker(folder, manifest)
    elif args.action == "continue":
        if not args.job:
            parser.error("continue requires --job")
        print(json.dumps(continue_after(folder, args.job)))
    else:
        while True:
            status = monitor_step(folder)
            print(json.dumps(status), flush=True)
            if status["complete"] or args.action == "step":
                return
            time.sleep(3600)


if __name__ == "__main__":
    main()
