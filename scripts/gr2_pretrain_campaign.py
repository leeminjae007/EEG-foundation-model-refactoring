"""Pretrain-only GR2 geometry campaign; frozen verified Slurm/GPU-health launcher."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import yaml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def submit(folder, manifest, kind, indices=None):
    if kind != "pretrain":
        raise ValueError("This campaign only submits pretraining")
    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    cmd = ["sbatch", "--parsable", "--account=system",
           "--output=" + str(logs / "%x-%A_%a.out"), "--error=" + str(logs / "%x-%A_%a.err")]
    if kind == "downstream":
        indices = indices if indices is not None else list(range(40))
        cmd += ["--nodes=1", "--ntasks=1", "--job-name=mjde12-ft", "--partition=gl40s_short,gl40s_long", "--gres=gpu:1",
                "--cpus-per-task=2", "--mem=20G", "--time=04:00:00",
                "--array=" + ",".join(map(str, indices)) + "%4", "--export=ALL,MJ12_KIND=downstream"]
        if manifest.get("downstream_excluded_nodes"):
            cmd += ["--exclude=" + ",".join(manifest["downstream_excluded_nodes"])]
    else:
        index = indices[0]
        entry = manifest["pretrain_entries"][index]
        resources = (["--nodes=1-4", "--ntasks=4", "--gpus-per-task=a100:1",
                      "--cpus-per-task=4", "--mem-per-gpu=32G"]
                     if manifest.get("pretrain_launcher") == "slurm_flexible" else
                     ["--nodes=1", "--ntasks=1", "--gres=gpu:a100:4", "--cpus-per-task=16", "--mem=128G"])
        cmd += resources + ["--job-name=gr2-" + entry["arm"],
                "--partition=" + entry.get("gpu_partitions", "a100_short,a100_long"),
                "--time=" + entry.get("time_limit", "24:00:00"),
                f"--export=ALL,MJ12_KIND=pretrain,MJ12_INDEX={index}"]
        if manifest.get("pretrain_excluded_nodes"):
            cmd += ["--exclude=" + ",".join(manifest["pretrain_excluded_nodes"])]
    cmd.append(str(folder / "worker.sh"))
    job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    for i in indices:
        e = manifest["submitted_entries" if kind == "downstream" else "pretrain_entries"][i]
        e.setdefault("job_history", []).append(job + "_" + str(i) if kind == "downstream" else job)
        e["job"] = e["job_history"][-1]
    if kind == "downstream" and len(indices) == 40:
        manifest["downstream_submitted"] = True
    write_json(folder / "manifest.json", manifest)
    if kind == "pretrain" and entry.get("timeout_continuation"):
        attach_continuation(folder, manifest, entry)
    return job

def attach_continuation(folder, manifest, entry):
    """A CPU-only afterany callback avoids an hourly delay after a dev timeout."""
    if entry.get("continuation_for_job") == entry["job"]:
        return entry["continuation_job"]
    command = shlex.join([manifest["python"], str(folder / "controller/gr2_pretrain_campaign.py"),
                          "continue", "--folder", str(folder), "--job", entry["job"]])
    cmd = ["sbatch", "--parsable", "--account=system", "--job-name=gr2-resume-" + entry["arm"],
           "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
           "--mem=4G", "--time=00:15:00", "--dependency=afterany:" + entry["job"],
           "--output=" + str(folder / "logs/continuation-%j.out"),
           "--error=" + str(folder / "logs/continuation-%j.err"),
           "--wrap=export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec " + command]
    followup = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    entry.update(continuation_for_job=entry["job"], continuation_job=followup)
    entry.setdefault("continuation_history", []).append(followup)
    write_json(folder / "manifest.json", manifest)
    return followup

def timeout_resume_epoch(entry, config, saved):
    if saved["config"] != config or saved.get("extra", {}).get("partial_epoch_smoke") or saved["epoch"] < 1:
        raise ValueError("Unsafe timeout resume: " + entry["result_dir"])
    if entry["kind"] == "pretrain":
        if len(saved.get("rng_states", [])) != 4 or any(k not in saved for k in ("model", "optimizer", "scheduler")):
            raise ValueError("Pretrain resume requires model, optimizer, scheduler and all four RNG states")
        if saved["epoch"] <= entry.get("last_resume_epoch", 0):
            raise ValueError("No completed-epoch progress since previous resume: " + entry["result_dir"])
        if saved["epoch"] >= config["optimization"]["epochs"]:
            raise ValueError("Target epoch reached; verify the final checkpoint instead of resubmitting")
    return saved["epoch"]

def worker(folder, manifest):
    kind = os.environ["MJ12_KIND"]
    if kind != "pretrain":
        raise ValueError("Pretrain only")
    index = int(os.environ.get("MJ12_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    e = manifest["submitted_entries" if kind == "downstream" else "pretrain_entries"][index]
    cfg = yaml.safe_load(Path(e["config"]).read_text())
    if digest(e["config"]) != e["config_sha256"]:
        raise ValueError("Campaign config changed")
    os.chdir(e["source"])
    last = Path(e["result_dir"]) / "last.pth"
    flexible = kind == "pretrain" and manifest.get("pretrain_launcher") == "slurm_flexible"
    if kind == "downstream":
        if not manifest.get("checkpoint_sha256") or digest(cfg["model"]["checkpoint"]) != manifest["checkpoint_sha256"]:
            raise ValueError("Downstream requires the verified MJDE checkpoint")
        if manifest.get("downstream_runner") == "base":
            cmd = [manifest["python"], "finetune.py", "--config", e["config"]]
        else:
            cmd = [manifest["python"], "-m", "ablation.finetune", "--config", e["config"],
                   "--checkpoint", cfg["model"]["checkpoint"], "--output", e["result_dir"]]
    elif flexible:
        hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()
        os.environ.update(MASTER_ADDR=hosts[0], MASTER_PORT=str(15000 + int(os.environ["SLURM_JOB_ID"]) % 40000),
                          SLURM_EXPORT_ENV="ALL", NCCL_ASYNC_ERROR_HANDLING="1")
        cmd = ["srun", "--nodes=" + os.environ["SLURM_JOB_NUM_NODES"], "--ntasks=4",
               "--gpus-per-task=a100:1", "--gpu-bind=single:1", "--distribution=block",
               "--kill-on-bad-exit=1", "--export=ALL", manifest["python"], str(Path(__file__).resolve()),
               "rank-worker", "--folder", str(folder)]
    else:
        probe = "import torch; assert torch.cuda.device_count()==4; " + \
                "[(torch.ones(1,device='cuda:'+str(i)).sum().item(),torch.cuda.synchronize(i)) for i in range(4)]; print('CUDA probe: four GPUs ready',flush=True)"
        subprocess.run([manifest["python"], "-c", probe], check=True)
        # This allocation owns four GPUs; torchrun performs the NCCL setup.
        cmd = [manifest["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--module", "ablation.pretrain", "--config", e["config"], "--distributed", "--output", e["result_dir"]]
    if last.exists() and not flexible:
        cmd += ["--resume", str(last)]
    out = Path(e["result_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(e["config"], out / "campaign_config.yaml")
    os.execvp(cmd[0], cmd)

def rank_environment(env):
    """One visible CUDA device per Slurm task, including uneven node layouts."""
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ValueError("Expected exactly one bound GPU per Slurm task")
    rank, world = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"])
    if world != 4 or not 0 <= rank < world or not env.get("MASTER_ADDR") or not env.get("MASTER_PORT"):
        raise ValueError("Invalid four-rank rendezvous")
    return dict(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0")

def rank_worker(folder, manifest):
    os.environ.update(rank_environment(os.environ))
    e = manifest["pretrain_entries"][int(os.environ["MJ12_INDEX"])]
    if digest(e["config"]) != e["config_sha256"]:
        raise ValueError("Campaign config changed")
    os.chdir(e["source"])
    # NVML exposes the GPU allowed by this task's Slurm device cgroup. Use its
    # UUID so a task-relative ordinal cannot be confused with a physical index.
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    os.environ.update(bound_gpu_environment(uuids))
    health_path = Path(e["result_dir"]) / ("gpu-health-" + os.environ["SLURM_JOB_ID"]
                                           + "-rank" + os.environ["RANK"] + ".json")
    query = subprocess.run(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"], "-q"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    health = dict(job=os.environ["SLURM_JOB_ID"], rank=int(os.environ["RANK"]),
                  node=os.environ.get("SLURMD_NODENAME"), gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                  recovery_action=gpu_recovery_action(query.stdout), nvidia_smi=query.stdout,
                  query_exit_code=query.returncode, cuda_probe="starting")
    write_json(health_path, health)
    print(json.dumps(dict(rank=os.environ["RANK"], node=os.environ.get("SLURMD_NODENAME"),
                          allocated_gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                          gpu_recovery_action=health["recovery_action"], cuda_probe="starting")), flush=True)
    # Fail on an unusable allocated GPU before loading the dataset or joining NCCL.
    try:
        if health["recovery_action"].lower() not in ("none", "n/a", "unavailable"):
            raise RuntimeError("Allocated GPU requires recovery: " + health["recovery_action"])
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError("Slurm task must expose exactly one CUDA GPU")
        torch.cuda.set_device(0)
        torch.ones(1, device="cuda:0").sum().item()
        torch.cuda.synchronize(0)
    except Exception as exc:
        health.update(cuda_probe="failed", error=repr(exc))
        try:
            kernel = subprocess.run(["dmesg", "--ctime"], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=10)
            health["kernel_exit_code"] = kernel.returncode
            health["kernel_nvrm"] = [line for line in kernel.stdout.splitlines() if "NVRM:" in line]
            if kernel.returncode:
                health["kernel_error"] = kernel.stdout
        except Exception as diagnostic_error:
            health["kernel_error"] = repr(diagnostic_error)
        write_json(health_path, health)
        raise
    health["cuda_probe"] = "passed"
    write_json(health_path, health)
    print(json.dumps(dict(rank=os.environ["RANK"], world_size=4, local_rank=0,
                          node=os.environ.get("SLURMD_NODENAME"),
                          visible_gpu=os.environ["CUDA_VISIBLE_DEVICES"], cuda_probe="passed")), flush=True)
    cmd = [manifest["python"], "-m", "ablation.pretrain", "--config", e["config"],
           "--distributed", "--output", e["result_dir"]]
    last = Path(e["result_dir"]) / "last.pth"
    if last.exists():
        cmd += ["--resume", str(last)]
    os.execv(cmd[0], cmd)

def bound_gpu_environment(nvml_output):
    uuids = [line.strip() for line in nvml_output.splitlines() if line.strip()]
    if len(uuids) != 1 or not uuids[0].startswith("GPU-") or "," in uuids[0]:
        raise ValueError("Slurm task must expose one allocated GPU UUID; refusing ambiguous binding")
    return dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=uuids[0])

def gpu_recovery_action(nvidia_smi_output):
    """Read the driver's recovery requirement independently of CUDA ordinal binding."""
    for line in nvidia_smi_output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "GPU Recovery Action":
            return value.strip()
    return "unavailable"

def states(entries):
    jobs = sorted({e["job"].split("_")[0] for e in entries if e.get("job")})
    if not jobs:
        return {}
    text = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
                                    "--format=JobID,State,ExitCode"], text=True)
    return {line.split("|")[0]: line.split("|")[1].split()[0] for line in text.splitlines() if "|" in line}


def continue_after_job(folder, expected_job):
    """Allow accounting to settle after afterany without an hourly retry delay."""
    active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED", "UNKNOWN"}
    for _ in range(30):
        manifest = json.loads((folder / "manifest.json").read_text())
        entries = [e for e in manifest["pretrain_entries"] if e.get("job") == expected_job]
        if not entries:
            return {"superseded_job": expected_job}
        if states(entries).get(expected_job, "UNKNOWN") not in active:
            return monitor_step(folder)
        time.sleep(10)
    raise RuntimeError("Accounting has not settled for " + expected_job + "; hourly monitor will retry")


def monitor_step(folder):
    import fcntl
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((folder / "manifest.json").read_text())
        entry = manifest["pretrain_entries"][0]
        state = states([entry]).get(entry.get("job"), "UNKNOWN")
        output = Path(entry["result_dir"])
        config = yaml.safe_load(Path(entry["config"]).read_text())
        complete = False
        final = output / ("checkpoint-epoch-%04d.pth" % config["optimization"]["epochs"])
        terminal = {"COMPLETED", "TIMEOUT", "PREEMPTED"}
        if state in terminal and final.is_file():
            if not entry.get("verified_complete"):
                import torch
                sys.path.insert(0, entry["source"])
                from ablation.bootstrap import ensure_data_imports
                ensure_data_imports()
                from ablation.models import build_pretrain
                from ablation.pretrain_resume import validate_resume
                saved = torch.load(final, map_location="cpu", weights_only=False)
                validate_resume(saved, config, 4)
                assert saved["epoch"] == config["optimization"]["epochs"]
                assert saved["config"] == config
                model = build_pretrain(config, torch.device("cpu"))
                model.load_state_dict(saved["model"], strict=True)
                for key in ("torch", "cuda"):
                    assert len({s[key].numpy().tobytes() for s in saved["rng_states"]}) == 4, key
                entry.update(verified_complete=True, checkpoint_sha256=digest(final),
                             completed_steps=saved["extra"]["step"],
                             dataset_fingerprint=saved["extra"]["dataset_fingerprint"])
                del saved, model
            complete = True
        elif state in terminal:
            if entry["retries"] >= entry.get("max_timeout_resumes", config["optimization"]["epochs"]):
                raise ValueError("Continuation limit reached before the target epoch")
            import torch
            saved = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
            epoch = timeout_resume_epoch(entry, config, saved)
            del saved
            entry.update(retries=entry["retries"] + 1, last_resume_epoch=epoch)
            submit(folder, manifest, "pretrain", [0])
            state = "RESUBMITTED"
        progress = None
        metrics = output / "metrics.jsonl"
        if metrics.exists():
            with metrics.open("rb") as stream:
                stream.seek(max(0, metrics.stat().st_size - 100000))
                for line in reversed(stream.read().decode(errors="replace").splitlines()):
                    try:
                        row = json.loads(line)
                        progress = {k: row[k] for k in ("epoch", "step", "loss", "lr_used") if k in row}
                        break
                    except ValueError:
                        continue
        job = entry.get("job")
        health = [json.loads(p.read_text()) for p in output.glob("gpu-health-%s-rank*.json" % job)]
        rng = [json.loads(p.read_text()) for p in output.glob("rng-start-%s-rank*.json" % job)]
        probes = len(health) == 4 and all(h["cuda_probe"] == "passed" for h in health)
        rng_distinct = len(rng) == 4 and len({r.get("cuda_sha256") for r in rng}) == 4
        active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESUBMITTED", "UNKNOWN"}
        payload = dict(checked_utc=datetime.now(timezone.utc).isoformat(), job=job, state=state,
                       complete=complete, progress=progress, four_cuda_probes_passed=probes,
                       four_rng_states_distinct=rng_distinct,
                       training_started=probes and rng_distinct and bool(progress)
                           and progress.get("step", 0) > max(r.get("start_step", 0) for r in rng),
                       attention=not complete and state not in active)
        write_json(folder / "manifest.json", manifest)
        write_json(folder / "status.json", payload)
        results = Path(manifest["results_dir"])
        results.mkdir(parents=True, exist_ok=True)
        write_json(results / "status.json", payload)
        shutil.copy2(entry["config"], results / "config.yaml")
        if complete:
            report = dict(alias=manifest["alias"], checkpoint=str(final),
                          checkpoint_sha256=entry["checkpoint_sha256"], epochs=config["optimization"]["epochs"],
                          optimizer_steps=entry["completed_steps"], dataset_fingerprint=entry["dataset_fingerprint"],
                          config=config, jobs=entry["job_history"], validation="strict model load; 4 independent rank RNG states",
                          downstream_evaluated=False)
            write_json(results / "pretrain_result.json", report)
            (results / "PRETRAIN.md").write_text("# Full MJDE — GR2 geometry pretrain\n\n"
                + "Completed 40 epochs. Checkpoint: `" + str(final) + "`.\n\n"
                + "35–75 degree geodesic radius, 2–10 second tubelets, 50% unique targets; rank RNG seed + rank.\n\n"
                + "This is a training completion report. Downstream performance has not been evaluated.\n", encoding="utf-8")
        error = folder / "controller_error.json"
        if error.exists():
            error.unlink()
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "worker", "rank-worker", "step", "monitor", "continue"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    folder = args.folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if args.action == "submit":
        if manifest["pretrain_entries"][0].get("job"):
            raise ValueError("Already submitted; use the controller for timeout recovery")
        print(submit(folder, manifest, "pretrain", [0]))
    elif args.action == "worker":
        worker(folder, manifest)
    elif args.action == "rank-worker":
        rank_worker(folder, manifest)
    elif args.action == "continue":
        if not args.job:
            parser.error("continue requires --job")
        print(json.dumps(continue_after_job(folder, args.job)), flush=True)
    else:
        while True:
            try:
                status = monitor_step(folder)
                print(json.dumps(status), flush=True)
                if status["complete"] or args.action == "step":
                    return
            except Exception as exc:
                write_json(folder / "controller_error.json", dict(error=repr(exc), checked_utc=datetime.now(timezone.utc).isoformat()))
                if args.action == "step":
                    raise
            time.sleep(manifest.get("monitor_interval_seconds", 3600))


if __name__ == "__main__":
    main()
