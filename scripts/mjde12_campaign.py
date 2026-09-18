"""Frozen MJDE downstream / twelve-block pretrain campaign and CPU controller."""
from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 696, 1001, 1234, 3407]
RECIPES = {
    "chb": ("CHB-MIT", "chb", 1e-4, .01, .1, 0., 20),
    "siena": ("SIENA", "siena", 1e-4, .05, .1, 0., 50),
    "physionet_mi": ("PHYSIONET-MI", "physio", 5e-5, .01, .3, .1, 50),
    "faced": ("FACED", "faced", 1e-4, .01, .1, .1, 50),
    "seedv": ("SEED-V", "seed-v", 1e-4, .01, .1, .1, 50),
    "bciciv2a": ("BCIC-IV-2a", "bciciv2a", 1e-4, .01, .2, .1, 50),
    "tusz": ("TUSZ", "tusz", 1e-4, .01, .3, 0., 50),
    "tusl": ("TUSL", "tusl", 1e-4, .01, .3, .1, 50),
}


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


def snapshot(destination, ablation=False):
    destination.mkdir(parents=True)
    for name in ("src", "configs") + (("ablation",) if ablation else ()):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for name in ("pretrain.py", "finetune.py"):
        shutil.copy2(ROOT / name, destination / name)
    return {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}


def prepare(folder):
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    source = folder / "source"
    hashes = snapshot(source, ablation=True)
    reference = ROOT / "outputs/geometry50_nearest3_7_t2_15_20260913/training/checkpoint-epoch-0040.pth"
    if digest(reference) != "689f67c47ee8c4b9d3e1d9aaf52d6d3c36ab930784011b100668ae4040c3304b":
        raise ValueError("Unexpected KNN37 checkpoint")
    checkpoint = reference
    entries = []
    for slug, (name, dataset, lr, wd, dropout, smoothing, epochs) in RECIPES.items():
        for seed in SEEDS:
            stem = ("nearest3_7_" if slug in ("tusz", "tusl") else "gr9-1_") + slug
            base = ROOT / "configs/downstream" / f"{stem}_seed{seed}.yaml"
            cfg = yaml.safe_load(base.read_text())
            assert cfg["data"]["dataset"] == dataset and cfg["seed"] == seed
            cfg["model"].update(checkpoint=str(checkpoint), head_dropout=dropout,
                                head_hidden_tokens=4 if slug == "seedv" else None)
            opt = cfg["optimization"]
            for key in list(opt):
                if "warmup" in key:
                    del opt[key]
            opt.update(epochs=epochs, batch_size_per_gpu=64, gradient_accumulation_steps=1,
                       tokenizer_learning_rate=lr, encoder_learning_rate=lr, head_learning_rate=lr,
                       min_learning_rate=1e-6, weight_decay=wd, label_smoothing=smoothing)
            if slug == "tusz":
                opt["class_counts"] = [28670, 12842]
            out = folder / "downstream" / f"{slug}_seed{seed}"
            cfg["runtime"]["output"] = str(out)
            cp = folder / "downstream_configs" / f"{slug}_seed{seed}.yaml"
            cp.parent.mkdir(exist_ok=True)
            cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
            entries.append(dict(index=len(entries), slug=slug, display_name=name, dataset=dataset,
                                seed=seed, config=str(cp), config_sha256=digest(cp), result_dir=str(out),
                                source=str(source), kind="downstream", retries=0))
    from ablation.config import load_config, resolve_ablation
    pretrains = []
    for arm in ("mjde_lite", "labram", "cbramod", "csbrain"):
        cfg = resolve_ablation(load_config("ablation/configs/encoder_" + arm + ".yaml"))
        if not arm.startswith("mjde"):
            assert cfg["ablation"]["depth"] == 12
        out = folder / "pretrain" / arm
        cfg["runtime"]["output"] = str(out)
        cp = folder / "pretrain_configs" / (arm + ".yaml")
        cp.parent.mkdir(exist_ok=True)
        cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        pretrains.append(dict(index=len(pretrains), kind="pretrain", arm=arm, seed=42,
                             config=str(cp), config_sha256=digest(cp), result_dir=str(out),
                             source=str(source), retries=0))
    reference_config = folder / "reference_full_config.yaml"
    expected = resolve_ablation(load_config("ablation/configs/encoder_mjde.yaml"))
    reference_config.write_text(yaml.safe_dump(expected, sort_keys=False))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                    python=str(ROOT / ".venv/bin/python"), checkpoint=str(checkpoint),
                    reference_checkpoint=str(reference), reference_sha256=digest(reference),
                    checkpoint_sha256=None, submitted_entries=entries, reused_entries=[],
                    reuse_mjde_reference=True, reference_full_config=str(reference_config),
                    downstream_runner="base", monitor_interval_seconds=3600,
                    pretrain_launcher="slurm_flexible",
                    source_hashes=hashes, pretrain_entries=pretrains, pretrain_prepared=True,
                    downstream_submitted=False, downstream_gate="verified full KNN37 epoch 40 reference; lite is an independent pretrain",
                    stamp=datetime.now().strftime("%m%d%H%M"), alias="knn37-full",
                    selector="validation balanced_accuracy", excluded_optuna=["tuev", "tuab", "mentalarithmetic", "isruc", "hmc"])
    write_json(folder / "manifest.json", manifest)
    # Freeze the controller and publisher separately from active Optuna code.
    ctl = folder / "controller"
    ctl.mkdir()
    for name in ("mjde12_campaign.py", "monitor_experiment_results.py"):
        shutil.copy2(ROOT / "scripts" / name, ctl / name)
    shutil.copy2(ROOT / "configs/results/literature_baselines.json", ctl / "literature.json")
    launch = folder / "worker.sh"
    launch.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                      "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1\n"
                      'export TMPDIR="${SLURM_TMPDIR:-/tmp}/mj12-${SLURM_JOB_ID}"\nmkdir -p "$TMPDIR"\n'
                      + shlex.join([manifest["python"], str(ctl / "mjde12_campaign.py"), "worker", "--folder", str(folder)]) + "\n")
    return manifest


def submit(folder, manifest, kind, indices=None):
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
        cmd += resources + ["--job-name=mj12-" + entry["arm"],
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
    command = shlex.join([manifest["python"], str(folder / "controller/mjde12_campaign.py"),
                          "continue", "--folder", str(folder), "--job", entry["job"]])
    cmd = ["sbatch", "--parsable", "--account=system", "--job-name=mj12-resume-" + entry["arm"],
           "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
           "--mem=8G", "--time=00:15:00", "--dependency=afterany:" + entry["job"],
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


def verify_reference(manifest):
    """Reuse only the known complete full model with unchanged training settings."""
    import torch
    path = Path(manifest["checkpoint"])
    if str(path) != manifest["reference_checkpoint"] or digest(path) != manifest["reference_sha256"]:
        raise ValueError("Reference checkpoint hash/path differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = yaml.safe_load(Path(manifest["reference_full_config"]).read_text())

    def training_config(config):
        config = copy.deepcopy(config)
        config.pop("ablation", None)
        config.get("runtime", {}).pop("output", None)
        return config

    if (expected.get("ablation", {}).get("encoder") != "mjde" or
            saved["epoch"] != 40 or saved.get("extra", {}).get("partial_epoch_smoke") or
            training_config(saved["config"]) != training_config(expected)):
        raise ValueError("Reference full MJDE configuration/epoch differs")
    if saved["config"].get("ablation", {}).get("encoder", "mjde") != "mjde":
        raise ValueError("Reference is not full MJDE")
    manifest["checkpoint_sha256"] = manifest["reference_sha256"]


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
    # Controller is the only writer after pretrain registration; use advisory lock
    # so registration and timeout recovery cannot overwrite each other's manifest.
    import fcntl
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((folder / "manifest.json").read_text())
        root = Path(manifest["root"])
        entries = manifest["submitted_entries"] + manifest["pretrain_entries"]
        status = states(entries)
        mjde = next((e for e in manifest["pretrain_entries"] if e["arm"] == "mjde"), None)
        if not manifest["downstream_submitted"] and manifest.get("reuse_mjde_reference"):
            verify_reference(manifest)
            submit(folder, manifest, "downstream")
        elif not manifest["downstream_submitted"] and mjde and status.get(mjde.get("job")) == "COMPLETED":
            checkpoint = Path(manifest["checkpoint"])
            if checkpoint.exists():
                import torch
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                expected = yaml.safe_load(Path(mjde["config"]).read_text())
                if saved["epoch"] != 40 or saved["config"] != expected or saved.get("extra", {}).get("partial_epoch_smoke"):
                    raise ValueError("New MJDE pretrain did not pass the downstream gate")
                del saved
                manifest["checkpoint_sha256"] = digest(checkpoint)
                submit(folder, manifest, "downstream")
                manifest["downstream_submitted"] = True
                write_json(folder / "manifest.json", manifest)
        rows, results = [], {}
        active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}
        for e in entries:
            out = Path(e["result_dir"])
            state = status.get(e.get("job"), "WAITING_PRETRAIN" if not e.get("job") else "UNKNOWN")
            complete = False
            if e["kind"] == "downstream":
                try:
                    result = json.loads((out / "result.json").read_text())
                    if not (out / "best-balanced_accuracy.pth").is_file():
                        raise ValueError("Missing selected weight")
                    import monitor_experiment_results as publisher
                    publisher.selector_test(result)
                    results[str(out / "result.json")] = result
                    complete = True
                except (OSError, ValueError, KeyError):
                    pass
            else:
                final = out / "checkpoint-epoch-0040.pth"
                if state in {"COMPLETED", "TIMEOUT", "PREEMPTED"} and final.is_file():
                    if not e.get("verified_complete"):
                        import torch
                        saved = torch.load(final, map_location="cpu", weights_only=False)
                        expected = yaml.safe_load(Path(e["config"]).read_text())
                        if saved["epoch"] != 40 or saved["config"] != expected or saved.get("extra", {}).get("partial_epoch_smoke"):
                            raise ValueError("Invalid completed pretrain: " + e["arm"])
                        e["checkpoint_sha256"] = digest(final)
                        e["verified_complete"] = True
                        del saved
                    complete = True
            # Resume interrupted pretraining, including a clean early exit, only
            # from a committed checkpoint that made progress. Never retry CUDA
            # failures, OOM, or a user cancellation as if they were time limits.
            resumable_states = {"TIMEOUT", "PREEMPTED", "COMPLETED"} if e["kind"] == "pretrain" else {"TIMEOUT"}
            retry_limit = e.get("max_timeout_resumes", 40 if e["kind"] == "pretrain" else 2)
            if not complete and state in resumable_states and e["retries"] < retry_limit and (out / "last.pth").exists():
                if e["kind"] == "pretrain" or not any(status.get(d.get("job")) in active for d in manifest["submitted_entries"]):
                    import torch
                    saved = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
                    cfg = yaml.safe_load(Path(e["config"]).read_text())
                    resume_epoch = timeout_resume_epoch(e, cfg, saved)
                    del saved
                    e["retries"] += 1
                    e["last_resume_epoch"] = resume_epoch
                    submit(folder, manifest, e["kind"], [e["index"]])
                    state = "RESUBMITTED"
            progress = None
            for filename in ("validation.jsonl", "metrics.jsonl"):
                p = out / filename
                if p.exists():
                    with p.open("rb") as stream:
                        stream.seek(max(0, p.stat().st_size - 100000))
                        tail = stream.read().decode("utf-8", errors="replace").splitlines()
                    for line in reversed(tail):
                        try:
                            record = json.loads(line)
                            progress = {k: record[k] for k in ("epoch", "step", "loss", "balanced_accuracy") if k in record}
                            break
                        except ValueError:
                            continue
                    if progress:
                        break
            rows.append(dict(kind=e["kind"], name=e.get("slug", e.get("arm")), seed=e["seed"],
                             job=e.get("job"), state=state, complete=complete, progress=progress))
        if len(results) == 40 and not manifest.get("published"):
            import monitor_experiment_results as publisher
            publisher.RESULTS_ROOT = root / "outputs/results"
            literature = json.loads((folder / "controller/literature.json").read_text())
            publisher.publish(manifest, results, manifest["stamp"], manifest["alias"], literature)
            manifest["published"] = True
        write_json(folder / "manifest.json", manifest)
        payload = dict(checked_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                       downstream_complete=len(results), downstream_expected=40,
                       published=manifest.get("published", False),
                       complete=manifest["pretrain_prepared"] and all(r["complete"] for r in rows),
                       attention=[r for r in rows if not r["complete"] and r["state"] not in active | {"UNKNOWN", "RESUBMITTED", "WAITING_PRETRAIN"}])
        write_json(folder / "status.json", payload)
        error_file = folder / "controller_error.json"
        if error_file.exists():
            error_file.unlink()
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "worker", "rank-worker", "step", "monitor", "continue"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(folder)))
    elif args.action == "submit":
        manifest = json.loads((folder / "manifest.json").read_text())
        jobs = []
        for entry in manifest["pretrain_entries"]:
            if not entry.get("job"):
                jobs.append(submit(folder, manifest, "pretrain", [entry["index"]]))
        print(json.dumps(jobs))
    elif args.action == "worker":
        worker(folder, json.loads((folder / "manifest.json").read_text()))
    elif args.action == "rank-worker":
        rank_worker(folder, json.loads((folder / "manifest.json").read_text()))
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
            manifest = json.loads((folder / "manifest.json").read_text())
            time.sleep(manifest.get("monitor_interval_seconds", 3600))


if __name__ == "__main__":
    main()
