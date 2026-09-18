"""Bind a Slurm downstream task to its allocated GPU before importing training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def allocated_gpu(text, minimum_free_mib):
    rows = [line.strip().split(",") for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("Expected exactly one GPU exposed by the Slurm task cgroup")
    uuid, total, free = [value.strip() for value in rows[0]]
    if not uuid.startswith("GPU-"):
        raise ValueError("Missing allocated GPU UUID")
    result = dict(gpu_uuid=uuid, total_mib=int(total), free_mib=int(free))
    if result["free_mib"] < minimum_free_mib:
        raise RuntimeError("Allocated GPU has insufficient free memory: " + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.folder / "manifest.json").read_text())
    entry = manifest["submitted_entries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = Path(entry["result_dir"])
    output.mkdir(parents=True, exist_ok=True)
    health_path = output / ("gpu-health-" + os.environ["SLURM_JOB_ID"] + ".json")
    health = dict(job=os.environ["SLURM_JOB_ID"], node=os.environ.get("SLURMD_NODENAME"),
                  original_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  checked_utc=datetime.now(timezone.utc).isoformat(), cuda_probe="starting")
    try:
        query = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], text=True, timeout=20)
        health["nvml_query"] = query
        health.update(allocated_gpu(query, manifest.get("downstream_min_free_mib", 12288)))
        os.environ.update(CUDA_VISIBLE_DEVICES=health["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one CUDA device after UUID binding")
        torch.cuda.set_device(0)
        torch.ones(1, device="cuda:0").sum().item()
        torch.cuda.synchronize(0)
        health.update(cuda_probe="passed", cuda_free_bytes=torch.cuda.mem_get_info(0)[0])
    except Exception as exc:
        health.update(cuda_probe="failed", error=repr(exc))
        raise
    finally:
        health_path.write_text(json.dumps(health, indent=2) + "\n")
        print(json.dumps(health), flush=True)
    command = [manifest["python"], str(args.folder / "controller/mjde12_campaign.py"),
               "worker", "--folder", str(args.folder)]
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
