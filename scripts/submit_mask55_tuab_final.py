"""Run the five missing mask-55 TUAB seeds in a fresh writable A100 campaign."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import digest, write_json

BASE = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55")
SEEDS = (42, 696, 1001, 1234, 3407)


def main():
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%y%m%d-%H%M")
    campaign = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results") / (stamp + "-gr2-d2-mask55-tuab-a100")
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        ".git", ".venv*", "outputs", "results", "__pycache__", "*.pyc", "*.pth"))
    verified = json.loads((BASE / "pretrain/verified.json").read_text())
    checkpoint = Path(verified["checkpoint"])
    if digest(checkpoint) != verified["sha256"]:
        raise ValueError("Mask-55 checkpoint changed")
    (campaign / "pretrain").mkdir()
    write_json(campaign / "pretrain/verified.json", verified)
    entries = []
    for seed in SEEDS:
        config = yaml.safe_load((BASE / "configs/downstream" / f"tuab_seed{seed}.yaml").read_text())
        output = campaign / "downstream/tuab" / f"seed-{seed}"
        config["model"]["checkpoint"] = str(checkpoint)
        config["runtime"]["output"] = str(output)
        config["data"]["num_workers"] = 2
        path = campaign / "configs/downstream" / f"tuab_seed{seed}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entries.append(dict(dataset="tuab", seed=seed, config=str(path), output=str(output)))
    write_json(campaign / "downstream_entries.json", entries)
    cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
    exclude = sorted(set(cluster["pretrain"].get("excluded_nodes", [])) | {"a100-4035", "a100-4046", "a100-4047"})
    logs = campaign / "downstream/logs"
    logs.mkdir(parents=True)
    command = ["sbatch", "--parsable", "--account=" + cluster["account"],
               "--job-name=mask55-tuab-final", "--partition=a100_short,a100_long",
               "--nodes=1", "--ntasks=1", "--gpus-per-task=a100:1", "--cpus-per-task=2",
               "--mem=32G", "--time=12:00:00", "--array=0-4%5", "--exclude=" + ",".join(exclude),
               "--chdir=" + str(source), "--output=" + str(logs / "%A_%a.out"),
               "--error=" + str(logs / "%A_%a.err"),
               "--wrap=exec " + shlex.join(["srun", "--ntasks=1", "--gpus-per-task=a100:1",
                   "--gpu-bind=single:1", "--kill-on-bad-exit=1", sys.executable,
                   str(source / "scripts/downstream_experiment_worker.py"), "--experiment", str(campaign),
                   "--source", str(source)])]
    job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    manifest = dict(experiment=campaign.name, alias="gr2-d2-patch-dimension-mask55",
                    preset="gr2-d2-patch-dimension-mask55", created_at_new_york=stamp,
                    publication_root=str(ROOT / "outputs/results"), checkpoint=str(checkpoint),
                    downstream_jobs=[dict(job=job, dataset="tuab", indices=list(range(5)),
                                          resources=dict(gpu="a100", partitions="a100_short,a100_long",
                                                         time="12:00:00"))], status="submitted")
    write_json(campaign / "manifest.json", manifest)
    finalizer = subprocess.check_output([
        "sbatch", "--parsable", "--account=" + cluster["account"], "--job-name=mask55-tuab-results",
        "--partition=cpu_short,cpu_long", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        "--mem=4G", "--time=00:30:00", "--dependency=afterany:" + job,
        "--output=" + str(logs / "results-%j.out"), "--error=" + str(logs / "results-%j.err"),
        "--wrap=exec " + shlex.join([sys.executable, str(source / "scripts/finalize_experiment.py"),
                                      "--experiment", str(campaign)])
    ], text=True).strip().split(";", 1)[0]
    manifest["finalizer_job"] = finalizer
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(dict(campaign=str(campaign), tuab_job=job, finalizer_job=finalizer), indent=2))


if __name__ == "__main__":
    main()
