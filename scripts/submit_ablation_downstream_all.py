"""Prepare and submit the three completed ablation checkpoints to downstream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "outputs/ablation/encoder_cbramod/downstream_no_seedvig_20260915"
ARMS = {
    "encoder_mjde": ("", "mjde-dense-zero-reference"),
    "encoder_labram": ("27463233", "labram-encoder-ablation"),
    "encoder_cbramod": ("27461315", "cbramod-encoder-ablation"),
    "pe_reve4d": ("27461313", "reve4d-positional-encoding-ablation"),
}
CAMPAIGN_NAME = "downstream_all13_20260916"
SEEDS = {42, 696, 1001, 1234, 3407}
CONFIG_SLUGS = {"seedv": "seed-v", "mentalarithmetic": "stress", "physionet_mi": "physio"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_no_warmup(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "warmup" in str(key).lower():
                raise ValueError("Downstream warmup key is forbidden: " + str(key))
            assert_no_warmup(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_warmup(child)


def prepare(arm: str, pretrain_job: str, campaign_name: str = CAMPAIGN_NAME) -> tuple[Path, list[dict]]:
    campaign = ROOT / "outputs/ablation" / arm / campaign_name
    if campaign.exists():
        raise FileExistsError(f"Refusing to overwrite campaign: {campaign}")
    checkpoint = ROOT / "outputs/ablation" / arm / "seed42/checkpoint-epoch-0040.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    template_manifest = json.loads((TEMPLATE / "manifest.json").read_text(encoding="utf-8"))
    old_entries = template_manifest["submitted_entries"] + template_manifest["reused_entries"]
    by_dataset: dict[str, set[int]] = {}
    for entry in old_entries:
        by_dataset.setdefault(entry["slug"], set()).add(int(entry["seed"]))
    if len(by_dataset) != 13 or any(seeds != SEEDS for seeds in by_dataset.values()):
        raise ValueError("Template must contain exactly 13 datasets x five seeds")

    shutil.copytree(TEMPLATE / "source", campaign / "source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entries = []
    for index, old in enumerate(old_entries):
        slug, seed = old["slug"], int(old["seed"])
        config = campaign / "source/configs_downstream" / f"{slug}_seed{seed}.yaml"
        content = Path(old["config"]).read_text(encoding="utf-8")
        content = content.replace(str(TEMPLATE), str(campaign))
        if str(TEMPLATE) in content or str(campaign) not in content:
            raise ValueError(f"Config output replacement failed: {old['config']}")
        parsed = yaml.safe_load(content)
        assert_no_warmup(parsed)
        if parsed["data"]["dataset"] != CONFIG_SLUGS.get(slug, slug) or int(parsed["seed"]) != seed:
            raise ValueError(f"Config identity mismatch: {config}")
        config.write_text(content, encoding="utf-8")
        output = campaign / "downstream" / f"{slug}_seed{seed}"
        entry = dict(old, index=index, config=str(config), config_sha256=digest(config),
                     checkpoint=str(checkpoint), output=str(output), result_dir=str(output))
        entries.append(entry)
    write_json(campaign / "array.json", entries)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), arm=arm,
                    pretrain_job_id=pretrain_job, checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint),
                    policy="Base learning rate from first downstream update; no warmup; validation checkpoint selector",
                    scope="13 datasets, five seeds each", submitted_entries=entries, reused_entries=[])
    write_json(campaign / "manifest.json", manifest)

    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    source = campaign / "source"
    subprocess.run([sys.executable, str(source / "verify_checkpoint.py"), str(campaign)],
                   cwd=source, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "ablation.finetune", "--config", entries[0]["config"],
                    "--checkpoint", str(checkpoint), "--device", "cpu", "--dry-run"],
                   cwd=source, env=env, check=True, stdout=subprocess.DEVNULL)
    print(f"prepared {arm}: {len(entries)} entries, checkpoint and dry run verified", flush=True)
    return campaign, entries


def submit(arm: str, campaign: Path, entries: list[dict]) -> str:
    source = campaign / "source"
    python = ROOT / ".venv/bin/python"
    wrapper = (
        "unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; "
        'export TMPDIR="/tmp/abl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; '
        'mkdir -p "$TMPDIR"; exec srun --ntasks=1 --gpus-per-task=l40s:1 '
        "--cpu-bind=none --kill-on-bad-exit=1 "
        + shlex.join([str(python), str(source / "run_downstream_array.py")])
    )
    command = ["sbatch", "--parsable", "--account=system", "--job-name=abl-" + arm + "-ds13",
               "--partition=gl40s_dev,gl40s_short,gl40s_long", f"--array=0-{len(entries)-1}%4",
               "--nodes=1", "--ntasks=1", "--gpus-per-task=l40s:1", "--cpus-per-task=2",
               "--mem=32G", "--time=04:00:00", "--chdir=" + str(source),
               "--output=" + str(campaign / "slurm-%A_%a.log"), "--wrap", wrapper]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError(f"Unexpected sbatch response: {result.stdout!r}")
    write_json(campaign / "submission.json", dict(job_id=job_id, command=command,
               submitted_utc=datetime.now(timezone.utc).isoformat()))
    print(f"submitted {arm}: {job_id}", flush=True)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", choices=tuple(ARMS),
                        help="Submit only the named arm; may be repeated")
    parser.add_argument("--pretrain-job", default="",
                        help="Pretrain job ID recorded for a single selected arm")
    parser.add_argument("--campaign-name", default=CAMPAIGN_NAME)
    args = parser.parse_args()
    selected = args.arm or [arm for arm in ARMS if arm != "encoder_mjde"]
    if args.pretrain_job and len(selected) != 1:
        parser.error("--pretrain-job requires exactly one --arm")
    prepared = [(arm, *prepare(arm, args.pretrain_job or ARMS[arm][0], args.campaign_name))
                for arm in selected]
    for arm, campaign, entries in prepared:
        submit(arm, campaign, entries)


if __name__ == "__main__":
    main()
