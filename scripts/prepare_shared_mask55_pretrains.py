"""Prepare owner-hosted mask55 pretrains and let assigned accounts only submit them."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results")
OWNER = "ml10266"
ASSIGNMENTS = {
    "hk4935": (
        ("pe-ch_order", "mjde", "channel_id"),
        ("pe-acpe", "mjde", "acpe"),
        ("pe-4dREVE", "mjde", "reve4d"),
    ),
    "yc8820": (
        ("cbramod", "cbramod", "shpe"),
        ("csbrain", "csbrain", "shpe"),
        ("labram", "labram", "shpe"),
    ),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def new_york_stamp():
    previous = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/New_York"
        if hasattr(time, "tzset"):
            time.tzset()
        return datetime.now().strftime("%y%m%d-%H%M")
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        if hasattr(time, "tzset"):
            time.tzset()


def resolved_configs():
    from ablation.config import resolve_ablation

    base = yaml.safe_load((ROOT / "configs/pretrain_gr2_patch_feature.yaml").read_text())
    configs = {}
    for account, arms in ASSIGNMENTS.items():
        for slug, encoder, position in arms:
            config = deepcopy(base)
            config["masking"]["mask_ratio"] = .55
            config["mae"]["decoder_depth"] = 2
            applicability = "active"
            if encoder != "mjde":
                # The paper encoder replaces MJDE wholesale, so there is no
                # dual-path fusion gate to execute. Retain the patch-dimension
                # family as explicit reference metadata without mislabelling
                # an inactive module as trainable.
                config["encoder"]["fusion_gate"] = "static_feature"
                applicability = "not_applicable_encoder_replaced"
            config["ablation"] = dict(
                name="mask55_d2_patchdim_" + slug.replace("-", "_"),
                encoder=encoder, position=position, pe_scope="both",
                reference_fusion_gate="patch_feature",
                fusion_gate_applicability=applicability,
            )
            if encoder != "mjde":
                config["ablation"]["depth"] = 12
            if position == "reve4d":
                config["ablation"].update(reve_freqs=4, reve_noise_ratio=0.0)
            config = resolve_ablation(config)
            assert config["masking"]["policy"] == "geometry_tubelet"
            assert config["masking"]["mask_ratio"] == .55
            assert config["mae"]["decoder_depth"] == 2
            assert config["ablation"]["reference_fusion_gate"] == "patch_feature"
            configs[slug] = dict(account=account, config=config)
    return configs


def source_commit():
    return subprocess.check_output([
        "git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "rev-parse", "HEAD"
    ], text=True).strip()


def snapshot(destination):
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(
        ".git", ".venv*", "outputs", "results", "__pycache__", "*.pyc", "*.pth", "Data"))
    return digest(destination / "scripts/prepare_shared_mask55_pretrains.py")


def grant_account(folder, account):
    subprocess.run(["setfacl", "-R", "-m", f"u:{account}:rwx,u:{OWNER}:rwx,m:rwx", str(folder)], check=True)
    for directory in [folder, *[path for path in folder.rglob("*") if path.is_dir()]]:
        subprocess.run(["setfacl", "-m", f"d:u:{account}:rwx,d:u:{OWNER}:rwx,d:m:rwx", str(directory)], check=True)


def prepare(results_root):
    if getpass.getuser() != OWNER:
        raise PermissionError("Only ml10266 prepares the shared campaign")
    matches = sorted(results_root.glob("*-mask55-d2-patchdim-shared-pretrain"))
    if matches:
        raise FileExistsError("Matching campaign already exists: " + ", ".join(map(str, matches)))
    # Resolve every dependency before creating the campaign so an environment
    # error cannot leave a directory that looks like a prepared experiment.
    configs = resolved_configs()
    policy = yaml.safe_load((ROOT / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
    commit = source_commit()
    campaign = results_root / (new_york_stamp() + "-mask55-d2-patchdim-shared-pretrain")
    campaign.mkdir(parents=True)
    source = campaign / "source"
    launcher_sha = snapshot(source)
    entries = []
    for slug, spec in configs.items():
        account = spec["account"]
        arm = campaign / "accounts" / account / slug
        for name in ("configs", "pretrain", "logs"):
            (arm / name).mkdir(parents=True, exist_ok=True)
        config = spec["config"]
        config["runtime"]["output"] = str(arm / "pretrain")
        config_path = arm / "configs/pretrain.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entry = dict(kind="pretrain", arm=slug, config=str(config_path),
                     config_sha256=digest(config_path), source=str(source),
                     result_dir=str(arm / "pretrain"), job=None, job_history=[], retries=0,
                     last_resume_epoch=0, max_timeout_resumes=40,
                     time_limit=policy["pretrain"]["time"],
                     gpu_partitions=policy["pretrain"]["partitions"], timeout_continuation=False)
        manifest = dict(experiment="mask55-d2-patchdim-" + slug, owner=OWNER,
                        assigned_account=account, source_commit=commit, python=None,
                        preset="mask55-d2-patch-dimension", pretrain_launcher="slurm_flexible",
                        pretrain_entries=[entry], pretrain_excluded_nodes=policy["pretrain"]["excluded_nodes"],
                        resource_policy=policy["pretrain"], auto_resume=False, verify_before_success=True,
                        results_dir=str(arm / "pretrain/report"), status="prepared",
                        downstream_submitted=False,
                        comparison_note=("patch_feature gate active" if config["ablation"]["fusion_gate_applicability"] == "active"
                                         else "patch_feature reference family; MJDE gate not applicable after encoder replacement"))
        write_json(arm / "manifest.json", manifest)
        worker = arm / "worker.sh"
        worker.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          'exec "${EEGFM_PYTHON:?activate the training environment before sbatch}" '
                          + str(source / "scripts/gr2_pretrain_campaign.py")
                          + " worker --folder " + str(arm) + "\n", encoding="utf-8")
        worker.chmod(0o755)
        grant_account(arm, account)
        entries.append(dict(slug=slug, assigned_account=account, arm_folder=str(arm),
                            config=str(config_path), config_sha256=entry["config_sha256"],
                            fusion_gate_applicability=config["ablation"]["fusion_gate_applicability"]))
    write_json(campaign / "manifest.json", dict(
        experiment=campaign.name, owner=OWNER, source_commit=commit, source=str(source),
        launcher_sha256=launcher_sha, assignments={key: [row[0] for row in value] for key, value in ASSIGNMENTS.items()},
        settings=dict(mask_ratio=.55, decoder_depth=2, reference_fusion_gate="patch_feature",
                      epochs=40, seed=42, gpu="a100", tasks=4, downstream=False),
        resources=policy["pretrain"], entries=entries, status="prepared"))
    return campaign


def require_environment():
    import torch
    if torch.__version__.split("+", 1)[0] != "2.0.1" or not torch.version.cuda:
        raise RuntimeError("Activate the CUDA training environment with torch 2.0.1 before submission")


def submit(campaign):
    account = getpass.getuser()
    if account not in ASSIGNMENTS:
        raise PermissionError("This launcher is assigned only to hk4935 or yc8820")
    require_environment()
    top = json.loads((campaign / "manifest.json").read_text())
    if top["source_commit"] != source_commit():
        raise ValueError("Shared checkout HEAD differs from the frozen campaign; pull the owner's repository first")
    jobs = {}
    controller = campaign / "source/scripts/gr2_pretrain_campaign.py"
    for slug, _, _ in ASSIGNMENTS[account]:
        arm = campaign / "accounts" / account / slug
        manifest_path = arm / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["pretrain_entries"][0]
        if entry.get("job"):
            jobs[slug] = dict(job=entry["job"], status="already_submitted")
            continue
        if digest(entry["config"]) != entry["config_sha256"]:
            raise ValueError("Frozen config changed: " + slug)
        manifest["python"] = sys.executable
        write_json(manifest_path, manifest)
        environment = dict(os.environ, EEGFM_PYTHON=sys.executable)
        job = subprocess.check_output([sys.executable, str(controller), "submit", "--folder", str(arm)],
                                      text=True, env=environment).strip().splitlines()[-1].split(";", 1)[0]
        jobs[slug] = dict(job=job, status="submitted")
    write_json(campaign / "accounts" / account / "submission.json",
               dict(account=account, python=sys.executable, jobs=jobs, downstream_submitted=False))
    return jobs


def status(campaign):
    top = json.loads((campaign / "manifest.json").read_text())
    rows = []
    for entry in top["entries"]:
        manifest = json.loads((Path(entry["arm_folder"]) / "manifest.json").read_text())
        job = manifest["pretrain_entries"][0].get("job")
        state = "NOT_SUBMITTED"
        if job:
            output = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job,
                                     "--format=JobID,State,ExitCode"], capture_output=True, text=True)
            matches = [line.split("|") for line in output.stdout.splitlines() if line.split("|")[0] == job]
            state = matches[-1][1].split()[0] if matches else "UNKNOWN"
        rows.append(dict(account=entry["assigned_account"], arm=entry["slug"], job=job, state=state,
                         verified=(Path(entry["arm_folder"]) / "pretrain/verified.json").is_file()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.results_root.resolve()))
        return
    if args.campaign is None:
        parser.error(args.action + " requires --campaign")
    result = submit(args.campaign.resolve()) if args.action == "submit" else status(args.campaign.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
