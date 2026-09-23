"""Submit only missing final-default downstream seeds from a collaborator account.

The owner updates the shared checkout.  Collaborators run this script once from
their own activated environment: --account hk4935 or --account yc8820.
--plan performs a read-only inventory and never submits jobs.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results")
SHARED = RESULTS / "260921-1456-mask55-d2-patchdim-shared-pretrain"
STAGE3 = RESULTS / "260922-0923-mask55-d2-singlepath3-pretrain"
SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ("chb", "tuev", "siena", "tusz", "seedv", "faced",
            "mentalarithmetic", "physionet_mi", "isruc", "hmc")
HP = {
    **{name: (1e-4, .01, .1) for name in ("chb", "faced", "seedv", "isruc")},
    "tuev": (1e-4, .01, .3),
    "mentalarithmetic": (1e-4, .02, .1),
    "physionet_mi": (1e-4, .02, .1),
    "siena": (2.5e-5, .005, .1),
    "hmc": (5e-5, .005, .3),
    "tusz": (2e-4, .005, .3),
}
ARMS = {
    "hk4935": {
        "pe-ch_order": (SHARED / "accounts/hk4935/pe-ch_order",
                        SHARED / "accounts/hk4935/fourtask_downstream/pe-ch_order", "a100", .55),
        "pe-acpe": (SHARED / "accounts/hk4935/pe-acpe",
                    SHARED / "accounts/hk4935/fourtask_downstream/pe-acpe", "a100", .55),
        "pe-4dREVE": (SHARED / "accounts/hk4935/pe-4dREVE",
                      SHARED / "accounts/hk4935/fourtask_downstream/pe-4dREVE", "a100", .55),
        "ours-lite": (RESULTS / "enc-lite", RESULTS / "enc-lite", "gl40s", .60),
        "enc-average-3s": (RESULTS / "enc-average-3s", None, "gl40s", .60),
    },
    "yc8820": {
        "cbramod": (SHARED / "accounts/yc8820/cbramod",
                    SHARED / "accounts/yc8820/fourtask_downstream/cbramod", "a100", .55),
        "csbrain": (SHARED / "accounts/yc8820/csbrain",
                    SHARED / "accounts/yc8820/fourtask_downstream/csbrain", "a100", .55),
        "labram": (SHARED / "accounts/yc8820/labram",
                   SHARED / "accounts/yc8820/fourtask_downstream/labram", "a100", .55),
        "enc-s2t-3stage": (STAGE3 / "accounts/yc8820/s2t3", None, "gl40s", .55),
        "enc-t2s-3stage": (STAGE3 / "accounts/yc8820/t2s3", None, "gl40s", .55),
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
    except (OSError, ValueError):
        return False
    selected = result.get("balanced_accuracy", {})
    return (isinstance(selected.get("selection", {}).get("score"), (int, float))
            and isinstance(selected.get("test", {}).get("balanced_accuracy"), (int, float)))


def matching_old(old: Path | None, dataset: str, seed: int) -> bool:
    if old is None:
        return False
    result = old / "downstream" / dataset / f"seed-{seed}" / "result.json"
    config = old / "configs/downstream" / f"{dataset}_seed{seed}.yaml"
    if not complete(result) or not config.is_file():
        return False
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    opt = cfg["optimization"]
    actual = (opt["encoder_learning_rate"], opt["weight_decay"],
              cfg["model"]["head_dropout"])
    return (all(abs(a - b) < 1e-12 for a, b in zip(actual, HP[dataset]))
            and all(abs(opt[name + "_learning_rate"] - HP[dataset][0]) < 1e-12
                    for name in ("tokenizer", "head")))


def verify_checkpoint(original: Path, ratio: float) -> tuple[Path, dict]:
    checkpoint = original / "pretrain/checkpoint-epoch-0040.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    proof_path = original / "pretrain/verified.json"
    if proof_path.is_file():
        proof = read_json(proof_path)
        if (not proof.get("strict_load") or proof.get("partial_epoch_smoke")
                or proof.get("checkpoint") != str(checkpoint)
                or proof.get("sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            raise ValueError(f"Invalid verified checkpoint: {original}")
        config = yaml.safe_load((original / "configs/pretrain.yaml").read_text(encoding="utf-8"))
    else:
        import torch
        from ablation.models import build_pretrain
        saved = torch.load(checkpoint, map_location="cpu")
        config = saved["config"]
        if (saved.get("epoch") != 40 or saved.get("extra", {}).get("partial_epoch_smoke")
                or not saved.get("extra", {}).get("dataset_fingerprint")):
            raise ValueError(f"Pretrain is incomplete: {original}")
        model = build_pretrain(config, torch.device("cpu"))
        model.load_state_dict(saved["model"], strict=True)
        proof = dict(checkpoint=str(checkpoint),
                     sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     epoch=40, strict_load=True, partial_epoch_smoke=False)
    if (abs(config["masking"]["mask_ratio"] - ratio) > 1e-12
            or config["mae"]["decoder_depth"] != 2):
        raise ValueError(f"Unexpected masking/depth: {original}")
    return checkpoint, proof


def resources(gpu: str, dataset: str) -> dict:
    long = dataset in ("chb", "tuev", "faced")
    partitions = f"{gpu}_short,{gpu}_long" if long else f"{gpu}_dev,{gpu}_short,{gpu}_long"
    excludes = (yaml.safe_load((ROOT / "configs/cluster/bigpurple_a100.yaml").read_text())
                ["slurm"]["downstream"]["excluded_nodes"] if gpu == "a100"
                else ["gl40s-8013"])
    return dict(partitions=partitions, time="12:00:00" if long else "04:00:00",
                gpu=gpu, cpus_per_task=4, memory="32G", array_parallelism=5,
                excluded_nodes=excludes)


def inventory(account: str) -> list[dict]:
    rows = []
    for arm, (original, old, gpu, ratio) in ARMS[account].items():
        for dataset in DATASETS:
            for seed in SEEDS:
                rows.append(dict(arm=arm, original=original, old=old, gpu=gpu,
                                 ratio=ratio, dataset=dataset, seed=seed,
                                 reuse=matching_old(old, dataset, seed)))
    return rows


def ensure_no_old_jobs(old: Path, dataset: str) -> None:
    manifest_path = old / "manifest.json"
    if not manifest_path.is_file():
        return
    for job in read_json(manifest_path).get("downstream_jobs", []):
        if job.get("dataset") != dataset:
            continue
        job_id = str(job["job"])
        status = subprocess.check_output(["squeue", "-h", "-j", job_id, "-o", "%T"], text=True).strip()
        if status:
            raise RuntimeError(f"Old {dataset} job {job_id} remains active: {status}")


def remove_mismatched_old(old: Path | None, dataset: str, seed: int) -> None:
    if old is None:
        return
    target = old / "downstream" / dataset / f"seed-{seed}"
    if not target.exists() or matching_old(old, dataset, seed):
        return
    ensure_no_old_jobs(old, dataset)
    parent = (old / "downstream" / dataset).resolve()
    if target.is_symlink() or target.resolve().parent != parent:
        raise ValueError(f"Unsafe old-result target: {target}")
    shutil.rmtree(target)
    print(f"removed outdated result: {target}", flush=True)


def prepare_stage(account: str, arm: str, rows: list[dict]) -> tuple[Path, list[dict]]:
    original, old, gpu, ratio = ARMS[account][arm]
    checkpoint, proof = verify_checkpoint(original, ratio)
    stage = SHARED / "accounts" / account / "final_default_downstream" / arm
    manifest_path = stage / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get("source_pretrain") != str(original)
                or manifest.get("checkpoint_sha256") != proof["sha256"]
                or manifest.get("hyperparameters") !=
                {name: list(value) for name, value in HP.items()}):
            raise ValueError(f"Existing stage provenance differs: {stage}")
    else:
        stage.mkdir(parents=True)
        (stage / "source").symlink_to(ROOT, target_is_directory=True)
        (stage / "pretrain").mkdir()
        (stage / "pretrain/checkpoint-epoch-0040.pth").symlink_to(checkpoint)
        stage_proof = dict(proof, checkpoint=str(stage / "pretrain/checkpoint-epoch-0040.pth"))
        (stage / "pretrain/verified.json").write_text(json.dumps(stage_proof, indent=2) + "\n")
        (stage / "configs").mkdir()
        config_path = original / "configs/pretrain.yaml"
        if config_path.is_file():
            shutil.copy2(config_path, stage / "configs/pretrain.yaml")
        else:
            import torch
            saved = torch.load(checkpoint, map_location="cpu")
            (stage / "configs/pretrain.yaml").write_text(
                yaml.safe_dump(saved["config"], sort_keys=False), encoding="utf-8")
        manifest = dict(preset=arm, owner=account, source_pretrain=str(original),
                        historical_results=str(old) if old else None,
                        checkpoint_sha256=proof["sha256"],
                        hyperparameters={name: list(value) for name, value in HP.items()},
                        selection="validation balanced_accuracy",
                        test_role="reporting only", downstream_jobs=[])
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for row in rows:
        if not row["reuse"]:
            remove_mismatched_old(old, row["dataset"], row["seed"])
    from scripts.submit_experiment_downstream import prepare
    fixed = {name: dict(learning_rate=lr, weight_decay=wd, head_dropout=drop)
             for name, (lr, wd, drop) in HP.items()}
    entries = prepare(stage, stage / "pretrain/checkpoint-epoch-0040.pth",
                      DATASETS, arm, fixed)
    missing = []
    for index, entry in enumerate(entries):
        row = next(r for r in rows if r["dataset"] == entry["dataset"] and r["seed"] == entry["seed"])
        if row["reuse"]:
            source_output = old / "downstream" / entry["dataset"] / f"seed-{entry['seed']}"
            target_output = Path(entry["output"])
            target_output.parent.mkdir(parents=True, exist_ok=True)
            if not target_output.exists() and not target_output.is_symlink():
                target_output.symlink_to(source_output, target_is_directory=True)
            elif not target_output.is_symlink() or target_output.resolve() != source_output.resolve():
                raise ValueError(f"Reused output is not the audited source: {target_output}")
            continue
        if not complete(Path(entry["output"]) / "result.json"):
            missing.append(dict(index=index, **entry))
    return stage, missing


def submit_stage(stage: Path, missing: list[dict], gpu: str, retry: bool) -> None:
    manifest_path = stage / "manifest.json"
    manifest = read_json(manifest_path)
    registered = {j["dataset"] for j in manifest["downstream_jobs"]}
    for dataset in DATASETS:
        current = [r for r in missing if r["dataset"] == dataset]
        if not current or (dataset in registered and not retry):
            continue
        if dataset in registered:
            for previous in manifest["downstream_jobs"]:
                if previous["dataset"] != dataset:
                    continue
                status = subprocess.check_output(
                    ["squeue", "-h", "-j", str(previous["job"]), "-o", "%T"],
                    text=True).strip()
                if status:
                    raise RuntimeError(f"{dataset}: prior array {previous['job']} is still active")
        policy = resources(gpu, dataset)
        logs = stage / "downstream/logs"
        logs.mkdir(parents=True, exist_ok=True)
        wrap = shlex.join(["srun", "--ntasks=1", f"--gpus-per-task={gpu}:1",
                            "--gpu-bind=single:1", "--kill-on-bad-exit=1",
                            sys.executable, str(ROOT / "scripts/downstream_experiment_worker.py"),
                            "--experiment", str(stage)])
        command = ["sbatch", "--parsable", "--account=system",
                   "--job-name=final-" + stage.name + "-" + dataset,
                   "--partition=" + policy["partitions"], "--nodes=1", "--ntasks=1",
                   f"--gpus-per-task={gpu}:1", "--cpus-per-task=4", "--mem=32G",
                   "--time=" + policy["time"],
                   "--array=" + ",".join(str(r["index"]) for r in current) + "%5",
                   "--output=" + str(logs / "%A_%a.out"),
                   "--error=" + str(logs / "%A_%a.err"), "--wrap=" + wrap]
        if policy["excluded_nodes"]:
            command.append("--exclude=" + ",".join(policy["excluded_nodes"]))
        job = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
        if not re.fullmatch(r"\d+", job):
            raise ValueError(f"Unexpected sbatch response: {job}")
        manifest["downstream_jobs"].append(dict(dataset=dataset, job=job,
                                                 indices=[r["index"] for r in current],
                                                 resources=policy))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{stage.name} {dataset} {job} {len(current)} seeds", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, choices=ARMS)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--retry", action="store_true",
                        help="After previous arrays finish, resubmit only still-missing seeds")
    args = parser.parse_args()
    rows = inventory(args.account)
    for arm in ARMS[args.account]:
        selected = [r for r in rows if r["arm"] == arm]
        print(f"{arm}: reuse {sum(r['reuse'] for r in selected)}/50; submit {sum(not r['reuse'] for r in selected)}")
    if args.plan:
        return
    if getpass.getuser() != args.account:
        parser.error(f"Must submit from {args.account}")
    for arm, (_, _, gpu, _) in ARMS[args.account].items():
        selected = [r for r in rows if r["arm"] == arm]
        stage, missing = prepare_stage(args.account, arm, selected)
        submit_stage(stage, missing, gpu, args.retry)


if __name__ == "__main__":
    main()
