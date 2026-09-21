"""Prepare one isolated GR2 pretrain experiment and submit its A100 job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path); parser.add_argument("--config", default="configs/pretrain_gr2_geometry.yaml"); parser.add_argument("--preset", choices=("gr2-mjde-d4-geometry", "gr2-d2-static", "gr2-d2-patch-scalar", "gr2-d2-patch-dimension", "gr2-d2-patch-dimension-mask55", "gr2-d2-patch-dimension-mask60", "gr2-d4-patch-dimension-mask55", "gr2-d4-patch-dimension-mask60")); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); experiment = args.experiment.resolve(); source = experiment / "source"
    if not (source / "ablation/pretrain.py").is_file(): raise FileNotFoundError("create_experiment_layout.py must run first")
    manifest_path = experiment / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    if manifest.get("pretrain_job"): raise ValueError("pretrain already submitted: " + str(manifest["pretrain_job"]))
    original = source / args.config
    config = yaml.safe_load(original.read_text()); policy = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]["pretrain"]
    profile = {
        "gr2-mjde-d4-geometry": (4, "static_feature"),
        "gr2-d2-static": (2, "static_feature"),
        "gr2-d2-patch-scalar": (2, "patch_scalar"),
        "gr2-d2-patch-dimension": (2, "patch_feature"),
        "gr2-d2-patch-dimension-mask55": (2, "patch_feature"),
        "gr2-d2-patch-dimension-mask60": (2, "patch_feature"),
        "gr2-d4-patch-dimension-mask55": (4, "patch_feature"),
        "gr2-d4-patch-dimension-mask60": (4, "patch_feature"),
    }.get(args.preset)
    if profile:
        depth, gate = profile
        config["mae"]["decoder_depth"] = depth
        config["encoder"]["fusion_gate"] = gate
    if args.preset in ("gr2-d2-patch-dimension-mask55", "gr2-d2-patch-dimension-mask60", "gr2-d4-patch-dimension-mask55", "gr2-d4-patch-dimension-mask60"):
        config["masking"]["mask_ratio"] = 0.55 if args.preset.endswith("mask55") else 0.60
    sys.path.insert(0, str(source))
    from ablation.bootstrap import ensure_data_imports
    ensure_data_imports()
    from ablation.config import resolve_ablation
    config = resolve_ablation(config)
    config["runtime"]["output"] = str(experiment / "pretrain")
    frozen = experiment / "configs/pretrain.yaml"; frozen.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    controller = experiment / "controller"; controller.mkdir(exist_ok=True); controller_file = controller / "gr2_pretrain_campaign.py"; shutil.copy2(source / "scripts/gr2_pretrain_campaign.py", controller_file)
    entry = {"kind": "pretrain", "arm": experiment.name, "config": str(frozen), "config_sha256": digest(frozen), "source": str(source), "result_dir": str(experiment / "pretrain"), "job": None, "job_history": [], "retries": 0, "last_resume_epoch": 0, "max_timeout_resumes": 40, "time_limit": policy["time"], "gpu_partitions": policy["partitions"]}
    manifest.update({"python": sys.executable, "pretrain_launcher": "slurm_flexible", "preset": args.preset, "results_dir": str(experiment / "pretrain/report"), "pretrain_entries": [entry], "resource_policy": policy})
    manifest.update(alias=experiment.name, monitor_interval_seconds=3600,
                    pretrain_excluded_nodes=sorted(set(manifest.get("pretrain_excluded_nodes", [])) | set(policy.get("excluded_nodes", []))))
    manifest["auto_resume"] = bool(policy.get("auto_resume", False))
    manifest["verify_before_success"] = True
    entry["timeout_continuation"] = manifest["auto_resume"]
    worker = experiment / "worker.sh"; worker.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + sys.executable + " " + str(controller_file) + " worker --folder " + str(experiment) + "\n", encoding="utf-8"); worker.chmod(0o750)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only: print(experiment); return
    job = subprocess.check_output([sys.executable, str(controller_file), "submit", "--folder", str(experiment)], text=True).strip()
    manifest = json.loads(manifest_path.read_text()); manifest["pretrain_job"] = job; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"); print(job)
if __name__ == "__main__": main()
