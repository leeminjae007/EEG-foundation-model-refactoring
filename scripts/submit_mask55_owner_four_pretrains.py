"""Prepare and submit the four legacy encoder ablations as new mask-55 pretrains.

Run as ml10266 in the training environment. The campaign is pretrain-only;
the old mask-50/60 checkpoints and downstream results are never modified.
"""

from __future__ import annotations

from copy import deepcopy
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_shared_mask55_pretrains as shared

RESULTS = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results")
SUFFIX = "-mask55-d2-patchdim-owner-encoder-four-pretrain"
ARMS = {
    "ours-lite": "mjde_lite",
    "enc-s2t-6stage": "mjde_s2t6",
    "enc-t2s-6stage": "mjde_t2s6",
    "enc-average-3s": "mjde_average",
}


def resolved_configs():
    from ablation.config import resolve_ablation

    base = yaml.safe_load((ROOT / "configs/pretrain_gr2_patch_feature.yaml").read_text(encoding="utf-8"))
    result = {}
    for slug, encoder in ARMS.items():
        config = deepcopy(base)
        config["masking"]["mask_ratio"] = .55
        config["mae"]["decoder_depth"] = 2
        # These ablation encoders do not implement the full model's dynamic
        # patch-feature fusion. Record the matched reference without claiming
        # that a nonexistent dynamic gate is trainable in these variants.
        config["encoder"]["fusion_gate"] = "static_feature"
        config["ablation"] = {
            "name": "mask55_d2_patchdim_" + slug.replace("-", "_"),
            "encoder": encoder,
            "position": "shpe",
            "pe_scope": "both",
            "reference_fusion_gate": "patch_feature",
            "fusion_gate_applicability": "not_applicable_encoder_replaced",
        }
        config = resolve_ablation(config)
        assert config["masking"]["policy"] == "geometry_tubelet"
        assert config["masking"]["mask_ratio"] == .55
        assert config["mae"]["decoder_depth"] == 2
        assert config["optimization"]["epochs"] == 40
        result[slug] = config
    return result


def prepare(results_root=RESULTS):
    if getpass.getuser() != shared.OWNER:
        raise PermissionError("Only ml10266 may submit this owner campaign")
    matches = sorted(results_root.glob("*" + SUFFIX))
    if len(matches) > 1:
        raise ValueError("Multiple owner campaigns exist: " + str(matches))
    if matches:
        return matches[0]
    configs = resolved_configs()
    policy = yaml.safe_load((ROOT / "configs/cluster/bigpurple_a100.yaml").read_text(encoding="utf-8"))["slurm"]["pretrain"]
    commit = shared.source_commit()
    campaign = results_root / (shared.new_york_stamp() + SUFFIX)
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    shared.snapshot(source)
    entries = []
    for slug, config in configs.items():
        arm = campaign / slug
        for child in ("configs", "pretrain", "logs"):
            (arm / child).mkdir(parents=True, exist_ok=True)
        (arm / "source").symlink_to(source, target_is_directory=True)
        config["runtime"]["output"] = str(arm / "pretrain")
        config_path = arm / "configs/pretrain.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entry = {
            "kind": "pretrain", "arm": slug, "config": str(config_path),
            "config_sha256": shared.digest(config_path), "source": str(source),
            "result_dir": str(arm / "pretrain"), "job": None,
            "job_history": [], "retries": 0, "last_resume_epoch": 0,
            "max_timeout_resumes": 40, "time_limit": policy["time"],
            "gpu_partitions": policy["partitions"], "timeout_continuation": False,
        }
        manifest = {
            "experiment": "mask55-d2-patchdim-" + slug,
            "owner": shared.OWNER, "assigned_account": shared.OWNER,
            "source_commit": commit, "python": None,
            "pretrain_launcher": "slurm_flexible", "pretrain_entries": [entry],
            "pretrain_excluded_nodes": policy["excluded_nodes"],
            "resource_policy": policy, "auto_resume": False,
            "verify_before_success": True, "results_dir": str(arm / "pretrain/report"),
            "status": "prepared", "downstream_submitted": False,
        }
        shared.write_json(arm / "manifest.json", manifest)
        worker = arm / "worker.sh"
        worker.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "exec \"${EEGFM_PYTHON:?activate the training environment before sbatch}\" "
            + str(source / "scripts/gr2_pretrain_campaign.py")
            + " worker --folder " + str(arm) + "\n", encoding="utf-8")
        worker.chmod(0o755)
        entries.append({"arm": slug, "encoder": ARMS[slug], "folder": str(arm),
                        "config_sha256": entry["config_sha256"]})
    shared.write_json(campaign / "manifest.json", {
        "owner": shared.OWNER, "source_commit": commit, "source": str(source),
        "settings": {"mask_ratio": .55, "decoder_depth": 2,
                     "reference_fusion_gate": "patch_feature", "epochs": 40,
                     "seed": 42, "gpu": "a100", "downstream": False},
        "entries": entries, "status": "prepared",
    })
    return campaign


def submit(campaign):
    if getpass.getuser() != shared.OWNER:
        raise PermissionError("Only ml10266 may submit this owner campaign")
    shared.require_environment()
    top = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    if not shared.checkout_contains(top["source_commit"]):
        raise ValueError("Frozen campaign source commit is not in the checkout")
    jobs = {}
    controller = campaign / "source/scripts/gr2_pretrain_campaign.py"
    for slug, expected_encoder in ARMS.items():
        arm = campaign / slug
        manifest_path = arm / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["pretrain_entries"][0]
        if entry.get("job"):
            jobs[slug] = entry["job"]
            continue
        config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
        if (shared.digest(entry["config"]) != entry["config_sha256"]
                or manifest["source_commit"] != top["source_commit"]
                or config["ablation"]["encoder"] != expected_encoder
                or config["masking"]["mask_ratio"] != .55
                or config["mae"]["decoder_depth"] != 2):
            raise ValueError("Frozen config mismatch: " + slug)
        manifest["python"] = sys.executable
        shared.write_json(manifest_path, manifest)
        environment = dict(os.environ, EEGFM_PYTHON=sys.executable)
        job = subprocess.check_output(
            [sys.executable, str(controller), "submit", "--folder", str(arm)],
            text=True, env=environment).strip().splitlines()[-1].split(";", 1)[0]
        jobs[slug] = job
    shared.write_json(campaign / "submission.json", {"python": sys.executable, "jobs": jobs})
    return jobs


def main():
    shared.require_environment()
    campaign = prepare()
    print(json.dumps({"campaign": str(campaign), "jobs": submit(campaign)}, indent=2))


if __name__ == "__main__":
    main()
