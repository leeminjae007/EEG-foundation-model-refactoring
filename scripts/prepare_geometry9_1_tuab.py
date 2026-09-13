"""Freeze the five-seed TUAB handoff for the geometry-only GR9-1 pretrain."""

import hashlib
import json
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "outputs/geometry50_reve3cm_t2_15_20260912"
CAMPAIGN = ROOT / "outputs/geometry9_1_tuab_dropout01_20260912"
CHECKPOINT = PRETRAIN / "training/checkpoint-epoch-0040.pth"
PRETRAIN_JOB = "27393348"
SEEDS = (42, 1234, 696, 1001, 3407)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    CAMPAIGN.mkdir(exist_ok=True)
    manifest_path = CAMPAIGN / "manifest.json"
    if manifest_path.exists():
        print(manifest_path)
        return

    source = CAMPAIGN / "source"
    if source.exists():
        raise RuntimeError("source snapshot exists without manifest; inspect manually")
    source.mkdir()
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "finetune.py", source / "finetune.py")
    (source / "configs").mkdir()

    base_path = ROOT / "configs/downstream/gr9-1_warmup5_tuab_seed42.yaml"
    base = yaml.safe_load(base_path.read_text())
    assert base["model"]["head_dropout"] == 0.1
    assert base["optimization"]["warmup_epochs"] == 5
    assert base["optimization"]["class_counts"] == [147894, 149209]
    assert all(base["optimization"][key] == 1e-5 for key in
               ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate"))

    entries = []
    for seed in SEEDS:
        config = yaml.safe_load(base_path.read_text())
        config["seed"] = seed
        config["model"]["checkpoint"] = str(CHECKPOINT)
        result_dir = CAMPAIGN / "downstream" / f"tuab_seed{seed}"
        config["runtime"]["output"] = str(result_dir)
        config_path = source / "configs" / f"geometry9-1_tuab_dropout01_seed{seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        entries.append({"index": len(entries), "seed": seed, "config": str(config_path),
                        "result_dir": str(result_dir), "config_sha256": sha256(config_path)})

    array_path = CAMPAIGN / "array.json"
    atomic_json(array_path, entries)
    runner = source / "run_array.py"
    runner.write_text('''import json, os, subprocess, sys\nfrom pathlib import Path\nroot = Path(__file__).resolve().parent\nentries = json.loads((root.parent / "array.json").read_text())\nentry = entries[int(os.environ["SLURM_ARRAY_TASK_ID"])]\nsubprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",\n                str(root / "finetune.py"), "--config", entry["config"], "--distributed"],\n               check=True, cwd=root)\n''')

    files = [path for path in source.rglob("*") if path.is_file()]
    manifest = {
        "experiment": "geometry9-1 TUAB head dropout 0.1",
        "pretrain_job_id": PRETRAIN_JOB,
        "pretrain_checkpoint": str(CHECKPOINT),
        "scientific_delta_from_gr9_1": "pretraining mask only: geometry tubelet, ratio 0.5, Euclidean radius 0.03 m, duration 2-15 patches",
        "tuab": {
            "seeds": list(SEEDS), "head_dropout": 0.1, "warmup_epochs": 5,
            "learning_rates": {"tokenizer": 1e-5, "encoder": 1e-5, "head": 1e-5},
            "epochs": 20, "batch_size_per_gpu": 64, "gradient_accumulation_steps": 8,
            "loss": "weighted cross entropy", "class_counts": [147894, 149209],
            "selectors": ["balanced_accuracy", "auroc"]
        },
        "array": str(array_path), "tasks": len(entries), "max_concurrent_tasks": 3,
        "source_files": [{"path": str(path.relative_to(source)), "sha256": sha256(path)}
                         for path in sorted(files)],
    }
    atomic_json(manifest_path, manifest)
    for path in files:
        path.chmod(0o444)
    print(manifest_path)


if __name__ == "__main__":
    prepare()
