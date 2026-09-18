"""Bounded, validation-only BCIC-IV-2a beam search, followed by five-seed testing."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

import yaml
from ablation.tuab.campaign import CHECKPOINT, CHECKPOINT_SHA256, SEEDS, digest, verify_source, write_json
from ablation.bciciv2a.connection import connected_source, without_removed_seedvig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = "outputs/bciciv2a_beam_20260915"
BASELINE = "outputs/nearest3_7_downstream_20260913"
SEARCH_SEEDS = SEEDS[:3]
SPACE = {"backbone_lr": [5e-5, 1e-4, 2e-4], "head_lr": [5e-5, 1e-4, 2e-4],
         "weight_decay": [.005, .01, .05], "dropout": [.1, .2, .3]}
ANCHOR = {"backbone_lr": 1e-4, "head_lr": 1e-4, "weight_decay": .01, "dropout": .2}
BEAM_WIDTH, MAX_ROUNDS, MAX_CANDIDATES = 2, 3, 25


def candidate_id(hp):
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:12]


def neighbors(beams, seen):
    """Expand adjacent values, alternating beam parents within each dimension."""
    candidates = []
    emitted = set(seen)
    for key, values in SPACE.items():
        for hp in beams:
            index = values.index(hp[key])
            for next_index in (index - 1, index + 1):
                if not 0 <= next_index < len(values):
                    continue
                new = dict(hp, **{key: values[next_index]})
                key_id = candidate_id(new)
                if key_id not in emitted:
                    candidates.append(new)
                    emitted.add(key_id)
    return candidates


def validate_recipe(config):
    opt, model = config["optimization"], config["model"]
    if config["data"]["dataset"] != "bciciv2a" or config["seed"] not in SEEDS:
        raise ValueError("Only BCIC-IV-2a and the five fixed seeds are authorized")
    fixed = {"epochs": 50, "batch_size_per_gpu": 64,
             "gradient_accumulation_steps": 1, "min_learning_rate": 1e-6,
             "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
             "label_smoothing": .1, "gradient_clip_norm": 1.0}
    if any(opt[k] != v for k, v in fixed.items()):
        raise ValueError("Fixed recipe changed")
    hp = {"backbone_lr": opt["encoder_learning_rate"], "head_lr": opt["head_learning_rate"],
          "weight_decay": opt["weight_decay"], "dropout": model["head_dropout"]}
    if any(hp[k] not in values for k, values in SPACE.items()):
        raise ValueError("Candidate is outside the search space")
    if opt["tokenizer_learning_rate"] != hp["backbone_lr"] or model["head_hidden_tokens"] is not None:
        raise ValueError("Tokenizer/encoder LR must match; keep the original H4 head")
    if not isinstance(config["runtime"]["evaluate_test"], bool):
        raise ValueError("Explicit test-evaluation policy required")
    return hp


def make_config(base, hp, output, evaluate_test=False):
    config = deepcopy(base)
    opt = config["optimization"]
    opt.pop("warmup_epochs", None)
    opt.pop("warmup_ratio", None)
    opt.update(tokenizer_learning_rate=hp["backbone_lr"],
               encoder_learning_rate=hp["backbone_lr"], head_learning_rate=hp["head_lr"],
               weight_decay=hp["weight_decay"])
    config["model"]["head_dropout"] = hp["dropout"]
    config["runtime"].update(output=str(output), evaluate_test=evaluate_test)
    validate_recipe(config)
    return config


def prepare(campaign):
    if (campaign / "manifest.json").exists():
        return verify_source(campaign)
    baseline = ROOT / BASELINE
    old = json.loads((baseline / "manifest.json").read_text())
    if digest(baseline / "array.json") != old["array_sha256"]:
        raise ValueError("Baseline array changed")
    removed = {"src/data/datasets/seedvig_dataset.py", "src/data/preprocessing/preprocessing_seedvig.py"}
    removed.update("configs/nearest3_7_seedvig_seed%d.yaml" % seed for seed in SEEDS)
    missing_excluded = []
    for entry in old["source_files"]:
        path = baseline / "source" / entry["path"]
        if not path.exists() and entry["path"] in removed:
            missing_excluded.append(entry["path"])
            continue
        if digest(path) != entry["sha256"]:
            raise ValueError("Baseline source changed")
    checkpoint = ROOT / CHECKPOINT
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Wrong nearest3-7 epoch40 checkpoint")
    array = json.loads((baseline / "array.json").read_text())
    bases = []
    for seed in SEEDS:
        match = [e for e in array if e["dataset"] == "bciciv2a" and e["seed"] == seed]
        if len(match) != 1:
            raise ValueError("Expected one baseline per seed")
        e = match[0]
        config = yaml.safe_load(Path(e["config"]).read_text())
        if digest(e["config"]) != e["config_sha256"] or config["seed"] != seed:
            raise ValueError("Baseline config changed")
        if Path(config["model"]["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("Wrong baseline weight")
        if not (Path(e["result_dir"]) / "result.json").exists():
            raise ValueError("Baseline incomplete")
        make_config(config, ANCHOR, campaign / "unused")
        bases.append(e)
    source = campaign / "source"
    source.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(baseline / "source/src", source / "src", ignore=ignore)
    engine = source / "src/training/engine.py"
    original = engine.read_text()
    engine.chmod(0o644)
    engine.write_text(connected_source(original))
    registry = source / "src/data/datasets/registry.py"
    registry.chmod(0o644)
    registry.write_text(without_removed_seedvig(registry.read_text()))
    (source / "ablation").mkdir()
    shutil.copy2(ROOT / "ablation/__init__.py", source / "ablation/__init__.py")
    for package in ("tuab", "bciciv2a"):
        shutil.copytree(ROOT / "ablation" / package, source / "ablation" / package, ignore=ignore)
    (source / "baseline_configs").mkdir()
    for e in bases:
        shutil.copy2(e["config"], source / "baseline_configs" / (str(e["seed"]) + ".yaml"))
    files = [{"path": f.relative_to(source).as_posix(), "sha256": digest(f)}
             for f in sorted(source.rglob("*")) if f.is_file()]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "repo_root": str(ROOT),
                "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
                "python": str(ROOT / ".venv/bin/python"), "source_files": files,
                "source_origin": str(baseline / "source/src"), "baseline_entries": bases,
                "engine_connection": "Only gate test dataset creation/evaluation; all training and validation unchanged",
                "excluded_missing_baseline_files": missing_excluded,
                "registry_connection": "Remove the two SEED-VIG registry lines after its external removal",
                "baseline_engine_sha256": digest(baseline / "source/src/training/engine.py"),
                "space": SPACE, "anchor": ANCHOR, "beam_width": BEAM_WIDTH,
                "max_rounds": MAX_ROUNDS, "max_candidates": MAX_CANDIDATES,
                "search_seeds": SEARCH_SEEDS, "final_seeds": SEEDS,
                "selection": "Mean of all-epoch best validation BAcc over all three search seeds; no test access",
                "tie_break": "Lower population SD, then candidate ID", "time_limit": "04:00:00"}
    write_json(campaign / "manifest.json", manifest)
    for f in files:
        (source / f["path"]).chmod(0o444)
    return manifest


def stage_dir(campaign, stage):
    if stage not in ["round_%02d" % i for i in range(MAX_ROUNDS)] + ["final"]:
        raise ValueError("Invalid stage")
    return campaign / "stages" / stage


def plan_stage(campaign, stage, candidates):
    directory = stage_dir(campaign, stage)
    if (directory / "plan.json").exists():
        plan = load_plan(campaign, stage)
        if plan["candidates"] != candidates:
            raise ValueError("Existing stage has different candidates")
        return plan
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    entries = []
    for hp in candidates:
        for seed in (SEEDS if stage == "final" else SEARCH_SEEDS):
            base = yaml.safe_load((campaign / "source/baseline_configs" / (str(seed) + ".yaml")).read_text())
            output = directory / "runs" / (candidate_id(hp) + "_seed" + str(seed))
            config = make_config(base, hp, output, stage == "final")
            path = directory / "configs" / (output.name + ".yaml")
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            entries.append({"index": len(entries), "candidate": candidate_id(hp), "hp": hp,
                            "seed": seed, "config": str(path), "config_sha256": digest(path),
                            "output": str(output)})
            path.chmod(0o444)
    plan = {"stage": stage, "candidates": candidates, "entries": entries,
            "manifest_sha256": digest(campaign / "manifest.json")}
    write_json(directory / "plan.json", plan)
    (directory / "plan.json").chmod(0o444)
    return plan


def load_plan(campaign, stage):
    plan = json.loads((stage_dir(campaign, stage) / "plan.json").read_text())
    if plan["manifest_sha256"] != digest(campaign / "manifest.json"):
        raise ValueError("Manifest changed")
    for e in plan["entries"]:
        if digest(e["config"]) != e["config_sha256"]:
            raise ValueError("Candidate config changed")
        cfg = yaml.safe_load(Path(e["config"]).read_text())
        if validate_recipe(cfg) != e["hp"] or cfg["runtime"]["evaluate_test"] != (stage == "final"):
            raise ValueError("Candidate/test policy mismatch")
    return plan


def submit_once(path, command):
    if path.exists():
        return json.loads(path.read_text())["job_id"]
    intent = path.with_name(path.stem + "_intent.json")
    if intent.exists():
        raise RuntimeError("Prior ambiguous submission: reconcile " + str(intent))
    write_json(intent, {"command": command, "utc": datetime.now(timezone.utc).isoformat()})
    result = subprocess.run(command, capture_output=True, text=True)
    write_json(path.with_name(path.stem + "_response.json"),
               {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    job = result.stdout.strip().split(";")[0]
    if result.returncode or not job.isdigit():
        raise RuntimeError("Slurm submission failed/ambiguous: " + result.stderr)
    write_json(path, {"job_id": job, "command": command, "utc": datetime.now(timezone.utc).isoformat()})
    return job


def submit_stage(campaign, stage):
    manifest = verify_source(campaign)
    plan = load_plan(campaign, stage)
    smoke = json.loads((campaign / "smoke_validation.json").read_text())
    if not smoke["passed"] or smoke["manifest_sha256"] != digest(campaign / "manifest.json"):
        raise ValueError("Snapshot has not passed smoke validation")
    directory = stage_dir(campaign, stage)
    common = ["sbatch", "--parsable", "--account=system", "--nodes=1", "--ntasks=1",
              "--chdir=" + str(campaign / "source")]
    env = ("unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
           "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; ")
    runner = [manifest["python"], "-m", "ablation.bciciv2a.campaign", "run-entry", "--campaign", str(campaign), "--stage", stage]
    wrapper = (env + 'export TMPDIR="/tmp/bcbeam-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"; '
               'exec srun --ntasks=1 --gpus-per-task=l40s:1 --cpu-bind=none --kill-on-bad-exit=1 ' + shlex.join(runner))
    command = common + ["--job-name=bcbeam-" + stage + "-20260915", "--partition=gl40s_dev,gl40s_short,gl40s_long",
        "--array=0-%d%%10" % (len(plan["entries"]) - 1), "--gpus-per-task=l40s:1", "--cpus-per-task=8",
        "--mem=32G", "--time=" + manifest["time_limit"], "--output=" + str(directory / "slurm-%A_%a.log"), "--wrap", wrapper]
    gpu = submit_once(directory / "submission.json", command)
    follow = [manifest["python"], "-m", "ablation.bciciv2a.campaign", "advance", "--campaign", str(campaign), "--stage", stage]
    controller = common + ["--job-name=bcbeam-next-" + stage, "--partition=cpu_short,cpu_long", "--cpus-per-task=2",
        "--mem=4G", "--time=00:15:00", "--dependency=afterany:" + gpu, "--kill-on-invalid-dep=yes",
        "--output=" + str(directory / "controller-%j.log"), "--wrap", env + "exec " + shlex.join(follow)]
    control = submit_once(directory / "controller_submission.json", controller)
    record = {"stage": stage, "gpu_job_id": gpu, "controller_job_id": control,
              "candidates": len(plan["candidates"]), "runs": len(plan["entries"])}
    write_json(campaign / "status.json", dict(record, status="submitted"))
    print(json.dumps(record), flush=True)
    return record


def run_entry(campaign, stage):
    manifest = verify_source(campaign)
    plan = load_plan(campaign, stage)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(plan["entries"]):
        raise ValueError("Array index outside plan")
    e = plan["entries"][index]
    output = Path(e["output"])
    if (output / "result.json").exists():
        return
    command = [manifest["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
               "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=bcbeam-%s-%s" % (os.environ["SLURM_JOB_ID"], index),
               "--module", "ablation.bciciv2a.finetune", "--config", e["config"]]
    if (output / "last.pth").exists():
        command += ["--resume", str(output / "last.pth")]
    subprocess.run(command, cwd=campaign / "source", check=True)


def read_scores(campaign, stage):
    plan = load_plan(campaign, stage)
    scores, failures = [], []
    for hp in plan["candidates"]:
        rows = []
        for e in plan["entries"]:
            if e["candidate"] != candidate_id(hp):
                continue
            try:
                output = Path(e["output"])
                cfg = yaml.safe_load(Path(e["config"]).read_text())
                if yaml.safe_load((output / "resolved_config.yaml").read_text()) != cfg:
                    raise ValueError("Executed config differs")
                result = json.loads((output / "result.json").read_text())
                history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
                if [r["epoch"] for r in history] != list(range(1, 51)):
                    raise ValueError("Expected all 50 validation epochs")
                value = result["balanced_accuracy"]["selection"]["score"]
                if not math.isfinite(value) or value != max(r["balanced_accuracy"] for r in history):
                    raise ValueError("Best validation BAcc mismatch")
                row = {"seed": e["seed"], "validation_bacc": value, "output": e["output"]}
                if stage == "final":
                    row["test"] = {k: v for k, v in result["balanced_accuracy"]["test"].items()
                                   if isinstance(v, (int, float)) and k in ("balanced_accuracy", "kappa", "weighted_f1")}
                elif any(v["test"] is not None for v in result.values()):
                    raise ValueError("Search candidate accessed test data")
                rows.append(row)
            except (OSError, ValueError, KeyError) as error:
                failures.append({"entry": e, "error": str(error)})
        expected = len(SEEDS if stage == "final" else SEARCH_SEEDS)
        if len(rows) == expected:
            scores.append({"candidate": candidate_id(hp), "hp": hp, "stage": stage, "rows": rows,
                "validation_mean": statistics.mean(r["validation_bacc"] for r in rows),
                "validation_sd": statistics.pstdev(r["validation_bacc"] for r in rows)})
    return {"scores": scores, "failures": failures}


def rank_scores(scores):
    return sorted(scores, key=lambda r: (-r["validation_mean"], r["validation_sd"], r["candidate"]))


def advance(campaign, stage):
    verify_source(campaign)
    report = read_scores(campaign, stage)
    write_json(stage_dir(campaign, stage) / "scores.json", report)
    if stage == "final":
        if len(report["scores"]) != 1:
            write_json(campaign / "status.json", {"status": "final_incomplete", **report})
            raise RuntimeError("Final five-seed validation incomplete; inspect scores.json")
        final = report["scores"][0]
        final["test_summary"] = {k: {"mean": statistics.mean(r["test"][k] for r in final["rows"]),
                                     "sd": statistics.pstdev(r["test"][k] for r in final["rows"])}
                                 for k in final["rows"][0]["test"]}
        write_json(campaign / "final_summary.json", final)
        write_json(campaign / "status.json", {"status": "complete", "best_hp": final["hp"]})
        print(json.dumps(final), flush=True)
        return
    round_number = int(stage[-2:])
    all_scores, seen = [], set()
    for i in range(round_number + 1):
        previous = "round_%02d" % i
        plan = load_plan(campaign, previous)
        seen.update(candidate_id(hp) for hp in plan["candidates"])
        all_scores += json.loads((stage_dir(campaign, previous) / "scores.json").read_text())["scores"]
    ranking = rank_scores(all_scores)
    if not ranking:
        write_json(campaign / "status.json", {"status": "no_complete_candidate", **report})
        raise RuntimeError("No candidate completed all three seeds")
    write_json(campaign / "ranking.json", ranking)
    expanded = neighbors([r["hp"] for r in ranking[:BEAM_WIDTH]], seen)[:max(0, MAX_CANDIDATES - len(seen))]
    if round_number + 1 < MAX_ROUNDS and expanded:
        next_stage = "round_%02d" % (round_number + 1)
        plan_stage(campaign, next_stage, expanded)
    else:
        next_stage = "final"
        write_json(campaign / "winner.json", {"selection": ranking[0], "selected_before_test": True,
                    "evaluated_candidates": len(ranking), "planned_candidates": len(seen)})
        plan_stage(campaign, next_stage, [ranking[0]["hp"]])
    submit_stage(campaign, next_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "all", "submit", "advance", "run-entry"))
    parser.add_argument("--campaign", type=Path, default=ROOT / DEFAULT_CAMPAIGN)
    parser.add_argument("--stage", default="round_00")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action in ("prepare", "all"):
        prepare(campaign)
        plan_stage(campaign, "round_00", [ANCHOR] + neighbors([ANCHOR], set()))
    if args.action == "all":
        subprocess.run([sys.executable, "-m", "ablation.bciciv2a.smoke", "--campaign", str(campaign)],
                       cwd=campaign / "source", check=True)
    if args.action in ("all", "submit", "advance"):
        import fcntl
        with (campaign / ".control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == "advance":
                advance(campaign, args.stage)
            else:
                submit_stage(campaign, args.stage)
    if args.action == "run-entry":
        run_entry(campaign, args.stage)


if __name__ == "__main__":
    main()
