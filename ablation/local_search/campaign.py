"""One-seed screening, bounded combinations, then conditional seed confirmation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
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

SEEDS = [42, 1234, 696, 1001, 3407]
NAMES = {"hmc": "HMC", "isruc": "ISRUC", "mentalarithmetic": "MentalArithmetic",
         "tuev": "TUEV", "tuab": "TUAB"}
START_SEEDS = {"hmc": 1001, "isruc": 696, "mentalarithmetic": 3407, "tuev": 1001, "tuab": 696}
LR_KEYS = ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate")
EPS = 1e-12


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cid(hp):
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:12]


def anchor(config):
    opt = config["optimization"]
    if len({opt[k] for k in LR_KEYS}) != 1:
        raise ValueError("The baseline must use one common learning rate")
    return dict(lr=opt[LR_KEYS[0]], wd=opt["weight_decay"], dropout=config["model"]["head_dropout"])


def space(slug):
    return {
        "lr": [5e-6, 1e-5, 1.5e-5] if slug == "tuab" else [5e-5, 1e-4, 1.5e-4, 5e-4],
        "wd": ([2.5e-5, 5e-5, 1e-4] if slug == "tuab" else
               [0.025, 0.05, 0.1] if slug == "hmc" else [0.005, 0.01, 0.02]),
        "dropout": [0.1, 0.2, 0.3],
    }


def first_candidates(slug, base):
    return [dict(base, **{key: value}) for key, values in space(slug).items()
            for value in values if value != base[key]]


def config_for(base, hp, output):
    config = deepcopy(base)
    for k in config["optimization"]:
        if "warm" + "up" in k:
            raise ValueError("Unexpected downstream schedule option: " + k)
    for key in LR_KEYS:
        config["optimization"][key] = hp["lr"]
    config["optimization"]["weight_decay"] = hp["wd"]
    config["model"]["head_dropout"] = hp["dropout"]
    dataset = config["data"]["dataset"]
    minimum, patience = ((12, 8) if dataset in ("hmc", "tuev") else
                         (4, 3) if dataset == "tuab" else (6, 5))
    config["runtime"].update(output=str(output), evaluate_test=False,
                            early_stopping={"min_epochs": minimum, "patience": patience})
    return config


def connected_engine(original):
    changes = {
        '    for split in ("train", "val", "test"):\n':
        '    for split in (("train", "val", "test") if config["runtime"].get("evaluate_test", True) else ("train", "val")):\n',
        '            test = evaluate(model, loaders["test"], spec.task, data["dataset"], device, world)\n':
        '            test = (evaluate(model, loaders["test"], spec.task, data["dataset"], device, world)\n'
        '                    if config["runtime"].get("evaluate_test", True) else None)\n',
    }
    for old, new in changes.items():
        if original.count(old) != 1:
            raise ValueError("Inspect the changed baseline engine before applying the test gate")
        original = original.replace(old, new)
    old = '        if args.smoke:\n            break\n    if args.distributed:\n        dist.barrier()\n    if not args.smoke:\n        result = {}\n'
    new = '''        if args.smoke:
            break
        stop = config["runtime"].get("early_stopping")
        if stop and epoch + 1 >= stop["min_epochs"] and epoch + 1 - best["balanced_accuracy"]["epoch"] >= stop["patience"]:
            if rank == 0:
                (output / "early_stop.json").write_text(json.dumps({"epoch": epoch + 1,
                    "best_epoch": best["balanced_accuracy"]["epoch"], "reason": "validation_patience", **stop}))
            break
    if args.distributed:
        dist.barrier()
    if not args.smoke:
        result = {}
'''
    if original.count(old) != 1:
        raise ValueError("Inspect the engine before connecting campaign-only early stopping")
    original = original.replace(old, new)
    compile(original, "engine.py", "exec")
    return original


def check_result(entry, baseline=False):
    output = Path(entry["output"])
    cfg = yaml.safe_load(Path(entry["config"]).read_text())
    if digest(entry["config"]) != entry["config_sha256"]:
        raise ValueError("Config changed")
    if not baseline and yaml.safe_load((output / "resolved_config.yaml").read_text()) != cfg:
        raise ValueError("Executed config differs")
    result = read(output / "result.json")
    selection = result["balanced_accuracy"]["selection"]
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines() if line.strip()]
    epochs = cfg["optimization"]["epochs"]
    # Recovered runs can contain the last epoch twice after interruption.
    by_epoch = {}
    for row in history:
        epoch = row["epoch"]
        if epoch in by_epoch and by_epoch[epoch] != row:
            raise ValueError("Conflicting duplicate validation epoch")
        by_epoch[epoch] = row
    finished_epoch = max(by_epoch) if by_epoch else 0
    if sorted(by_epoch) != list(range(1, finished_epoch + 1)) or not 1 <= finished_epoch <= epochs:
        raise ValueError("Missing validation epochs")
    if finished_epoch < epochs:
        stop = read(output / "early_stop.json")
        policy = cfg["runtime"].get("early_stopping")
        if baseline or not policy or stop["epoch"] != finished_epoch or finished_epoch < policy["min_epochs"]:
            raise ValueError("Incomplete run without authorized early stopping")
        if finished_epoch - selection["epoch"] < policy["patience"]:
            raise ValueError("Early stopping patience was not reached")
    score = float(selection["score"])
    if not math.isfinite(score) or abs(score - max(r["balanced_accuracy"] for r in history)) > EPS:
        raise ValueError("Validation selector mismatch")
    if abs(by_epoch[selection["epoch"]]["balanced_accuracy"] - score) > EPS:
        raise ValueError("Selected epoch does not match validation")
    if not baseline and any(v.get("test") is not None for v in result.values()):
        raise ValueError("Search must not evaluate test data")
    if not (output / "best-balanced_accuracy.pth").is_file():
        raise ValueError("Missing selected checkpoint")
    return dict(seed=entry["seed"], score=score, epoch=selection["epoch"], output=str(output),
                config=entry["config"], hp=entry["hp"], candidate=cid(entry["hp"]))


def prepare(root, campaign):
    if (campaign / "manifest.json").exists():
        verify(campaign)
        return
    baseline = root / "outputs/knn37_downstream_all13_20260915"
    old = read(baseline / "manifest.json")
    baseline_entries = old.get("submitted_entries", []) + old.get("reused_entries", [])
    datasets = {}
    checkpoint = None
    for slug in NAMES:
        bases = []
        for seed in SEEDS:
            matches = [e for e in baseline_entries if e["slug"] == slug and int(e["seed"]) == seed]
            if len(matches) != 1:
                raise ValueError("Expected one baseline for each seed")
            e = matches[0]
            cfg = yaml.safe_load(Path(e["config"]).read_text())
            if cfg["seed"] != seed:
                raise ValueError("Seed mismatch")
            hp = anchor(cfg)
            if any(hp[k] not in space(slug)[k] for k in hp):
                raise ValueError("Baseline not inside requested search space")
            if slug not in ("tuab", "mentalarithmetic") and cfg["optimization"]["label_smoothing"] != 0.1:
                raise ValueError("Multiclass smoothing must be 0.1")
            config_for(cfg, hp, campaign / "unused")
            e = dict(e, output=e["result_dir"], hp=hp)
            value = check_result(e, baseline=True)
            e["validation"] = value["score"]
            e["best_epoch"] = value["epoch"]
            bases.append(e)
            cp = str(Path(cfg["model"]["checkpoint"]).resolve())
            if checkpoint is not None and cp != checkpoint:
                raise ValueError("Mixed pretrained checkpoints")
            checkpoint = cp
        start = min(bases, key=lambda e: (e["validation"], e["seed"]))["seed"]
        if start != START_SEEDS[slug]:
            raise ValueError("Baseline changed since the agreed seed selection")
        datasets[slug] = dict(name=NAMES[slug], start_seed=start, baseline=bases,
                              anchor=bases[0]["hp"], space=space(slug),
                              confirmation_seeds=[start] + [s for s in SEEDS if s != start][:2])
    source = campaign / "source"
    source.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(baseline / "source/src", source / "src", ignore=ignore)
    engine = source / "src/training/engine.py"
    original_sha = digest(engine)
    engine.chmod(0o644)
    engine.write_text(connected_engine(engine.read_text()), encoding="utf-8")
    (source / "ablation").mkdir()
    (source / "ablation/__init__.py").write_text("")
    shutil.copytree(Path(__file__).parent, source / "ablation/local_search", ignore=ignore)
    for slug, dataset in datasets.items():
        for e in dataset["baseline"]:
            saved = source / "baseline_configs" / slug / (str(e["seed"]) + ".yaml")
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(e["config"], saved)
            e["original_config"] = e["config"]
            e["config"] = str(saved)
    files = [{"path": p.relative_to(source).as_posix(), "sha256": digest(p)}
             for p in sorted(source.rglob("*")) if p.is_file()]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                    python=str(root / ".venv/bin/python"), checkpoint=checkpoint,
                    checkpoint_sha256=digest(checkpoint), source_files=files,
                    baseline_engine_sha256=original_sha, datasets=datasets,
                    selector="best all-epoch validation balanced_accuracy",
                    policy="Common LR; base LR from first update; fixed batches, heads and epoch horizons",
                    stages="single-axis trials; at most four improving combinations; top two on three seeds; lock winner then remaining two seeds",
                    max_combinations_per_dataset=4, gpu_concurrency_per_dataset=2,
                    time_limit="04:00:00", gpu_partitions="gl40s_dev,gl40s_short,gl40s_long",
                    confirmation_requires="positive mean delta on all three AND the two new seeds",
                    baseline_manifest=str(baseline / "manifest.json"))
    write_json(campaign / "manifest.json", manifest)
    for slug, d in datasets.items():
        plan_stage(campaign, slug, "screen", first_candidates(slug, d["anchor"]), [d["start_seed"]])
    write_json(campaign / "status.json", {"status": "prepared", "initial_runs": 34})


def verify(campaign):
    m = read(campaign / "manifest.json")
    for f in m["source_files"]:
        if digest(campaign / "source" / f["path"]) != f["sha256"]:
            raise ValueError("Snapshot changed: " + f["path"])
    return m


def stage_path(campaign, slug, stage):
    if slug not in NAMES or stage not in ("screen", "combine", "confirm", "remaining", "evaluate"):
        raise ValueError("Unknown dataset/stage")
    return campaign / "datasets" / slug / stage


def plan_stage(campaign, slug, stage, candidates, seeds, origins=None):
    d = stage_path(campaign, slug, stage)
    if (d / "plan.json").exists():
        old = read(d / "plan.json")
        if old["candidates"] != candidates or old["seeds"] != seeds:
            raise ValueError("Stage plan changed")
        return old
    manifest = read(campaign / "manifest.json")
    entries = []
    for hp in candidates:
        if any(hp[k] not in space(slug)[k] for k in hp):
            raise ValueError("Outside authorized space")
        for seed in seeds:
            base = next(e for e in manifest["datasets"][slug]["baseline"] if e["seed"] == seed)
            cfg = yaml.safe_load(Path(base["config"]).read_text())
            output = d / "runs" / (cid(hp) + "_seed" + str(seed))
            path = d / "configs" / (output.name + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config_for(cfg, hp, output), sort_keys=False))
            e = dict(index=len(entries), slug=slug, seed=seed, hp=hp, candidate=cid(hp),
                     config=str(path), config_sha256=digest(path), output=str(output))
            if origins:
                e["origin"] = next(r for r in origins if r["seed"] == seed)
            entries.append(e)
    plan = dict(stage=stage, slug=slug, candidates=candidates, seeds=seeds, entries=entries,
                manifest_sha256=digest(campaign / "manifest.json"))
    write_json(d / "plan.json", plan)
    return plan


def load_plan(campaign, slug, stage):
    p = read(stage_path(campaign, slug, stage) / "plan.json")
    if p["manifest_sha256"] != digest(campaign / "manifest.json"):
        raise ValueError("Manifest changed")
    m = read(campaign / "manifest.json")
    for e in p["entries"]:
        cfg = yaml.safe_load(Path(e["config"]).read_text())
        base = next(b for b in m["datasets"][slug]["baseline"] if b["seed"] == e["seed"])
        expected = config_for(yaml.safe_load(Path(base["config"]).read_text()), e["hp"], e["output"])
        if digest(e["config"]) != e["config_sha256"] or cfg != expected:
            raise ValueError("Candidate config or fixed recipe changed")
    return p


def submit_once(path, command):
    if path.exists():
        return read(path)["job_id"]
    intent = path.with_name(path.stem + "_intent.json")
    if intent.exists():
        raise RuntimeError("Ambiguous prior submission; reconcile " + str(intent))
    write_json(intent, dict(command=command, utc=datetime.now(timezone.utc).isoformat()))
    r = subprocess.run(command, capture_output=True, text=True)
    write_json(path.with_name(path.stem + "_response.json"), dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    job = r.stdout.strip().split(";")[0]
    if r.returncode or not job.isdigit():
        raise RuntimeError("Submission failed/ambiguous: " + r.stderr)
    write_json(path, dict(job_id=job, command=command, utc=datetime.now(timezone.utc).isoformat()))
    return job


def submit_stage(campaign, slug, stage, retry=False, indices=None):
    m = verify(campaign)
    smoke = read(campaign / "smoke_validation.json")
    if not smoke["passed"] or smoke["manifest_sha256"] != digest(campaign / "manifest.json"):
        raise ValueError("Snapshot smoke validation required")
    plan = load_plan(campaign, slug, stage)
    d = stage_path(campaign, slug, stage)
    label = "retry" if retry else "submission"
    array = (",".join(map(str, indices)) if indices is not None else "0-" + str(len(plan["entries"]) - 1)) + "%2"
    cmd = [m["python"], "-m", "ablation.local_search.campaign", "run", "--campaign", str(campaign), "--slug", slug, "--stage", stage]
    env = "unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; "
    common = ["sbatch", "--parsable", "--account=system", "--nodes=1", "--ntasks=1", "--chdir=" + str(campaign / "source")]
    wrapper = env + 'export TMPDIR="/tmp/knn-hp-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"; exec ' + shlex.join(cmd)
    job = submit_once(d / (label + ".json"), common + ["--job-name=knnhp-" + slug + "-" + stage,
        "--partition=" + m.get("gpu_partitions", "gl40s_short,gl40s_long"), "--array=" + array, "--gpus-per-task=l40s:1",
        "--cpus-per-task=8", "--mem=64G", "--time=" + m["time_limit"],
        "--output=" + str(d / "slurm-%A_%a.log"), "--wrap", wrapper])
    follow = [m["python"], "-m", "ablation.local_search.campaign", "advance", "--campaign", str(campaign), "--slug", slug, "--stage", stage]
    control = submit_once(d / (label + "_controller.json"), common + ["--job-name=knnhp-next-" + slug,
        "--partition=cpu_short,cpu_long", "--cpus-per-task=1", "--mem=4G", "--time=00:15:00",
        "--dependency=afterany:" + job, "--kill-on-invalid-dep=yes",
        "--output=" + str(d / "controller-%j.log"), "--wrap", env + "exec " + shlex.join(follow)])
    status = dict(status="submitted", stage=stage, gpu_job_id=job, cpu_controller_job_id=control,
                  runs=len(indices) if indices is not None else len(plan["entries"]), retry=retry)
    write_json(campaign / "datasets" / slug / "status.json", status)
    return status


def rows_for(campaign, slug, stages):
    rows = []
    for stage in stages:
        if not (stage_path(campaign, slug, stage) / "plan.json").exists():
            continue
        rows += [check_result(e) for e in load_plan(campaign, slug, stage)["entries"]
                 if not (Path(e["output"]) / "pruned.json").exists()]
    return rows


def combinations(dataset, screen_rows):
    base = dataset["anchor"]
    score = next(e["validation"] for e in dataset["baseline"] if e["seed"] == dataset["start_seed"])
    axes = {}
    for key in base:
        improved = [r for r in screen_rows if r["score"] > score + EPS and r["hp"][key] != base[key]]
        if improved:
            axes[key] = max(improved, key=lambda r: (r["score"], r["candidate"]))["hp"][key]
    candidates = []
    for size in (2, 3):
        for keys in itertools.combinations(axes, size):
            candidates.append(dict(base, **{k: axes[k] for k in keys}))
    return candidates[:4]


def confirmed_winner(dataset, rows):
    wanted = dataset["confirmation_seeds"]
    base_scores = {e["seed"]: e["validation"] for e in dataset["baseline"]}
    ranking = []
    for candidate in sorted({r["candidate"] for r in rows}):
        group = [r for r in rows if r["candidate"] == candidate and r["seed"] in wanted]
        if len(group) != 3 or {r["seed"] for r in group} != set(wanted):
            continue
        deltas = [r["score"] - base_scores[r["seed"]] for r in group]
        new = [r["score"] - base_scores[r["seed"]] for r in group if r["seed"] != dataset["start_seed"]]
        if statistics.mean(deltas) > EPS and statistics.mean(new) > EPS:
            ranking.append(dict(candidate=candidate, hp=group[0]["hp"], rows=group,
                                delta=statistics.mean(deltas), new_seed_delta=statistics.mean(new),
                                sd=statistics.pstdev(r["score"] for r in group)))
    return sorted(ranking, key=lambda r: (-r["delta"], r["sd"], r["candidate"]))


def finish_baseline(campaign, slug, reason):
    m = read(campaign / "manifest.json")
    d = m["datasets"][slug]
    entries = [dict(e, display_name=d["name"]) for e in d["baseline"]]
    write_json(campaign / "datasets" / slug / "final.json", dict(entries=entries, selected="baseline", reason=reason, hp=d["anchor"]))
    write_json(campaign / "datasets" / slug / "status.json", dict(status="final_artifacts_ready", selected="baseline", reason=reason))


def advance(campaign, slug, stage):
    m = verify(campaign)
    dataset = m["datasets"][slug]
    plan = load_plan(campaign, slug, stage)
    failures = []
    for e in plan["entries"]:
        pruned = Path(e["output"]) / "pruned.json"
        if stage in ("screen", "combine", "confirm") and pruned.exists():
            if read(pruned).get("reason") == "nonfinite_gradient":
                continue
        try:
            if stage == "evaluate":
                result = read(Path(e["output"]) / "result.json")
                if not math.isfinite(result["balanced_accuracy"]["test"]["balanced_accuracy"]):
                    raise ValueError("Invalid final metric")
            else:
                check_result(e)
        except (OSError, ValueError, KeyError, TypeError) as ex:
            failures.append(dict(index=e["index"], seed=e["seed"], candidate=e["candidate"], error=str(ex)))
    if failures:
        d = stage_path(campaign, slug, stage)
        write_json(d / "missing_artifacts.json", failures)
        if not (d / "retry.json").exists() and all("No such file" in e["error"] for e in failures):
            return submit_stage(campaign, slug, stage, retry=True, indices=[e["index"] for e in failures])
        write_json(campaign / "datasets" / slug / "status.json", dict(status="incomplete_artifacts", stage=stage, missing=failures))
        return
    if stage == "evaluate":
        winner = read(campaign / "datasets" / slug / "winner.json")
        entries = [dict(e, display_name=dataset["name"]) for e in plan["entries"]]
        write_json(campaign / "datasets" / slug / "final.json", dict(entries=entries, selected="tuned", hp=winner["hp"]))
        write_json(campaign / "datasets" / slug / "status.json", dict(status="final_artifacts_ready", selected="tuned"))
        return
    stage_rows = [check_result(e) for e in plan["entries"] if not (Path(e["output"]) / "pruned.json").exists()]
    write_json(stage_path(campaign, slug, stage) / "scores.json", stage_rows)
    if stage == "screen":
        candidates = combinations(dataset, stage_rows)
        if candidates:
            plan_stage(campaign, slug, "combine", candidates, [dataset["start_seed"]])
            return submit_stage(campaign, slug, "combine")
    if stage in ("screen", "combine"):
        rows = rows_for(campaign, slug, ["screen", "combine"])
        baseline = next(e["validation"] for e in dataset["baseline"] if e["seed"] == dataset["start_seed"])
        better = sorted([r for r in rows if r["score"] > baseline + EPS], key=lambda r: (-r["score"], r["candidate"]))[:2]
        if not better:
            return finish_baseline(campaign, slug, "No single-seed validation improvement")
        plan_stage(campaign, slug, "confirm", [r["hp"] for r in better], dataset["confirmation_seeds"][1:])
        return submit_stage(campaign, slug, "confirm")
    if stage == "confirm":
        ranked = confirmed_winner(dataset, rows_for(campaign, slug, ["screen", "combine", "confirm"]))
        write_json(campaign / "datasets" / slug / "confirmation.json", ranked)
        if not ranked:
            return finish_baseline(campaign, slug, "No improvement reproduced on the additional seeds")
        winner = dict(ranked[0], locked_before_test=True, selection_seeds=dataset["confirmation_seeds"])
        write_json(campaign / "datasets" / slug / "winner.json", winner)
        remaining = [s for s in SEEDS if s not in dataset["confirmation_seeds"]]
        plan_stage(campaign, slug, "remaining", [winner["hp"]], remaining)
        return submit_stage(campaign, slug, "remaining")
    if stage == "remaining":
        winner = read(campaign / "datasets" / slug / "winner.json")
        rows = [r for r in rows_for(campaign, slug, ["screen", "combine", "confirm", "remaining"]) if r["candidate"] == winner["candidate"]]
        if len(rows) != 5 or {r["seed"] for r in rows} != set(SEEDS):
            raise ValueError("Need exactly five complete seeds before final evaluation")
        plan_stage(campaign, slug, "evaluate", [winner["hp"]], SEEDS, origins=rows)
        return submit_stage(campaign, slug, "evaluate")


def run_entry(campaign, slug, stage):
    m = verify(campaign)
    e = load_plan(campaign, slug, stage)["entries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = Path(e["output"])
    if (output / "result.json").exists():
        return
    args = [m["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
            "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
            "--rdzv_id=knnhp-%s-%s" % (os.environ["SLURM_JOB_ID"], e["index"]),
            "--module", "ablation.local_search.train", "--campaign", str(campaign), "--slug", slug,
            "--stage", stage, "--index", str(e["index"])]
    subprocess.run(args, cwd=campaign / "source", check=True)


def inspect(campaign):
    m = read(campaign / "manifest.json")
    report = dict(campaign=str(campaign), created_utc=m["created_utc"], checked_utc=datetime.now(timezone.utc).isoformat(), datasets={})
    for slug, ds in m["datasets"].items():
        directory = campaign / "datasets" / slug
        status = read(directory / "status.json") if (directory / "status.json").exists() else {"status": "prepared"}
        trials = []
        for stage in ("screen", "combine", "confirm", "remaining", "evaluate"):
            p = stage_path(campaign, slug, stage) / "plan.json"
            if not p.exists():
                continue
            for e in read(p)["entries"]:
                row = {k: e[k] for k in ("seed", "hp", "candidate", "output")}
                row["stage"] = stage
                try:
                    r = read(Path(e["output"]) / "result.json")
                    row.update(status="finished", validation=r["balanced_accuracy"]["selection"]["score"])
                except (OSError, ValueError, KeyError):
                    row["status"] = "pending_or_running"
                    pruned = Path(e["output"]) / "pruned.json"
                    if pruned.exists():
                        row.update(status="pruned", termination=read(pruned))
                    v = Path(e["output"]) / "validation.jsonl"
                    if v.exists():
                        history = []
                        for line in v.read_text().splitlines():
                            try:
                                history.append(json.loads(line))
                            except ValueError:
                                pass
                        if history:
                            row.update(epoch=history[-1]["epoch"], validation=max(r["balanced_accuracy"] for r in history))
                trials.append(row)
        report["datasets"][slug] = dict(name=ds["name"], start_seed=ds["start_seed"], status=status, trials=trials)
        if (directory / "final.json").exists():
            report["datasets"][slug]["final"] = read(directory / "final.json")
    report["ready_to_publish"] = all("final" in d for d in report["datasets"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "advance", "run", "inspect"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slug", choices=list(NAMES))
    parser.add_argument("--stage", default="screen")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(args.root.resolve(), campaign)
    elif args.action == "inspect":
        print(json.dumps(inspect(campaign)), flush=True)
    elif args.action == "run":
        run_entry(campaign, args.slug, args.stage)
    else:
        import fcntl
        with (campaign / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == "submit":
                print(json.dumps({s: submit_stage(campaign, s, args.stage) for s in ([args.slug] if args.slug else NAMES)}), flush=True)
            else:
                advance(campaign, args.slug, args.stage)


if __name__ == "__main__":
    main()
