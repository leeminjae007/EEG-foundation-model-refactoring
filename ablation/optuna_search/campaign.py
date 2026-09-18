"""A CPU controller owns Optuna; GPU arrays only train the fixed five seeds."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import shlex
import shutil
import statistics
import subprocess

import yaml

SEEDS = [42, 696, 1001, 1234, 3407]
NAMES = {"tuev": "TUEV", "tuab": "TUAB", "mentalarithmetic": "Mental Arithmetic",
         "isruc": "ISRUC", "hmc": "HMC"}
LR_KEYS = ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate")
BUDGETS = {"tuev": 16, "tuab": 12, "mentalarithmetic": 1, "isruc": 12, "hmc": 12}
STOP = {"tuev": (12, 8), "tuab": (5, 3), "mentalarithmetic": (5, 5), "isruc": (8, 5), "hmc": (12, 8)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(hp):
    return json.dumps(hp, sort_keys=True)


def space(slug):
    return dict(lr=[5e-6, 1e-5, 1.5e-5, 5e-5] if slug == "tuab" else [5e-5, 1e-4, 1.5e-4, 5e-4],
                wd=[5e-5, .005, .05, .1] if slug == "tuab" else
                   [.025, .05, .1] if slug == "hmc" else [.005, .01, .02, .05, .1],
                dropout=[.1, .2, .3])


def initial_candidates(slug, anchor, leader=None):
    values = [anchor]
    if leader:
        values.append(leader)
    if slug == "tuab":
        values += [dict(anchor, dropout=.3), dict(anchor, wd=.05),
                   dict(anchor, wd=.05, dropout=.3), dict(lr=1.5e-5, wd=.05, dropout=.2),
                   dict(lr=5e-5, wd=.1, dropout=.3)]
    else:
        values += [dict(anchor, lr=5e-5, dropout=.2), dict(anchor, lr=1.5e-4, dropout=.3),
                   dict(lr=5e-4, wd=.05, dropout=.2), dict(anchor, wd=.1, dropout=.3),
                   dict(anchor, lr=1.5e-4, wd=.05, dropout=.1)]
    unique = {key(h): h for h in values}
    return list(unique.values())[:6]


def config_for(base, slug, hp, output):
    cfg = deepcopy(base)
    if any("warm" + "up" in k for k in cfg["optimization"]):
        raise ValueError("Unsupported downstream schedule option")
    for name, value in hp.items():
        if value not in space(slug)[name]:
            raise ValueError("Outside the recorded search space")
    for name in LR_KEYS:
        cfg["optimization"][name] = hp["lr"]
    cfg["optimization"]["weight_decay"] = hp["wd"]
    cfg["optimization"]["label_smoothing"] = 0.0 if slug in ("tuab", "mentalarithmetic") else .1
    cfg["model"]["head_dropout"] = hp["dropout"]
    minimum, patience = STOP[slug]
    cfg["runtime"].update(output=str(output), early_stopping={"min_epochs": minimum, "patience": patience})
    return cfg


def should_stop(epoch, best, policy):
    return (epoch >= policy["min_epochs"] and
            epoch - best["balanced_accuracy"]["epoch"] >= policy["patience"])


def connected_engine(original):
    """Keep training math; select checkpoints and early-stop on validation only."""
    head, marker, tail = original.partition("def run_finetune(")
    if not marker:
        raise ValueError("Missing baseline finetune engine")
    old = '    for epoch in range(start, optimization["epochs"]):\n'
    new = ('    from ablation.optuna_search.campaign import should_stop, write\n'
           '    stop = config["runtime"]["early_stopping"]\n'
           '    end = start if should_stop(start, best, stop) else optimization["epochs"]\n'
           '    for epoch in range(start, end):\n')
    tail = replace_once(tail, old, new)
    tail = replace_once(tail, '    best = {}\n', '    selectors = ["balanced_accuracy"]\n    best = {}\n')
    old = '        if args.smoke:\n            break\n    if args.distributed:\n        dist.barrier()\n'
    new = ('        if args.smoke:\n            break\n'
           '        if should_stop(epoch + 1, best, stop):\n'
           '            if rank == 0:\n'
           '                write(output / "early_stop.json", {"epoch": epoch + 1, "best_epoch": best["balanced_accuracy"]["epoch"], **stop})\n'
           '            break\n'
           '    if should_stop(start, best, stop) and rank == 0:\n'
           '        write(output / "early_stop.json", {"epoch": start, "best_epoch": best["balanced_accuracy"]["epoch"], **stop})\n'
           '    if args.distributed:\n        dist.barrier()\n')
    tail = replace_once(tail, old, new)
    tail = replace_once(tail, '            (output / "result.json").write_text(json.dumps(result, indent=2))',
                        '            write(output / "result.json", result)')
    result = head + marker + tail
    compile(result, "engine.py", "exec")
    return result


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Baseline engine changed; inspect patch anchor: " + repr(old))
    return text.replace(old, new)


def prepare(root, campaign):
    if campaign.exists():
        raise FileExistsError(campaign)
    baseline = root / "outputs/knn37_local_search_20260916_1157"
    old = read(baseline / "manifest.json")
    source = campaign / "source"
    source.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(root / "outputs/knn37_downstream_all13_20260915/source/src", source / "src", ignore=ignore)
    engine = source / "src/training/engine.py"
    original_hash = digest(engine)
    engine.chmod(0o644)
    engine.write_text(connected_engine(engine.read_text()), encoding="utf-8")
    (source / "ablation").mkdir()
    (source / "ablation/__init__.py").write_text("")
    shutil.copytree(Path(__file__).parent, source / "ablation/optuna_search", ignore=ignore)
    datasets = {}
    for slug, name in NAMES.items():
        ds = old["datasets"][slug]
        bases = {}
        for seed in SEEDS:
            entry = next(e for e in ds["baseline"] if e["seed"] == seed)
            cfg = yaml.safe_load(Path(entry["config"]).read_text())
            if cfg["seed"] != seed or str(cfg["model"]["checkpoint"]) != old["checkpoint"]:
                raise ValueError("Baseline seed/checkpoint mismatch")
            if len({cfg["optimization"][k] for k in LR_KEYS}) != 1:
                raise ValueError("Baseline does not have common LR")
            saved = source / "baseline_configs" / slug / (str(seed) + ".yaml")
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(yaml.safe_dump(cfg, sort_keys=False))
            bases[str(seed)] = str(saved)
        lock = baseline / "datasets" / slug / "locked_five_seed/plan.json"
        leader = read(lock)["entries"][0]["hp"] if lock.exists() else None
        datasets[slug] = dict(name=name, bases=bases, space=space(slug), budget=BUDGETS[slug],
                              initial=initial_candidates(slug, ds["anchor"], leader))
    manifest = dict(created_utc=now(), root=str(root), python=str(root / ".venv/bin/python"),
                    controller_python=str(root / ".venv-optuna-search/bin/python"),
                    checkpoint=old["checkpoint"], checkpoint_sha256=digest(old["checkpoint"]),
                    baseline_engine_sha256=original_hash, datasets=datasets, seeds=SEEDS,
                    objective="mean of five seed test balanced_accuracy", test_informed=True,
                    checkpoint_selector="validation balanced_accuracy", sampler="multivariate TPE",
                    startup_trials=6, pruning="none across seeds; per-seed validation patience",
                    time_limit="04:00:00", gpu_partitions="gl40s_dev,gl40s_short,gl40s_long",
                    concurrency_per_dataset=2, cpus=2, memory_gib=20, source_files=[
                        dict(path=p.relative_to(source).as_posix(), sha256=digest(p))
                        for p in sorted(source.rglob("*")) if p.is_file()])
    write(campaign / "manifest.json", manifest)
    print(json.dumps(dict(prepared=str(campaign), seed_runs=sum(BUDGETS.values()) * 5)))


def verify(campaign):
    m = read(campaign / "manifest.json")
    for f in m["source_files"]:
        if digest(campaign / "source" / f["path"]) != f["sha256"]:
            raise ValueError("Campaign source changed: " + f["path"])
    return m


def validate_seed(entry):
    output = Path(entry["output"])
    if digest(entry["config"]) != entry["config_sha256"]:
        raise ValueError("Config changed")
    cfg = yaml.safe_load(Path(entry["config"]).read_text())
    if yaml.safe_load((output / "resolved_config.yaml").read_text()) != cfg:
        raise ValueError("Executed config differs")
    result = read(output / "result.json")["balanced_accuracy"]
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines() if line.strip()]
    if [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Noncontiguous validation history")
    selected = result["selection"]
    if not history or selected["score"] != max(r["balanced_accuracy"] for r in history):
        raise ValueError("Invalid validation checkpoint selection")
    if history[selected["epoch"] - 1]["balanced_accuracy"] != selected["score"]:
        raise ValueError("Wrong selected epoch")
    if len(history) < cfg["optimization"]["epochs"]:
        stop = read(output / "early_stop.json")
        if stop["epoch"] != len(history) or not should_stop(len(history), {"balanced_accuracy": selected}, cfg["runtime"]["early_stopping"]):
            raise ValueError("Invalid early stopping")
    if len(history) < cfg["runtime"]["early_stopping"]["min_epochs"]:
        raise ValueError("Fewer than the required observation epochs")
    required = ("balanced_accuracy", "auroc", "auprc") if cfg["data"]["dataset"] in ("tuab", "stress") else ("balanced_accuracy", "weighted_f1", "kappa")
    metrics = {name: float(result["test"][name]) for name in required}
    if not all(math.isfinite(x) for x in metrics.values()) or not (output / "best-balanced_accuracy.pth").is_file():
        raise ValueError("Missing checkpoint or finite metrics")
    return dict(seed=entry["seed"], test=metrics, selection={**selected, "split": "validation"}, epochs=len(history),
                source_result=str(output / "result.json"))


def five_seed_summary(rows):
    if len(rows) != 5 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("A trial needs exactly five distinct fixed seeds")
    metrics = set(rows[0]["test"])
    if any(set(r["test"]) != metrics for r in rows):
        raise ValueError("Inconsistent metric coverage")
    result = {}
    for metric in sorted(metrics):
        values = [float(r["test"][metric]) for r in rows]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite objective")
        result[metric] = dict(mean=statistics.fmean(values), population_sd=statistics.pstdev(values), n=5)
    return result


def submit_once(path, command):
    if path.exists():
        return read(path)["job_id"]
    intent = path.with_suffix(".intent.json")
    if intent.exists():
        raise RuntimeError("Ambiguous submission: reconcile " + str(intent))
    write(intent, dict(command=command, utc=now()))
    p = subprocess.run(command, text=True, capture_output=True, timeout=45)
    write(path.with_suffix(".response.json"), dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
    job = p.stdout.strip().split(";")[0]
    if p.returncode or not job.isdigit():
        raise RuntimeError("Submission failed: " + p.stderr)
    write(path, dict(job_id=job, utc=now(), command=command))
    return job


def dataset_control(campaign, slug):
    path = campaign / "dataset_controls.json"
    if not path.exists():
        return None
    controls = read(path)
    if controls.get("version") != 1:
        raise ValueError("Unsupported dataset control version")
    control = controls.get("datasets", {}).get(slug)
    if control is not None and control.get("action") != "fix_existing_trial":
        raise ValueError("Unsupported dataset control action: " + slug)
    return control


def fixed_dataset_status(campaign, slug, control):
    number = control["trial"]
    if type(number) is not int or number < 0:
        raise ValueError("Invalid fixed trial number")
    directory = campaign / "datasets" / slug / ("trial-%03d" % number)
    plan = read(directory / "plan.json")
    if plan["number"] != number or plan["slug"] != slug or plan["hp"] != control["hp"]:
        raise ValueError("Fixed trial identity or parameters changed")
    rows = [validate_seed(entry) for entry in plan["entries"]]
    summary = read(directory / "summary.json")
    if rows != summary["rows"] or five_seed_summary(rows) != summary["metrics"]:
        raise ValueError("Fixed trial artifacts changed")
    return dict(status="fixed", trial=number, hp=plan["hp"], completed_seeds=5,
                required_seeds=5, submissions_disabled=True, retries_disabled=True,
                fixed_by="user", reason=control["reason"],
                reporting_note=control["reporting_note"], independent_test_evaluation=False,
                historical_summary=str(directory / "summary.json"))


def campaign_status(datasets):
    states = {entry["status"] for entry in datasets.values()}
    if states == {"complete"}:
        return "complete"
    if states and states <= {"complete", "fixed"}:
        return "finished_with_fixed_settings"
    return "running"


def submit_trial(campaign, m, slug, plan, retry_indices=None):
    if dataset_control(campaign, slug) is not None:
        raise RuntimeError("User disabled new trials and retries for " + slug)
    directory = Path(plan["directory"])
    indices = list(range(5)) if retry_indices is None else retry_indices
    path = directory / ("submission.json" if retry_indices is None else "retry.json")
    cmd = [m["python"], "-m", "ablation.optuna_search.worker", "--campaign", str(campaign), "--plan", str(directory / "plan.json")]
    env = "unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; "
    wrapper = env + 'export TMPDIR="/tmp/knnopt-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"; exec ' + shlex.join(cmd)
    return submit_once(path, ["sbatch", "--parsable", "--account=system", "--nodes=1", "--ntasks=1",
        "--chdir=" + str(campaign / "source"), "--partition=" + m["gpu_partitions"],
        "--gpus-per-task=l40s:1", "--cpus-per-task=" + str(m["cpus"]), "--mem=" + str(m.get("memory_gib", 20)) + "G", "--time=" + m["time_limit"],
        "--array=" + ",".join(map(str, indices)) + "%" + str(m["concurrency_per_dataset"]),
        "--job-name=knnopt-" + slug + "-t" + str(plan["number"]),
        "--output=" + str(directory / "slurm-%A_%a.log"), "--wrap", wrapper])


def job_terminal(job):
    queued = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, timeout=30)
    if queued.returncode or queued.stdout.strip():
        return False
    accounting = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
                                capture_output=True, text=True, timeout=30)
    rows = [line.split("|") for line in accounting.stdout.splitlines() if line.strip()]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    return bool(rows) and not accounting.returncode and all(r[1].split()[0].rstrip("+") in terminal for r in rows)


def plan_trial(campaign, slug, ds, number, hp):
    directory = campaign / "datasets" / slug / ("trial-%03d" % number)
    if (directory / "plan.json").exists():
        return read(directory / "plan.json")
    entries = []
    for seed in SEEDS:
        base = yaml.safe_load(Path(ds["bases"][str(seed)]).read_text())
        output = directory / ("seed" + str(seed))
        config = directory / "configs" / (str(seed) + ".yaml")
        cfg = config_for(base, slug, hp, output)
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        entries.append(dict(seed=seed, output=str(output), config=str(config), config_sha256=digest(config)))
    plan = dict(number=number, hp=hp, directory=str(directory), slug=slug, entries=entries)
    write(directory / "plan.json", plan)
    return plan


def study_for(campaign, slug, ds):
    import optuna
    directory = campaign / "datasets" / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sampler.pkl"
    sampler = pickle.loads(path.read_bytes()) if path.exists() else optuna.samplers.TPESampler(
        seed=20260916 + list(NAMES).index(slug), multivariate=True, n_startup_trials=6)
    study = optuna.create_study(study_name=slug, direction="maximize", sampler=sampler,
        storage="sqlite:///" + str(directory / "study.sqlite3"), load_if_exists=True)
    if not study.trials:
        for hp in ds["initial"]:
            study.enqueue_trial(hp, skip_if_exists=True)
    return study, path


def advance_dataset(campaign, m, slug):
    control = dataset_control(campaign, slug)
    if control is not None:
        return fixed_dataset_status(campaign, slug, control)
    import optuna
    ds = m["datasets"][slug]
    study, sampler_path = study_for(campaign, slug, ds)
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1:
        raise ValueError("Only one five-seed trial may run per dataset")
    for trial in running:
        # Reconstruct deterministic categorical suggestions if the controller died during ask.
        live = optuna.trial.Trial(study, trial._trial_id)
        hp = {name: live.suggest_categorical(name, values) for name, values in ds["space"].items()}
        plan = plan_trial(campaign, slug, ds, trial.number, hp)
        directory = Path(plan["directory"])
        rows, missing, errors = [], [], []
        for index, entry in enumerate(plan["entries"]):
            try:
                rows.append(validate_seed(entry))
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                missing.append(index)
                errors.append(dict(seed=entry["seed"], error=str(exc)))
        if not missing:
            summary = five_seed_summary(rows)
            write(directory / "summary.json", dict(hp=hp, rows=rows, metrics=summary, completed_utc=now(), test_informed=True))
            live.set_user_attr("summary", summary)
            study.tell(live, summary["balanced_accuracy"]["mean"])
        else:
            if not (directory / "submission.json").exists():
                job = submit_trial(campaign, m, slug, plan)
            else:
                submission = directory / ("retry.json" if (directory / "retry.json").exists() else "submission.json")
                job = read(submission)["job_id"]
                if job_terminal(job):
                    if submission.name == "retry.json":
                        write(directory / "failure.json", dict(errors=errors, utc=now()))
                        return dict(status="needs_attention", trial=trial.number, completed_seeds=len(rows), errors=errors)
                    job = submit_trial(campaign, m, slug, plan, missing)
            return dict(status="running", trial=trial.number, hp=hp, job_id=job,
                        completed_seeds=len(rows), required_seeds=5, missing=errors)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= ds["budget"]:
        best = study.best_trial
        directory = campaign / "datasets" / slug / ("trial-%03d" % best.number)
        summary = read(directory / "summary.json")
        rows = [validate_seed(e) for e in read(directory / "plan.json")["entries"]]
        if rows != summary["rows"] or five_seed_summary(rows) != summary["metrics"]:
            raise ValueError("Winning seed artifacts changed since trial completion")
        winner = dict(trial=best.number, **summary)
        write(campaign / "datasets" / slug / "winner.json", winner)
        return dict(status="complete", completed_trials=len(completed), budget=ds["budget"], winner=winner)
    used = {key(t.params) for t in completed}
    for attempt in range(50):
        if attempt >= 8:
            all_hp = [dict(zip(ds["space"], values)) for values in itertools.product(*ds["space"].values())]
            unseen = next(h for h in all_hp if key(h) not in used)
            study.enqueue_trial(unseen, skip_if_exists=True)
        trial = study.ask()
        hp = {name: trial.suggest_categorical(name, values) for name, values in ds["space"].items()}
        temporary = sampler_path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(study.sampler))
        temporary.replace(sampler_path)
        if key(hp) in used:
            trial.set_user_attr("reason", "duplicate combination; no GPU run")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        plan = plan_trial(campaign, slug, ds, trial.number, hp)
        job = submit_trial(campaign, m, slug, plan)
        return dict(status="running", trial=trial.number, hp=hp, job_id=job,
                    completed_seeds=0, required_seeds=5, completed_trials=len(completed), budget=ds["budget"])
    raise RuntimeError("Could not generate a distinct Optuna candidate")


def step(campaign):
    import fcntl
    import optuna
    with (campaign / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = verify(campaign)
        smoke = campaign / "smoke_validation.json"
        if not smoke.exists():
            submission = campaign / "preflight_submission.json"
            job = read(submission)["job_id"] if submission.exists() else None
            failed = bool(job and job_terminal(job))
            log = campaign / ("preflight-" + str(job) + ".log")
            return dict(status="preflight_failed" if failed else "preflight_pending", checked_utc=now(),
                        preflight_job=job, log_tail=log.read_text(errors="replace")[-3000:] if log.exists() else "", datasets={})
        report = read(smoke)
        if not report["passed"] or report["manifest_sha256"] != digest(campaign / "manifest.json"):
            raise ValueError("CPU preflight does not match campaign")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        datasets = {}
        for slug in NAMES:
            try:
                datasets[slug] = advance_dataset(campaign, m, slug)
            except Exception as exc:
                datasets[slug] = dict(status="needs_attention", error=repr(exc))
        status = dict(checked_utc=now(), status=campaign_status(datasets),
                      test_informed=True, objective=m["objective"], datasets=datasets)
        write(campaign / "status.json", status)
        return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "step"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.campaign)
    else:
        print(json.dumps(step(args.campaign), ensure_ascii=False))


if __name__ == "__main__":
    main()
