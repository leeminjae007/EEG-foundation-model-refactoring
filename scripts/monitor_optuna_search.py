"""Local CPU controller transport and publication for test-informed Optuna."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from ablation.optuna_search.campaign import NAMES, five_seed_summary
from scripts.monitor_experiment_results import atomic_text, dense_ranks, display, update_global

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "Checkpoints and early stopping use validation BAcc; Optuna compares five-seed test BAcc, so this test set is not an independent final evaluation."


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def fetch(args):
    python = args.root + "/.venv-optuna-search/bin/python"
    if getattr(args, "read_only", False):
        code = "import json; from pathlib import Path; print(json.dumps(json.loads(Path(" + repr(args.campaign + "/status.json") + ").read_text())))"
        command = shlex.join([python, "-c", code])
    else:
        command = "cd " + shlex.quote(args.campaign + "/source") + " && " + shlex.join([
            python, "-m", "ablation.optuna_search.campaign", "step", "--campaign", args.campaign])
    errors = []
    for hostname in [None, *args.fallback_hosts]:
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if hostname:
            ssh += ["-o", "HostName=" + hostname, "-o", "HostKeyAlias=" + args.host]
        try:
            result = subprocess.run(ssh + [args.host, command], capture_output=True, text=True,
                                    encoding="utf-8", timeout=150, check=True)
            for line in reversed(result.stdout.splitlines()):
                if line.startswith("{"):
                    return json.loads(line)
            raise ValueError("Missing controller JSON: " + result.stdout[-1000:])
        except (subprocess.SubprocessError, ValueError) as exc:
            errors.append(str(exc))
    raise RuntimeError("SSH/controller failed: " + "; ".join(errors))


def fetch_provisional(args):
    command = "cd " + shlex.quote(args.root) + " && " + shlex.join([
        args.root + "/.venv-optuna-search/bin/python", "-m", "scripts.read_optuna_provisional",
        "--campaign", args.campaign])
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             args.host, command], capture_output=True, text=True, encoding="utf-8",
                            timeout=90, check=True)
    return json.loads(result.stdout.splitlines()[-1])


def publish_provisional(folder, rows):
    columns = ["dataset", "trial", "seed", "lr", "weight_decay", "dropout",
               "selected_epoch", "validation_bacc", "test_balanced_accuracy",
               "test_auroc", "test_auprc", "test_weighted_f1", "test_kappa", "source_result"]
    flattened = []
    for row in rows:
        hp, scores = row["hp"], row["test"]
        flattened.append(dict(dataset=row["dataset"], trial=row["trial"], seed=row["seed"],
                              lr=hp["lr"], weight_decay=hp["wd"], dropout=hp["dropout"],
                              selected_epoch=row["selected_epoch"],
                              validation_bacc=row["validation_bacc"],
                              test_balanced_accuracy=scores["balanced_accuracy"],
                              test_auroc=scores.get("auroc"), test_auprc=scores.get("auprc"),
                              test_weighted_f1=scores.get("weighted_f1"),
                              test_kappa=scores.get("kappa"), source_result=row["source_result"]))
    atomic_text(folder / "provisional_seed_results.csv", csv_text(flattened, columns))
    lines = ["# Completed seeds (provisional)", "",
             "Each checkpoint is selected on validation. Test scores compare Optuna trials; "
             "a trial is not complete until all five seeds finish.", "",
             "| Dataset | Trial | Seed | Val BAcc | Test BAcc | Test AUROC | Test AUPRC | Test weighted F1 | Test kappa |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in flattened:
        values = [row["dataset"], row["trial"], row["seed"], row["validation_bacc"],
                  row["test_balanced_accuracy"], row["test_auroc"], row["test_auprc"],
                  row["test_weighted_f1"], row["test_kappa"]]
        lines.append("| " + " | ".join("—" if value is None else str(value) for value in values) + " |")
    atomic_text(folder / "provisional.md", "\n".join(lines) + "\n")


def publish(args, status):
    if status["status"] != "complete" or set(status["datasets"]) != set(NAMES):
        raise ValueError("All five requested datasets must finish before publication")
    # Prepare every table in memory after checking every seed, before any final write.
    for slug, ds in status["datasets"].items():
        if ds["status"] != "complete":
            raise ValueError("Incomplete dataset")
        summary = five_seed_summary(ds["winner"]["rows"])
        if summary != ds["winner"]["metrics"]:
            raise ValueError("Winner mean/SD differs from its five individual seeds")
    literature = json.loads((ROOT / "configs/results/literature_baselines.json").read_text(encoding="utf-8"))
    methods = literature["model_order"]
    ours = "우리 (" + args.alias + ")"
    files, sections = [], []
    for slug, name in NAMES.items():
        winner = status["datasets"][slug]["winner"]
        folder = ROOT / "outputs/results" / (args.stamp + "-" + slug + "-" + args.alias)
        references = literature["datasets"].get(slug, literature["datasets"].get("stress", {}) if slug == "mentalarithmetic" else {})
        rows, seed_rows, mdrows = [], [], []
        for metric, aggregate in winner["metrics"].items():
            mean, sd = aggregate["mean"], aggregate["population_sd"]
            means = {ours: mean, **{model: values[0] for model, values in references.get(metric, {}).items() if model in methods}}
            ranks = dense_ranks(means)
            row = {"Metric": metric, ours: mean, **{model: means.get(model) for model in methods},
                   "ours_mean": mean, "ours_population_sd": sd, "ours_n": 5, "ours_rank": ranks[ours],
                   "selection_protocol": "validation_checkpoint_test_trial_selection"}
            text = [metric, display(mean, sd, ranks[ours])]
            for model in methods:
                pair = references.get(metric, {}).get(model)
                row.update({model + "_mean": pair[0] if pair else None,
                            model + "_sd": pair[1] if pair else None, model + "_n": None,
                            model + "_rank": ranks.get(model)})
                text.append(display(pair[0], pair[1], ranks[model]) if pair else "—")
            rows.append(row)
            mdrows.append("| " + " | ".join(text) + " |")
            for seed in winner["rows"]:
                seed_rows.append(dict(dataset=name, seed=seed["seed"], metric=metric, value=seed["test"][metric],
                                      selected_epoch=seed["selection"]["epoch"], selector="validation balanced_accuracy",
                                      source_result=seed["source_result"], trial=winner["trial"]))
        columns = ["Metric", ours, *methods, "ours_mean", "ours_population_sd", "ours_n", "ours_rank", "selection_protocol"]
        for model in methods:
            columns += [model + "_mean", model + "_sd", model + "_n", model + "_rank"]
        md = ["# " + name + " — " + args.alias, "", PROTOCOL, "",
              "Five seeds: 42, 696, 1001, 1234, 3407. Population SD. Trial: " + str(winner["trial"]),
              "Parameters: " + json.dumps(winner["hp"]), "",
              "| Metric | " + ours + " | " + " | ".join(methods) + " |",
              "|---|" + "---:|" * (len(methods) + 1), *mdrows, "",
              "**bold**: highest mean; <u>underline</u>: second-highest mean. Published studies were not retuned here.", ""]
        files += [(folder / "results.csv", csv_text(rows, columns)),
                  (folder / "seed_results.csv", csv_text(seed_rows, ["dataset", "seed", "metric", "value", "selected_epoch", "selector", "source_result", "trial"])),
                  (folder / "results.md", "\n".join(md)),
                  (folder / "selection.json", json.dumps(winner, indent=2) + "\n")]
        sections.append(dict(slug=slug, title=name + " — " + args.alias, folder=folder.name, markdown="\n".join(md[2:])))
    for path, content in files:
        atomic_text(path, content)
    index = ROOT / "outputs/results/RESULTS.md"
    heading = "## " + args.stamp + " — " + args.alias
    if not index.exists() or heading not in index.read_text(encoding="utf-8"):
        update_global(sections, args.stamp, args.alias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="bigpurple.nyumc.org")
    parser.add_argument("--fallback-hosts", nargs="*", default=["10.189.18.57", "10.189.18.58", "10.189.18.56"])
    parser.add_argument("--root", default="/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--alias", default="knn37-optuna-test")
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--read-only", action="store_true", help="Only sync status/results; a server CPU job advances Optuna")
    args = parser.parse_args()
    if len(args.stamp) != 8 or not args.stamp.isdigit():
        raise ValueError("Expected MMDDHHmm timestamp")
    folder = ROOT / "outputs/results" / ("search-" + args.stamp + "-" + args.alias)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_text(folder / "monitor.pid", str(os.getpid()))
    while True:
        if (folder / "STOP").exists():
            return
        try:
            status = fetch(args)
            atomic_text(folder / "status.json", json.dumps(status, indent=2) + "\n")
            try:
                publish_provisional(folder, fetch_provisional(args))
            except Exception as exc:
                atomic_text(folder / "provisional_error.json", json.dumps(dict(error=repr(exc), utc=datetime.now(timezone.utc).isoformat())))
            lines = ["# Optuna search — provisional progress", "", PROTOCOL, "",
                     "Updated: " + status["checked_utc"], "", "| Dataset | Status | Trial | Completed seeds | Job |",
                     "|---|---|---:|---:|---|"]
            for slug, ds in status["datasets"].items():
                lines.append("| " + " | ".join([NAMES[slug], ds["status"], str(ds.get("trial", "—")),
                    str(ds.get("completed_seeds", "—")) + "/5", str(ds.get("job_id", "—"))]) + " |")
            for slug, ds in status["datasets"].items():
                if ds.get("status") == "fixed":
                    lines += ["", NAMES[slug] + ": fixed by user; new trials and retries disabled. "
                              + ds["reporting_note"]]
            atomic_text(folder / "progress.md", "\n".join(lines) + "\n")
            print(json.dumps(dict(status=status["status"], utc=status["checked_utc"])), flush=True)
            if status["status"] == "complete":
                publish(args, status)
                atomic_text(folder / "published.json", json.dumps(dict(published_utc=datetime.now(timezone.utc).isoformat())))
                return
            if status["status"] == "finished_with_fixed_settings":
                atomic_text(folder / "finished_with_fixed_settings.json", json.dumps(dict(
                    utc=datetime.now(timezone.utc).isoformat(), final_aggregate_published=False,
                    reason="Campaign includes user-fixed exploratory settings; see progress and individual artifacts")))
                return
        except Exception as exc:
            atomic_text(folder / "monitor_error.json", json.dumps(dict(error=repr(exc), utc=datetime.now(timezone.utc).isoformat())))
            print(repr(exc), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
