"""Collect every active five-seed downstream campaign into the local results tree."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from scripts import monitor_experiment_results as publisher


REMOTE_ROOT = "/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model"
FULL_CAMPAIGNS = (
    ("knn37-default", "outputs/knn37_downstream_all13_20260915/manifest.json"),
    ("cbramod-encoder-ablation", "outputs/ablation/encoder_cbramod/downstream_no_seedvig_20260915/manifest.json"),
    ("reve4d-positional-encoding-ablation", "outputs/ablation/pe_reve4d/downstream_no_seedvig_20260915/manifest.json"),
)
TUAB_MANIFEST = "outputs/tuab_recovery_retry2_20260915/manifest.json"
TUAB_ALIASES = {
    "backbone_lr_x0p1": "knn37-parameter-tuning-tuab-backbone-lr1e6",
    "head_first2_backbone_lr_x0p1": "knn37-parameter-tuning-tuab-head-first2-backbone-lr1e6",
    "head_h4": "knn37-parameter-tuning-tuab-head-h4",
}
BEAM_ROOT = "outputs/bciciv2a_beam_w0_20260915"
METHODS = ("우리 모델", "CBraMod", "CSBrain", "REVE")
METRIC_ORDER = ("balanced_accuracy", "auroc", "auprc", "weighted_f1", "kappa")


def stamp(manifest: dict[str, Any]) -> str:
    created = datetime.fromisoformat(manifest["created_utc"])
    if ZoneInfo is None:
        return created.strftime("%Y%m%d-%H%M")
    return created.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M")


def normalized(entries: list[dict[str, Any]], slug: str | None = None,
               display: str | None = None) -> dict[str, Any]:
    values = []
    for entry in publisher.active_entries(entries):
        item = dict(entry)
        if slug is not None:
            item.update(slug=slug, dataset=slug, display_name=display or slug)
        values.append(item)
    return {"submitted_entries": values, "reused_entries": []}


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"published": {}, "checks": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["checked_at"] = datetime.now().astimezone().isoformat()
    publisher.atomic_text(path, json.dumps(state, indent=2) + "\n")


def formatted(mean: float | None, sd: float | None, rank: int | None) -> str:
    if mean is None:
        return "—"
    value = f"{mean:.4f}" if sd is None else f"{mean:.4f} ± {sd:.4f}"
    if rank == 1:
        return "**" + value + "**"
    if rank == 2:
        return "<u>" + value + "</u>"
    return value


def publish_experiment(entries_manifest: dict[str, Any], results: dict[str, dict[str, Any]],
                       filename: str, literature: dict[str, Any]) -> Path:
    entries = entries_manifest["submitted_entries"] + entries_manifest["reused_entries"]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_dataset.setdefault(entry["slug"], []).append(entry)
    rows: list[dict[str, Any]] = []
    for slug, dataset_entries in sorted(by_dataset.items(), key=lambda item: publisher.dataset_sort_key(item[0])):
        seeds = {int(entry["seed"]) for entry in dataset_entries}
        if seeds != publisher.SEEDS or len(dataset_entries) != 5:
            raise ValueError(f"{slug}: expected exactly five fixed seeds, got {sorted(seeds)}")
        per_metric: dict[str, dict[int, float]] = {}
        for entry in dataset_entries:
            result_path = str(entry.get("result_dir") or entry.get("output")).rstrip("/") + "/result.json"
            for metric, value in publisher.selector_test(results[result_path]).items():
                if metric in METRIC_ORDER:
                    per_metric.setdefault(metric, {})[int(entry["seed"])] = value
        if len(per_metric) != 3 or any(set(values) != publisher.SEEDS for values in per_metric.values()):
            raise ValueError(f"{slug}: expected three complete metrics across five seeds")
        literature_slug = publisher.SLUG_ALIASES.get(dataset_entries[0]["dataset"], slug)
        references = literature["datasets"].get(literature_slug, literature["datasets"].get(slug, {}))
        for metric in sorted(per_metric, key=METRIC_ORDER.index):
            seed_values = per_metric[metric]
            values = [seed_values[seed] for seed in sorted(publisher.SEEDS)]
            ours_mean, ours_sd = statistics.fmean(values), statistics.pstdev(values)
            means = {"우리 모델": ours_mean}
            for model, pair in references.get(metric, {}).items():
                means[model] = float(pair[0])
            ranks = publisher.dense_ranks(means)
            row: dict[str, Any] = {"Dataset": dataset_entries[0]["display_name"], "Metric": metric}
            for seed in sorted(publisher.SEEDS):
                row["우리 모델_seed" + str(seed)] = seed_values[seed]
            for method in METHODS:
                if method == "우리 모델":
                    mean, sd, n = ours_mean, ours_sd, 5
                else:
                    pair = references.get(metric, {}).get(method)
                    mean, sd, n = ((None, None, None) if pair is None else (pair[0], pair[1], None))
                rank = ranks.get(method)
                row[method] = formatted(mean, sd, rank)
                row[method + "_mean"] = mean
                row[method + "_sd"] = sd
                row[method + "_n"] = n
                row[method + "_rank"] = rank
            rows.append(row)
    columns = ["Dataset", "Metric", *METHODS]
    columns.extend("우리 모델_seed" + str(seed) for seed in sorted(publisher.SEEDS))
    for method in METHODS:
        columns.extend([method + "_mean", method + "_sd", method + "_n", method + "_rank"])
    path = publisher.RESULTS_ROOT / (filename + ".csv")
    publisher.write_csv(path, rows, columns)
    return path


def try_publish(ssh: list[str], key: str, alias: str, manifest: dict[str, Any],
                entries_manifest: dict[str, Any], literature: dict[str, Any],
                state: dict[str, Any]) -> None:
    if key in state["published"]:
        return
    entries = entries_manifest["submitted_entries"] + entries_manifest["reused_entries"]
    results, missing = publisher.fetch_results(ssh, entries)
    state["checks"][key] = {"expected": len(entries), "missing": len(missing)}
    if missing:
        return
    filename = alias + "-" + stamp(manifest)
    path = publish_experiment(entries_manifest, results, filename, literature)
    state["published"][key] = {"experiment": alias, "file": path.name}


def cycle(ssh: list[str], literature: dict[str, Any], state: dict[str, Any]) -> None:
    for alias, relative in FULL_CAMPAIGNS:
        manifest = publisher.remote_json(ssh, REMOTE_ROOT + "/" + relative)
        entries = normalized(manifest["submitted_entries"] + manifest["reused_entries"])
        try_publish(ssh, alias, alias, manifest, entries, literature, state)

    tuab = publisher.remote_json(ssh, REMOTE_ROOT + "/" + TUAB_MANIFEST)
    for arm, alias in TUAB_ALIASES.items():
        entries = [entry for entry in tuab["entries"] if entry["arm"] == arm]
        try_publish(ssh, "tuab:" + arm, alias, tuab,
                    normalized(entries, "tuab", "TUAB"), literature, state)

    status_path = REMOTE_ROOT + "/" + BEAM_ROOT + "/status.json"
    try:
        beam_status = publisher.remote_json(ssh, status_path)
    except Exception as error:
        state["checks"]["bciciv2a-beam"] = {"status": "unavailable", "error": str(error)}
        return
    state["checks"]["bciciv2a-beam"] = {"status": beam_status.get("status", "unknown")}
    if beam_status.get("status") != "complete":
        return
    beam_manifest = publisher.remote_json(ssh, REMOTE_ROOT + "/" + BEAM_ROOT + "/manifest.json")
    final_plan = publisher.remote_json(ssh, REMOTE_ROOT + "/" + BEAM_ROOT + "/stages/final/plan.json")
    try_publish(ssh, "bciciv2a-beam", "knn37-parameter-tuning-bciciv2a-beam", beam_manifest,
                normalized(final_plan["entries"], "bciciv2a", "BCIC-IV-2a"), literature, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="bigpurple.nyumc.org")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    publisher.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = publisher.RESULTS_ROOT / "collection-state.json"
    literature = json.loads(publisher.LITERATURE.read_text(encoding="utf-8"))
    ssh = publisher.ssh_base(args.host, None, None)
    while True:
        state = load_state(state_path)
        try:
            cycle(ssh, literature, state)
            state["last_error"] = None
        except Exception as error:
            state["last_error"] = repr(error)
        save_state(state_path, state)
        expected = 3 + len(TUAB_ALIASES) + 1
        if len(state["published"]) == expected:
            return 0
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
