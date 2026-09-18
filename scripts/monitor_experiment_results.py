"""CPU-only monitor and publisher for five-seed downstream experiments."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "outputs" / "results"
LITERATURE = ROOT / "configs" / "results" / "literature_baselines.json"
SEEDS = {42, 696, 1001, 1234, 3407}
START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"
SLUG_ALIASES = {"seed-v": "seedv", "stress": "mentalarithmetic", "physio": "physionet_mi"}
RETIRED_DATASETS = {"mumtaz", "speech", "bcic2020_3"}
DATASET_ORDER = (
    "chb", "siena", "physionet_mi", "tuev", "tuab", "faced",
    "seedv", "mentalarithmetic", "isruc", "hmc",
)
DATASET_RANK = {slug: index for index, slug in enumerate(DATASET_ORDER)}


def dataset_sort_key(slug: str) -> tuple[int, str]:
    canonical = SLUG_ALIASES.get(slug.lower(), slug.lower())
    return DATASET_RANK.get(canonical, len(DATASET_ORDER)), canonical


def active_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries
            if entry.get("slug", "").lower() not in RETIRED_DATASETS
            and entry.get("dataset", "").lower() not in RETIRED_DATASETS]


def run(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(command, input=input_text, text=True, capture_output=True, check=True)
    return completed.stdout


def parse_json_output(value: str) -> Any:
    starts = [position for token in ("{", "[") if (position := value.find(token)) >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON payload in command output", value, 0)
    return json.loads(value[min(starts):])


def ssh_base(host: str, hostname: str | None, hostkey_alias: str | None) -> list[str]:
    command = ["ssh", "-o", "BatchMode=yes"]
    if hostname:
        command += ["-o", "HostName=" + hostname]
    if hostkey_alias:
        command += ["-o", "HostKeyAlias=" + hostkey_alias]
    return command + [host]


def remote_json(ssh: list[str], path: str) -> dict[str, Any]:
    return parse_json_output(run(ssh + ["cat", path]))


def fetch_results(ssh: list[str], entries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    path_entries = []
    for entry in entries:
        result_dir = entry.get("result_dir") or entry.get("output")
        path = str(result_dir).rstrip("/") + "/result.json"
        path_entries.append((path, entry))
    paths = [item[0] for item in path_entries]
    code = (
        "import json,pathlib\n"
        "paths=" + repr(paths) + "\n"
        "found={}\nmissing=[]\n"
        "for p in paths:\n"
        " try: found[p]=json.loads(pathlib.Path(p).read_text())\n"
        " except (OSError,ValueError): missing.append(p)\n"
        "print(json.dumps({'found':found,'missing':missing}))\n"
    )
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    payload = parse_json_output(run(ssh + ["echo " + encoded + " | base64 -d | python3"]))
    missing = []
    by_path = dict(path_entries)
    for path in payload["missing"]:
        entry = by_path[path]
        missing.append({"dataset": entry["display_name"], "slug": entry["slug"],
                        "seed": int(entry["seed"]), "path": path})
    return payload["found"], missing


def job_states(ssh: list[str], jobs: list[str]) -> str:
    if not jobs:
        return ""
    return run(ssh + ["sacct", "-X", "-j", ",".join(jobs),
                "--format=JobIDRaw,State,ExitCode", "-n", "-P"])


def all_jobs_terminal(states: str, jobs: list[str]) -> bool:
    active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}
    observed: dict[str, str] = {}
    for line in states.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and fields[0] in jobs:
            observed[fields[0]] = fields[1].split()[0]
    return len(observed) == len(jobs) and all(state not in active for state in observed.values())


def selector_test(result: dict[str, Any]) -> dict[str, float]:
    if "balanced_accuracy" not in result:
        raise ValueError("Classification result lacks the balanced_accuracy validation selector")
    test = result["balanced_accuracy"].get("test", {})
    values = {key: float(value) for key, value in test.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))}
    if "balanced_accuracy" not in values:
        raise ValueError("Selected checkpoint lacks test balanced_accuracy")
    return values


def dense_ranks(values: dict[str, float]) -> dict[str, int]:
    distinct = sorted(set(values.values()), reverse=True)
    return {name: distinct.index(value) + 1 for name, value in values.items()}


def display(mean: float | None, sd: float | None, rank: int | None) -> str:
    if mean is None:
        return "—"
    value = f"{mean:.4f}" if sd is None else f"{mean:.4f} ± {sd:.4f}"
    if rank == 1:
        return "**" + value + "**"
    if rank == 2:
        return "<u>" + value + "</u>"
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # The server uses Python 3.8, where Path.write_text has no newline argument.
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish(manifest: dict[str, Any], results: dict[str, dict[str, Any]], stamp: str,
            alias: str, literature: dict[str, Any]) -> list[dict[str, str]]:
    entries = manifest["submitted_entries"] + manifest["reused_entries"]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_dataset.setdefault(entry["slug"], []).append(entry)
    sections: list[dict[str, str]] = []
    model_order = literature["model_order"]
    for slug, dataset_entries in sorted(by_dataset.items(), key=lambda item: dataset_sort_key(item[0])):
        seeds = {int(item["seed"]) for item in dataset_entries}
        if seeds != SEEDS or len(dataset_entries) != 5:
            raise ValueError(f"{slug}: expected the fixed five seeds, got {sorted(seeds)}")
        per_seed = []
        for entry in sorted(dataset_entries, key=lambda item: int(item["seed"])):
            result_path = str(entry.get("result_dir") or entry.get("output")).rstrip("/") + "/result.json"
            metrics = selector_test(results[result_path])
            for metric, value in sorted(metrics.items()):
                per_seed.append({"dataset": entry["display_name"], "seed": int(entry["seed"]),
                                 "metric": metric, "value": value,
                                 "selector": "validation balanced_accuracy", "source_result": result_path})
        metrics = sorted({row["metric"] for row in per_seed})
        literature_slug = SLUG_ALIASES.get(dataset_entries[0]["dataset"], slug)
        references = literature["datasets"].get(literature_slug, literature["datasets"].get(slug, {}))
        comparison_rows = []
        md_rows = []
        for metric in metrics:
            values = [row["value"] for row in per_seed if row["metric"] == metric]
            ours_mean = statistics.fmean(values)
            ours_sd = statistics.pstdev(values)
            means = {"ours": ours_mean}
            for model, pair in references.get(metric, {}).items():
                means[model] = float(pair[0])
            ranks = dense_ranks(means)
            row: dict[str, Any] = {"metric": metric, "ours_mean": ours_mean, "ours_population_sd": ours_sd,
                                   "ours_n": 5, "ours_rank": ranks["ours"]}
            md_values = [metric, display(ours_mean, ours_sd, ranks["ours"])]
            for model in model_order:
                pair = references.get(metric, {}).get(model)
                row[model + "_mean"] = None if pair is None else pair[0]
                row[model + "_sd"] = None if pair is None else pair[1]
                row[model + "_rank"] = None if pair is None else ranks[model]
                md_values.append("—" if pair is None else display(pair[0], pair[1], ranks[model]))
            comparison_rows.append(row)
            md_rows.append("| " + " | ".join(md_values) + " |")

        folder = RESULTS_ROOT / f"{stamp}-{slug}-{alias}"
        columns = ["metric", "ours_mean", "ours_population_sd", "ours_n", "ours_rank"]
        for model in model_order:
            columns.extend([model + "_mean", model + "_sd", model + "_rank"])
        write_csv(folder / "results.csv", comparison_rows, columns)
        write_csv(folder / "seed_results.csv", per_seed,
                  ["dataset", "seed", "metric", "value", "selector", "source_result"])
        title = dataset_entries[0]["display_name"] + " — " + alias
        md = ["# " + title, "", "5 seeds: 42, 696, 1001, 1234, 3407. Population SD.", "",
              "| Metric | 우리 (" + alias + ") | " + " | ".join(model_order) + " |",
              "|---|---:|---:|---:|---:|", *md_rows, "",
              "**bold**: metric 1위. <u>underline</u>: 2위.", ""]
        atomic_text(folder / "results.md", "\n".join(md))
        sections.append({"slug": slug, "title": title, "folder": folder.name,
                         "markdown": "\n".join(md[4:-1])})
    update_global(sections, stamp, alias)
    return sections


def update_global(sections: list[dict[str, str]], stamp: str, alias: str) -> None:
    path = RESULTS_ROOT / "RESULTS.md"
    original = path.read_text(encoding="utf-8") if path.exists() else "# Experiment results\n\n" + START + "\n" + END + "\n"
    if START not in original or END not in original:
        raise ValueError("Global results Markdown is missing auto-update markers")
    old = original.split(START, 1)[1].split(END, 1)[0].strip()
    heading = "## " + stamp + " — " + alias
    block = [heading, ""]
    for section in sections:
        block.extend(["### " + section["title"], "", section["markdown"], "",
                      "CSV: [`" + section["folder"] + "/results.csv`](" + section["folder"] + "/results.csv)", ""])
    retained = "" if old == "아직 집계 완료된 신규 실험이 없습니다." else old + "\n\n"
    replacement = START + "\n" + retained + "\n".join(block).rstrip() + "\n" + END
    atomic_text(path, original.split(START, 1)[0] + replacement + original.split(END, 1)[1])


def write_status(path: Path, status: str, missing: list[dict[str, Any]], states: str) -> None:
    payload = {"status": status, "checked_utc": datetime.now(timezone.utc).isoformat(),
               "missing_count": len(missing), "missing": missing, "slurm": states}
    atomic_text(path, json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="bigpurple.nyumc.org")
    parser.add_argument("--ssh-hostname")
    parser.add_argument("--ssh-hostkey-alias")
    parser.add_argument("--remote-manifest", required=True)
    parser.add_argument("--jobs", nargs="*", default=[])
    parser.add_argument("--stamp", required=True, help="MMDDHHmm")
    parser.add_argument("--alias", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--include-retired", action="store_true",
                        help="Monitor every dataset in the manifest, including legacy datasets")
    args = parser.parse_args()
    if len(args.stamp) != 8 or not args.stamp.isdigit():
        raise ValueError("--stamp must be MMDDHHmm")
    ssh = ssh_base(args.host, args.ssh_hostname, args.ssh_hostkey_alias)
    manifest = remote_json(ssh, args.remote_manifest)
    manifest = dict(manifest)
    if not args.include_retired:
        manifest["submitted_entries"] = active_entries(manifest["submitted_entries"])
        manifest["reused_entries"] = active_entries(manifest["reused_entries"])
    entries = manifest["submitted_entries"] + manifest["reused_entries"]
    if not entries:
        raise ValueError("No active downstream entries in campaign")
    literature = json.loads(LITERATURE.read_text(encoding="utf-8"))
    status_path = RESULTS_ROOT / ("monitor-" + args.stamp + "-" + args.alias + ".json")
    while True:
        results, missing = fetch_results(ssh, entries)
        states = job_states(ssh, args.jobs)
        if not missing:
            sections = publish(manifest, results, args.stamp, args.alias, literature)
            write_status(status_path, "complete", [], states)
            print(json.dumps({"status": "complete", "datasets": len(sections), "runs": len(entries)}))
            return 0
        if all_jobs_terminal(states, args.jobs):
            write_status(status_path, "incomplete_terminal_jobs", missing, states)
            print(json.dumps({"status": "incomplete_terminal_jobs", "missing": len(missing)}), flush=True)
            return 3
        write_status(status_path, "monitoring", missing, states)
        print(json.dumps({"status": "monitoring", "missing": len(missing)}), flush=True)
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
