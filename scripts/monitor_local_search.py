"""CPU-only local monitor and gated five-seed publisher for the knn37 search."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from scripts.monitor_experiment_results import atomic_text, dense_ranks, display, update_global

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {42, 696, 1001, 1234, 3407}


def remote(host, command):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                            capture_output=True, text=True, encoding="utf-8", timeout=90, check=True)
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("No JSON received from server")


def fetch_status(args):
    python = args.root + "/.venv/bin/python"
    script = """import json,pathlib,subprocess
from ablation.local_search.campaign import inspect
c = pathlib.Path(CAMPAIGN_LITERAL)
r = inspect(c)
jobs = []
for p in list(c.glob('*submission.json')) + list((c/'datasets').glob('*/*/*submission*.json')) + list((c/'datasets').glob('*/*/retry*.json')):
 try:
  d=json.loads(p.read_text())
  if 'job_id' in d: jobs.append(d['job_id'])
 except (OSError,ValueError): pass
r['slurm'] = subprocess.run(['sacct','-X','-j',','.join(sorted(set(jobs))), '--format=JobIDRaw,State,ExitCode,Elapsed','-n','-P'],capture_output=True,text=True,timeout=30).stdout if jobs else ''
r['cpu_smoke_passed'] = (c/'smoke_validation.json').exists()
if not r['cpu_smoke_passed']:
 for p in c.glob('smoke-*.log'):
  r['cpu_smoke_log_tail']=p.read_text(errors='replace')[-3000:]
print(json.dumps(r))
""".replace("CAMPAIGN_LITERAL", repr(args.campaign))
    command = "cd " + shlex.quote(args.campaign + "/source") + " && PYTHONNOUSERSITE=1 " + shlex.join([python, "-c", script])
    return remote(args.host, command)


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def progress(folder, status):
    rows = []
    md = ["# knn37 parameter search — provisional validation progress", "",
          "Not a final aggregate. Candidate selection uses validation BAcc only.", "",
          "Updated: " + status["checked_utc"], "",
          "CPU smoke passed: " + str(status.get("cpu_smoke_passed", False)), "",
          "| Dataset | Initial seed | Status | Finished / planned trials |", "|---|---:|---|---:|"]
    for slug, d in status["datasets"].items():
        trials = d["trials"]
        md.append(f"| {d['name']} | {d['start_seed']} | {d['status']['status']} | {sum(t['status'] == 'finished' for t in trials)} / {len(trials)} |")
        for r in trials:
            rows.append(dict(dataset=d["name"], stage=r["stage"], seed=r["seed"], candidate=r["candidate"],
                             **r["hp"], status=r["status"], validation_bacc=r.get("validation"),
                             latest_epoch=r.get("epoch"), output=r["output"]))
    if status.get("slurm"):
        md += ["", "Job ID | State | Exit code | Elapsed:", "", "```text", status["slurm"].strip(), "```", ""]
    atomic_text(folder / "status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    atomic_text(folder / "progress.md", "\n".join(md) + "\n")
    atomic_text(folder / "trials.csv", csv_text(rows, ["dataset", "stage", "seed", "candidate", "lr", "wd", "dropout", "status", "validation_bacc", "latest_epoch", "output"]))


def fetch_final(args, status):
    entries = {slug: d["final"]["entries"] for slug, d in status["datasets"].items()}
    script = """import json,pathlib,math
entries = ENTRY_LITERAL
out = {}
for slug, values in entries.items():
 out[slug] = []
 for e in values:
  path = pathlib.Path(e['output'])/'result.json'
  r = json.loads(path.read_text())['balanced_accuracy']
  metrics = {k:float(v) for k,v in r['test'].items() if k in ['balanced_accuracy','auroc','auprc','weighted_f1','kappa']}
  out[slug].append({'seed':e['seed'],'selection':r['selection'],'test':metrics,'source_result':str(path)})
print(json.dumps(out))
""".replace("ENTRY_LITERAL", repr(entries))
    encoded = base64.b64encode(script.encode()).decode()
    command = shlex.join([args.root + "/.venv/bin/python", "-c", "import base64;exec(base64.b64decode('" + encoded + "'))"])
    return remote(args.host, command)


def publish(args, status, final):
    literature = json.loads((ROOT / "configs/results/literature_baselines.json").read_text(encoding="utf-8"))
    if set(final) != set(status["datasets"]) or len(final) != 5:
        raise ValueError("All requested datasets are required before publication")
    # Validate the complete campaign before writing any final table.
    for slug, values in final.items():
        if len(values) != 5 or {v["seed"] for v in values} != SEEDS:
            raise ValueError(slug + ": missing or duplicate final seeds")
        expected = {"balanced_accuracy", "auroc", "auprc"} if slug in ("tuab", "mentalarithmetic") else {"balanced_accuracy", "weighted_f1", "kappa"}
        for value in values:
            if set(value["test"]) != expected or not all(math.isfinite(x) for x in value["test"].values()):
                raise ValueError(slug + ": missing or invalid final metrics")
    methods = literature["model_order"]
    ours = "우리 (" + args.alias + ")"
    prepared = []
    sections = []
    for slug, values in final.items():
        d = status["datasets"][slug]
        folder = ROOT / "outputs/results" / (args.stamp + "-" + slug + "-" + args.alias)
        references = literature["datasets"].get(slug, literature["datasets"].get("stress", {}) if slug == "mentalarithmetic" else {})
        rows, seeds, mdrows = [], [], []
        for metric in sorted(values[0]["test"]):
            samples = [v["test"][metric] for v in values]
            mean, sd = statistics.mean(samples), statistics.pstdev(samples)
            means = {ours: mean, **{k: pair[0] for k, pair in references.get(metric, {}).items() if k in methods}}
            ranks = dense_ranks(means)
            row = {"Metric": metric, ours: mean}
            row.update({method: means.get(method) for method in methods})
            row.update(ours_mean=mean, ours_population_sd=sd, ours_n=5, ours_rank=ranks[ours])
            text_values = [metric, display(mean, sd, ranks[ours])]
            for method in methods:
                pair = references.get(metric, {}).get(method)
                row.update({method + "_mean": pair[0] if pair else None,
                            method + "_sd": pair[1] if pair else None,
                            method + "_n": None, method + "_rank": ranks.get(method)})
                text_values.append(display(pair[0], pair[1], ranks[method]) if pair else "—")
            rows.append(row)
            mdrows.append("| " + " | ".join(text_values) + " |")
            for v in sorted(values, key=lambda v: v["seed"]):
                seeds.append(dict(dataset=d["name"], seed=v["seed"], metric=metric, value=v["test"][metric],
                                  validation_bacc=v["selection"]["score"], selected_epoch=v["selection"]["epoch"],
                                  selector="validation balanced_accuracy", source_result=v["source_result"]))
        columns = ["Metric", ours, *methods, "ours_mean", "ours_population_sd", "ours_n", "ours_rank"]
        for method in methods:
            columns += [method + "_mean", method + "_sd", method + "_n", method + "_rank"]
        metadata = d["final"]
        md = ["# " + d["name"] + " — " + args.alias, "",
              "Five completed seeds: 42, 696, 1001, 1234, 3407. Population SD.", "",
              "Selected: " + metadata["selected"] + "; parameters: " + json.dumps(metadata["hp"]),
              "Search: common LR, validation-only selection, campaign-specific early stopping; original cosine horizon retained.",
              "Baseline retained where candidate improvements did not reproduce. Published references were not retuned.", "",
              "| Metric | " + ours + " | " + " | ".join(methods) + " |", "|---|---:|---:|---:|---:|", *mdrows, "",
              "**bold**: highest mean; <u>underline</u>: second-highest mean. Unknown reference sample counts are blank.", ""]
        prepared += [(folder / "results.csv", csv_text(rows, columns)),
                     (folder / "seed_results.csv", csv_text(seeds, ["dataset", "seed", "metric", "value", "validation_bacc", "selected_epoch", "selector", "source_result"])),
                     (folder / "results.md", "\n".join(md)),
                     (folder / "selection.json", json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")]
        sections.append(dict(slug=slug, title=d["name"] + " — " + args.alias, folder=folder.name, markdown="\n".join(md[4:])))
    for path, content in prepared:
        atomic_text(path, content)
    global_path = ROOT / "outputs/results/RESULTS.md"
    heading = "## " + args.stamp + " — " + args.alias
    if not global_path.exists() or heading not in global_path.read_text(encoding="utf-8"):
        update_global(sections, args.stamp, args.alias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="bigpurple.nyumc.org")
    p.add_argument("--root", default="/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model")
    p.add_argument("--campaign", required=True)
    p.add_argument("--stamp", required=True)
    p.add_argument("--alias", default="knn37-hp")
    p.add_argument("--interval", type=int, default=180)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    folder = ROOT / "outputs/results" / ("search-" + args.stamp + "-" + args.alias)
    while True:
        try:
            status = fetch_status(args)
            progress(folder, status)
            if status["ready_to_publish"]:
                if not (folder / "published.json").exists():
                    publish(args, status, fetch_final(args, status))
                    atomic_text(folder / "published.json", json.dumps({"published_utc": datetime.now(timezone.utc).isoformat(), "datasets": 5, "seeds_per_dataset": 5}))
                print("Five-seed final results published", flush=True)
                return
            print(json.dumps({slug: d["status"] for slug, d in status["datasets"].items()}), flush=True)
        except Exception as exc:
            atomic_text(folder / "monitor_error.json", json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}, indent=2))
            print(repr(exc), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
