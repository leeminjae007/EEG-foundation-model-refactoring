"""Run one prepared experiment downstream array entry and publish its CSV row."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path); parser.add_argument("--source", type=Path); args = parser.parse_args()
entries = json.loads((args.experiment / "downstream_entries.json").read_text(encoding="utf-8"))
entry = entries[int(os.environ["SLURM_ARRAY_TASK_ID"])]
source = (args.source or args.experiment / "source").resolve()
command = [sys.executable, str(source / "scripts/run_downstream_with_results.py"),
           "--experiment", str(args.experiment), "--config", entry["config"],
           "--dataset", entry["dataset"], "--seed", str(entry["seed"]), "--source", str(source)]
output = Path(entry["output"])
if (output / "result.json").exists():
    command += ["--resume", str(output / "last.pth")] if (output / "last.pth").exists() else []
subprocess.run(command, check=True)
