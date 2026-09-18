"""Create an isolated, timestamped server experiment folder and source snapshot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("experiment_name"); parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(); stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%y%m%d-%H%M")
    folder = args.results_root / (stamp + "-" + args.experiment_name)
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    shutil.copytree(ROOT, folder / "source", ignore=shutil.ignore_patterns(".git", ".venv*", "outputs", "Results", "cache", "tmp", "__pycache__", "*.pyc", "Data"))
    for name in ("configs", "pretrain/checkpoints", "pretrain/logs", "pretrain/cache", "pretrain/tmp", "downstream"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    (folder / "manifest.json").write_text(json.dumps({"experiment": args.experiment_name, "created_at_new_york": stamp, "source_commit": commit, "status": "prepared"}, indent=2) + "\n", encoding="utf-8")
    print(folder)
