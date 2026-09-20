"""Create an isolated, timestamped server experiment folder and source snapshot."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get(
    "EEGFM_RESULTS_ROOT",
    "/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results",
))


def new_york_stamp():
    """Return New York wall time without requiring zoneinfo on Python 3.8."""
    if hasattr(time, "tzset"):
        previous = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            time.tzset()
            return datetime.now().strftime("%y%m%d-%H%M")
        finally:
            if previous is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous
            time.tzset()
    return datetime.utcnow().strftime("%y%m%d-%H%M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("experiment_name"); parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(); stamp = new_york_stamp()
    folder = args.results_root / (stamp + "-" + args.experiment_name)
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    standard_ignore = shutil.ignore_patterns(".git", ".venv*", "outputs", "Results", "cache", "tmp", "__pycache__", "*.pyc", "Data", "_smoke*")

    def snapshot_ignore(directory, names):
        ignored = set(standard_ignore(directory, names))
        # Usually results is outside the repository.  If a caller chooses a
        # repository-local root (for a smoke test, for example), never copy
        # the just-created experiment back into its own source snapshot.
        parent = Path(directory)
        for name in names:
            candidate = parent / name
            if folder.is_relative_to(candidate):
                ignored.add(name)
        return ignored

    shutil.copytree(ROOT, folder / "source", ignore=snapshot_ignore)
    for name in ("configs", "pretrain/checkpoints", "pretrain/logs", "pretrain/cache", "pretrain/tmp", "downstream"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    (folder / "manifest.json").write_text(json.dumps({"experiment": args.experiment_name, "created_at_new_york": stamp, "source_commit": commit, "publication_root": str(ROOT / "outputs/results"), "status": "prepared"}, indent=2) + "\n", encoding="utf-8")
    print(folder)
