"""Create an isolated experiment and immediately submit its A100 pretrain."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("--config", default="configs/pretrain_gr2_geometry.yaml")
parser.add_argument("--results-root")
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()

create = [sys.executable, str(ROOT / "scripts/create_experiment_layout.py"), args.experiment_name]
if args.results_root:
    create += ["--results-root", args.results_root]
experiment = Path(subprocess.check_output(create, text=True).strip())
submit = [sys.executable, str(ROOT / "scripts/submit_experiment_pretrain.py"),
          "--experiment", str(experiment), "--config", args.config]
if args.prepare_only:
    submit.append("--prepare-only")
subprocess.run(submit, check=True)
print(experiment)
