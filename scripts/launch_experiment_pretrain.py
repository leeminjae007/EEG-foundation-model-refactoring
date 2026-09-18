"""Create an isolated experiment and immediately submit its A100 pretrain."""
import argparse
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("--config", default="configs/pretrain_gr2_geometry.yaml")
parser.add_argument(
    "--preset",
    choices=("gr2-mjde-d4-geometry", "gr2-d2-static", "gr2-d2-patch-scalar", "gr2-d2-patch-dimension"),
    help="Named, reproducible GR2 experiment profile. If omitted, a matching experiment name selects it.",
)
parser.add_argument("--results-root")
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()


def require_training_environment():
    missing = []
    for module in ("torch", "yaml", "numpy", "scipy", "mne", "lmdb", "sklearn", "h5py", "pandas"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Training environment is not active (missing: %s). Run `conda env create -f environment.yml` "
            "once, then `conda activate eeg-foundation-model-cu118`." % ", ".join(missing)
        )
    import torch
    if torch.__version__ != "2.0.1":
        raise SystemExit(
            "Expected torch 2.0.1, found %s. Activate eeg-foundation-model-cu118 before submitting." % torch.__version__
        )


if not args.prepare_only:
    require_training_environment()

# A matching experiment name selects the named profile, while --preset lets a
# user choose a more descriptive folder name without changing its settings.
PRESETS = {
    "gr2-mjde-d4-geometry": "gr2-mjde-d4-geometry",
    "gr2-d2-static": "gr2-d2-static",
    "gr2-d2-patch-scalar": "gr2-d2-patch-scalar",
    "gr2-d2-patch-dimension": "gr2-d2-patch-dimension",
}
preset = args.preset or PRESETS.get(args.experiment_name)

create = [sys.executable, str(ROOT / "scripts/create_experiment_layout.py"), args.experiment_name]
if args.results_root:
    create += ["--results-root", args.results_root]
experiment = Path(subprocess.check_output(create, text=True).strip())
submit = [sys.executable, str(ROOT / "scripts/submit_experiment_pretrain.py"),
          "--experiment", str(experiment), "--config", args.config]
if preset:
    submit += ["--preset", preset]
if args.prepare_only:
    submit.append("--prepare-only")
pretrain = subprocess.check_output(submit, text=True).strip()

# Prepare downstream configs now.  For an actual launch, submit the complete
# array held: the CPU verifier is the only process allowed to release it.
downstream = [sys.executable, str(ROOT / "scripts/submit_experiment_downstream.py"),
              "--experiment", str(experiment), "--hold"]
if args.prepare_only:
    downstream.append("--prepare-only")
    subprocess.run(downstream, check=True)
    print(experiment)
    sys.exit(0)
downstream_job = subprocess.check_output(downstream, text=True).strip().splitlines()[-1].split(";")[0]

source = experiment / "source"
cluster = yaml.safe_load((source / "configs/cluster/bigpurple_a100.yaml").read_text())["slurm"]
logs = experiment / "monitor/logs"; logs.mkdir(parents=True, exist_ok=True)
monitor_command = shlex.join([
    sys.executable, str(source / "scripts/release_downstream_after_pretrain.py"),
    "--experiment", str(experiment), "--downstream-job", downstream_job,
])
monitor_job = subprocess.check_output([
    "sbatch", "--parsable", "--account=" + cluster["account"],
    "--job-name=" + experiment.name + "-release-ds", "--partition=cpu_long",
    "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=7-00:00:00",
    "--output=" + str(logs / "%j.out"), "--error=" + str(logs / "%j.err"),
    "--wrap=exec " + monitor_command,
], text=True).strip().split(";")[0]
manifest_path = experiment / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update({"pretrain_job": pretrain.splitlines()[-1].split(";")[0], "downstream_job": downstream_job,
                 "downstream_state": "held", "downstream_release_monitor_job": monitor_job})
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(pretrain)
print(downstream_job)
print(monitor_job)
print(experiment)
