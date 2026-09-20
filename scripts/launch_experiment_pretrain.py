"""Create an isolated experiment and immediately submit its A100 pretrain."""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("--config", default="configs/pretrain_gr2_geometry.yaml")
parser.add_argument(
    "--preset",
    choices=("gr2-mjde-d4-geometry", "gr2-d2-static", "gr2-d2-patch-scalar", "gr2-d2-patch-dimension", "gr2-d2-patch-dimension-mask60"),
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
    installed_version = torch.__version__.split("+", 1)[0]
    if installed_version != "2.0.1":
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
    "gr2-d2-patch-dimension-mask60": "gr2-d2-patch-dimension-mask60",
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

# Arrays wait in Slurm; no CPU polling or held-job release service.
downstream = [sys.executable, str(ROOT / "scripts/submit_experiment_downstream.py"),
              "--experiment", str(experiment)]
if args.prepare_only:
    subprocess.run(downstream + ["--prepare-only"], check=True)
    print(experiment)
    sys.exit(0)
pretrain_job = pretrain.splitlines()[-1].split(";")[0]
subprocess.run(downstream + ["--dependency", pretrain_job], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/finalize_experiment.py"),
                "--experiment", str(experiment), "--submit"], check=True)
print("pretrain=" + pretrain_job)
print(experiment)
