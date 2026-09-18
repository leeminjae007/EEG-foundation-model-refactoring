"""Run one downstream seed, then publish its server-side CSV row on success."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--dataset", required=True); parser.add_argument("--seed", required=True, type=int); parser.add_argument("--resume")
    args = parser.parse_args(); source = args.experiment / "source"; config = yaml.safe_load(args.config.read_text(encoding="utf-8")); output = Path(config["runtime"]["output"])
    if not output.is_absolute() or args.experiment.resolve() not in output.resolve().parents:
        raise ValueError("downstream runtime.output must be an absolute path below the experiment folder")
    result = output / "result.json"
    if not result.exists():
        command = [sys.executable, str(source / "finetune.py"), "--config", str(args.config)] + (["--resume", args.resume] if args.resume else [])
        subprocess.run(command, cwd=source, check=True)
    subprocess.run([sys.executable, str(source / "scripts/experiment_results.py"), "--experiment", str(args.experiment), "--dataset", args.dataset, "--seed", str(args.seed), "--result", str(result)], check=True)
