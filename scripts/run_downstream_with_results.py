"""Run one downstream seed, then publish its server-side CSV row on success."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--dataset", required=True); parser.add_argument("--seed", required=True, type=int); parser.add_argument("--resume"); parser.add_argument("--source", type=Path)
    args = parser.parse_args(); source = (args.source or args.experiment / "source").resolve(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); output = Path(config["runtime"]["output"])
    if not output.is_absolute() or args.experiment.resolve() not in output.resolve().parents:
        raise ValueError("downstream runtime.output must be an absolute path below the experiment folder")
    result = output / "result.json"
    if not result.exists():
        command = [sys.executable, str(source / "finetune.py"), "--config", str(args.config)] + (["--resume", args.resume] if args.resume else [])
        subprocess.run(command, cwd=source, check=True)
    # `last.pth` contains the post-finetune encoder state, including gates.
    # It is deliberately distinct from the validation-selected best checkpoint
    # used for test metrics in result.json.
    final_checkpoint = output / "last.pth"
    if final_checkpoint.is_file():
        subprocess.run([
            sys.executable, str(source / "scripts/visualize_fusion_gates.py"),
            "--checkpoint", str(final_checkpoint),
            "--output-dir", str(output / "gate_final"),
            "--prefix", "final_fusion_gate",
        ], cwd=source, check=True)
    subprocess.run([sys.executable, str(source / "scripts/experiment_results.py"), "--experiment", str(args.experiment), "--dataset", args.dataset, "--seed", str(args.seed), "--result", str(result)], check=True)
