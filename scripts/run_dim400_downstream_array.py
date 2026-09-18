"""Run one validation-selected D=400 downstream seed from the fixed manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/dim400_downstream_20260917"


def main():
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    entries = json.loads((CAMPAIGN / "array.json").read_text(encoding="utf-8"))
    entry = entries[index]
    config = Path(entry["config"])
    if hashlib.sha256(config.read_bytes()).hexdigest() != entry["config_sha256"]:
        raise ValueError("Downstream config changed after submission")
    verification = json.loads((CAMPAIGN / "pretrain_verified.json").read_text(encoding="utf-8"))
    if not verification.get("verified"):
        raise ValueError("D=400 pretraining checkpoint not verified")
    output = Path(entry["output"])
    if (output / "result.json").is_file():
        print(f"already complete: {entry['slug']} seed={entry['seed']}", flush=True)
        return
    command = [sys.executable, str(ROOT / "finetune.py"), "--config", str(config)]
    last = output / "last.pth"
    if last.is_file():
        command += ["--resume", str(last)]
    print(json.dumps({"index": index, "dataset": entry["slug"], "seed": entry["seed"],
                      "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
