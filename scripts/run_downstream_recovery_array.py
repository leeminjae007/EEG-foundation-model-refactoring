"""Resume selected downstream array entries from their last checkpoints."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


campaign = Path(__file__).resolve().parent.parent
entries = json.loads((campaign / "recovery_array.json").read_text())
entry = entries[int(os.environ["SLURM_ARRAY_TASK_ID"])]
config = Path(entry["config"])
if hashlib.sha256(config.read_bytes()).hexdigest() != entry["config_sha256"]:
    raise RuntimeError("Config hash mismatch")
output = Path(entry["result_dir"])
if (output / "result.json").exists():
    print("Already completed: " + str(output), flush=True)
    raise SystemExit(0)
checkpoint = output / "last.pth"
if not checkpoint.exists():
    raise FileNotFoundError("Recovery checkpoint missing: " + str(checkpoint))
command = [
    sys.executable, "-m", "torch.distributed.run",
    "--nnodes=1", "--nproc_per_node=1",
    "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
    "--rdzv_id=recovery-%s-%s" % (
        os.environ["SLURM_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"]),
    str(campaign / "source/finetune.py"),
    "--config", str(config), "--distributed", "--resume", str(checkpoint),
]
print(json.dumps({"entry": entry, "resume": str(checkpoint), "command": command}), flush=True)
subprocess.run(command, cwd=campaign / "source", check=True)
