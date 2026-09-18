"""Poll the pretrain-only controller using a local CPU process."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    local = ROOT / "outputs/results/a100-pretrain-priority-20260916"
    local.mkdir(parents=True, exist_ok=True)
    (local / "monitor.pid").write_text(str(os.getpid()))
    remote = "cd " + shlex.quote(args.root) + " && " + shlex.join([
        args.root + "/.venv/bin/python", "-m", "ablation.pretrain_queue", "step", "--folder", args.folder])
    while not (local / "STOP").exists():
        try:
            errors = []
            status = None
            for host in (None, "10.189.18.57", "10.189.18.58", "10.189.18.56"):
                cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
                if host:
                    cmd += ["-o", "HostName=" + host, "-o", "HostKeyAlias=bigpurple.nyumc.org"]
                try:
                    result = subprocess.run(cmd + ["bigpurple.nyumc.org", remote], capture_output=True,
                        text=True, encoding="utf-8", timeout=180, check=True)
                    status = next(json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith("{"))
                    break
                except Exception as exc:
                    errors.append(repr(exc))
            if status is None:
                raise RuntimeError("; ".join(errors))
            tmp = local / "status.tmp"
            tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            tmp.replace(local / "status.json")
            print(json.dumps(status), flush=True)
            if status["complete"]:
                return
        except Exception as exc:
            (local / "monitor_error.json").write_text(json.dumps(dict(error=repr(exc))), encoding="utf-8")
            print(repr(exc), flush=True)
        if args.once:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
