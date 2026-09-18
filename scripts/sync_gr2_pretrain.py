"""Read-only hourly sync of one pretrain campaign; never submits Slurm jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sync(remote_folder, local_folder):
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "HostName=10.189.18.57",
               "-o", "HostKeyAlias=bigpurple.nyumc.org"]
    root = "/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model"
    code = "from pathlib import Path\nimport json\nf=Path(" + repr(remote_folder) + ")\n"
    code += "m=json.loads((f/'manifest.json').read_text());r=Path(m['results_dir'])\nfiles={}\n"
    code += "for name in ['manifest.json','status.json','config.yaml','controller_error.json','validation/result.json']:\n p=f/name\n if p.is_file():files[name]=p.read_text()\n"
    code += "for name in ['pretrain_result.json','PRETRAIN.md']:\n p=r/name\n if p.is_file():files[name]=p.read_text()\n"
    code += "print(json.dumps({'files':files}))\n"
    result = subprocess.run(["ssh", *options, "bigpurple.nyumc.org", root + "/.venv/bin/python", "-"],
                            input=code, capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith('{"files":'))
    files = json.loads(line)["files"]
    allowed = {"manifest.json", "status.json", "config.yaml", "controller_error.json", "validation/result.json", "pretrain_result.json", "PRETRAIN.md"}
    for name, content in files.items():
        if name not in allowed:
            raise ValueError("Unexpected remote result name")
        path = local_folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    state = json.loads(files.get("status.json", "{}"))
    complete = state.get("complete", False) and "pretrain_result.json" in files
    if complete:
        report = json.loads(files["pretrain_result.json"])
        target = local_folder / "checkpoint-epoch-0040.pth"
        def digest(path):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        if not target.exists() or digest(target) != report["checkpoint_sha256"]:
            temporary = target.with_suffix(".download")
            subprocess.run(["scp", *options, "bigpurple.nyumc.org:" + report["checkpoint"], str(temporary)],
                           capture_output=True, timeout=600, check=True)
            if digest(temporary) != report["checkpoint_sha256"]:
                raise ValueError("Downloaded checkpoint hash differs")
            temporary.replace(target)
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), complete=complete, remote_state=state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-folder", required=True)
    parser.add_argument("--local-folder", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.local_folder.mkdir(parents=True, exist_ok=True)
    while not (args.local_folder / "STOP_SYNC").exists():
        try:
            status = sync(args.remote_folder, args.local_folder)
        except Exception as exc:
            status = dict(checked_utc=datetime.now(timezone.utc).isoformat(), error=repr(exc), complete=False)
        (args.local_folder / "sync_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)
        if args.once or status["complete"]:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
