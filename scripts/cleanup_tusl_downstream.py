"""Remove historical TUSL downstream artifacts, never pretraining or raw data.

Owner-only, exact reviewed campaign allowlist.  Dry-run unless --apply is given.
"""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path("/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results")
CAMPAIGNS = (
    "260920-2314-gr2-d4-patch-dimension-mask60",
    "260919-0301-gr2-d2-static",
    "enc-s2t-6stage",
    "260918-2033-gr2-d2-patch-dimension",
    "enc-lite",
    "enc-t2s-6stage",
    "260918-1759-gr2-d2-patch-scalar",
    "260920-0342-gr2-d2-patch-dimension-mask55",
    "260918-2228-gr2-mjde-d4-geometry",
    "260919-2214-gr2-d2-patch-dimension-mask60",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and getpass.getuser() != "ml10266":
        parser.error("Only ml10266 may remove owner-hosted historical results")
    active = subprocess.check_output(
        ["squeue", "-h", "-u", "ml10266,hk4935,yc8820", "-o", "%j"],
        text=True).lower().splitlines()
    if any("tusl" in job for job in active):
        raise RuntimeError("A TUSL job is still queued or running")
    for name in CAMPAIGNS:
        campaign = ROOT / name
        target = campaign / "downstream/tusl"
        if (target.is_symlink() or target.name != "tusl"
                or target.resolve().parent != (campaign / "downstream").resolve()
                or campaign.resolve().parent != ROOT.resolve()):
            raise ValueError(f"Unsafe cleanup target: {target}")
        if target.is_dir():
            print(("REMOVE " if args.apply else "WOULD REMOVE ") + str(target))
            if args.apply:
                shutil.rmtree(target)
        for config in (campaign / "configs/downstream").glob("tusl_seed*.yaml"):
            if config.is_symlink() or config.resolve().parent != (campaign / "configs/downstream").resolve():
                raise ValueError(f"Unsafe config target: {config}")
            print(("REMOVE " if args.apply else "WOULD REMOVE ") + str(config))
            if args.apply:
                config.unlink()
        manifest_path = campaign / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            jobs = {str(entry["job"]) for entry in manifest.get("downstream_jobs", [])
                    if entry.get("dataset") == "tusl"}
            logs = campaign / "downstream/logs"
            if logs.is_dir():
                for log in logs.iterdir():
                    if (log.is_file() and not log.is_symlink() and
                            any(re.search(r"(?<!\d)" + re.escape(job) + r"(?!\d)", log.name)
                                for job in jobs)):
                        if log.resolve().parent != logs.resolve():
                            raise ValueError(f"Unsafe log target: {log}")
                        print(("REMOVE " if args.apply else "WOULD REMOVE ") + str(log))
                        if args.apply:
                            log.unlink()
        if args.apply:
            marker = campaign / "tusl-removal.json"
            marker.write_text(json.dumps(dict(dataset="tusl", removed_raw_downstream=True,
                                              reason="User excluded TUSL from downstream"),
                                         indent=2) + "\n")


if __name__ == "__main__":
    main()
