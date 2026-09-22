"""Skip results completed by the old TUSZ array before running a priority wave."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_shared_mask55_four import complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    entry = entries[index]
    if entry['index'] != index or entry['dataset'] != 'tusz':
        raise ValueError(f'Unexpected TUSZ frozen entry {index}')
    if complete(entry):
        print(f'SKIP completed TUSZ index {index}', flush=True)
        return
    subprocess.run([sys.executable, str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                    '--experiment', str(campaign), '--source', str(campaign / 'source')], check=True)


if __name__ == '__main__':
    main()
