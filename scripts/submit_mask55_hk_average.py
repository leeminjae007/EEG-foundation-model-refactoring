"""One-command hk4935 handoff for mask55 enc-average-3s downstream.

If the owner pretrain is still running, register a CPU-only callback after its
successful completion. The callback submits TUAB first, then the ten remaining
five-seed datasets on GL40S using the frozen final-default settings.
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
ORIGINAL = RESULTS / '260923-0033-mask55-d2-patchdim-owner-encoder-four-pretrain/enc-average-3s'
HANDOFF = RESULTS / '260921-1456-mask55-d2-patchdim-shared-pretrain/accounts/hk4935/enc-average-3s-handoff'
RECORD = HANDOFF / 'callback.json'


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def live(job: str) -> bool:
    query = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                           capture_output=True, text=True)
    if query.returncode:
        if 'Invalid job id specified' in query.stderr:
            return False
        raise RuntimeError(f'squeue {job}: {query.stderr.strip()}')
    return bool(query.stdout.strip())


def submit_now() -> None:
    if not (ORIGINAL / 'pretrain/verified.json').is_file():
        raise RuntimeError('Owner enc-average-3s pretrain is not strictly verified yet')
    scripts = [
        ['submit_mask55_tuab_comparison.py', '--account', 'hk4935', '--arm', 'enc-average-3s',
         '--lr', '0.0005', '--wd', '0.05', '--dropout', '0.3'],
        ['submit_mask55_final_comparison.py', '--account', 'hk4935', '--arm', 'enc-average-3s'],
    ]
    for script, *args in scripts:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args],
                       cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit-now', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if getpass.getuser() != 'hk4935':
        parser.error('This handoff must be started by hk4935')
    if args.submit_now or (ORIGINAL / 'pretrain/verified.json').is_file():
        submit_now()
        return
    manifest = load(ORIGINAL / 'manifest.json')
    pretrain_job = str(manifest['pretrain_job'])
    if not re.fullmatch(r'\d+', pretrain_job) or not live(pretrain_job):
        raise RuntimeError(f'Pretrain is not verified and job {pretrain_job} is not active')
    HANDOFF.mkdir(parents=True, exist_ok=True)
    if RECORD.is_file():
        callback = load(RECORD)
        if callback.get('pretrain_job') != pretrain_job:
            raise ValueError('Existing handoff references a different pretrain job')
        print(f"Already registered callback {callback['job']} for pretrain {pretrain_job}")
        return
    logs = HANDOFF / 'logs'
    logs.mkdir(exist_ok=True)
    wrap = shlex.join([sys.executable, str(ROOT / 'scripts/submit_mask55_hk_average.py'),
                       '--submit-now'])
    command = [
        'sbatch', '--parsable', '--account=system',
        '--job-name=mask55-hk-average-handoff', '--partition=cpu_short,cpu_long',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=8G',
        '--time=00:30:00', '--dependency=afterok:' + pretrain_job,
        '--kill-on-invalid-dep=yes', '--chdir=' + str(ROOT),
        '--output=' + str(logs / '%j.out'), '--error=' + str(logs / '%j.err'),
        '--wrap=' + wrap,
    ]
    job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    if not re.fullmatch(r'\d+', job):
        raise RuntimeError(f'Unexpected callback job ID: {job}')
    RECORD.write_text(json.dumps({'job': job, 'pretrain_job': pretrain_job,
                                  'arm': 'enc-average-3s', 'owner': 'hk4935'}, indent=2) + '\n')
    print(f'Pretrain {pretrain_job} running; hk4935 callback {job} waits for afterok. '
          'It will submit TUAB first, then the ten remaining five-seed GL40S datasets.')


if __name__ == '__main__':
    main()
