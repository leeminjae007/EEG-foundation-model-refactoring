"""Move wholly unstarted clinical HP arrays to L40S with early stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import submit_mask55_clinical_hp_grid as original
from scripts.submit_mask55_hp_grid import digest, write_json

RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
OLD_CAMPAIGN = RESULTS / '260921-1423-gr2-d2-mask55-isruc-hmc-siena-grid'
OLD_JOBS = {'isruc': 27665212, 'hmc': 27665213, 'siena': 27665214}
OLD_FINALIZER = 27665215
EARLY_STOPPING = dict(monitor='balanced_accuracy', patience=10,
                      min_epochs=15, min_delta=0.0)


def require_no_started_tasks():
    for dataset, job in OLD_JOBS.items():
        output = subprocess.check_output([
            'sacct', '-j', str(job), '--format=JobID,State', '-n', '-P'
        ], text=True)
        for line in output.splitlines():
            match = re.fullmatch(rf'{job}_(\d+)\|([^|]+)', line.strip())
            if match and match.group(2) != 'PENDING':
                raise RuntimeError(f'{dataset} must remain on A100: {line}')
    for dataset in OLD_JOBS:
        if any((OLD_CAMPAIGN / 'runs' / dataset).rglob('result.json')):
            raise RuntimeError(f'{dataset} already has result files; retain original array')


def prepare():
    require_no_started_tasks()
    output_root = RESULTS / 'l40s-earlystop-replacements'
    campaign = original.prepare(output_root)
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    if len(entries) != 540:
        raise ValueError('Expected 540 grid runs')
    for entry in entries:
        cfg = yaml.safe_load(Path(entry['config']).read_text())
        if cfg['optimization'].get('early_stopping') != EARLY_STOPPING:
            raise ValueError('HP early stopping missing from frozen config')
        if digest(entry['config']) != entry['config_sha256']:
            raise ValueError('Config checksum mismatch')
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(replacement_of=str(OLD_CAMPAIGN), replaced_jobs=OLD_JOBS,
                    early_stopping=EARLY_STOPPING,
                    test_policy='Test reporting only from validation-BAcc checkpoint',
                    resources=dict(gpu='l40s', partitions='gl40s_short,gl40s_long',
                                   cpus=2, memory='32G', concurrency_per_dataset=10,
                                   time_limits={'isruc': '08:00:00', 'hmc': '08:00:00',
                                                'siena': '04:00:00'}))
    write_json(manifest_path, manifest)
    print(campaign)


def submit(campaign):
    campaign = campaign.resolve()
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'prepared' or manifest['jobs']:
        raise ValueError('Replacement campaign already submitted')
    require_no_started_tasks()
    # All old tasks are still pending. Cancel only these three arrays and their
    # dependent finalizer; preserve the old campaign as an audit trail.
    subprocess.run(['scancel', *map(str, OLD_JOBS.values()), str(OLD_FINALIZER)], check=True)
    if subprocess.check_output(['squeue', '-h', '-j',
                                ','.join(map(str, (*OLD_JOBS.values(), OLD_FINALIZER))),
                                '-o', '%i'], text=True).strip():
        raise RuntimeError('Old clinical grid jobs remain active after cancellation')
    old_manifest_path = OLD_CAMPAIGN / 'manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    old_manifest.update(status='superseded_unstarted', replacement_campaign=str(campaign))
    write_json(old_manifest_path, old_manifest)
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    for dataset in original.DATASETS:
        indices = [e['index'] for e in entries if e['dataset'] == dataset]
        wrap = ['srun', '--ntasks=1', '--gpus-per-task=l40s:1', '--gpu-bind=single:1',
                '--kill-on-bad-exit=1', sys.executable,
                str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                '--experiment', str(campaign), '--source', str(campaign / 'source')]
        command = ['sbatch', '--parsable', '--account=system',
                   '--job-name=mask55-clinical-es-' + dataset,
                   '--partition=gl40s_short,gl40s_long', '--nodes=1', '--ntasks=1',
                   '--gpus-per-task=l40s:1', '--cpus-per-task=2', '--mem=32G',
                   '--time=' + manifest['resources']['time_limits'][dataset],
                   '--array=' + ','.join(map(str, indices)) + '%10',
                   '--chdir=' + str(campaign / 'source'),
                   '--output=' + str(logs / (dataset + '-%A_%a.out')),
                   '--error=' + str(logs / (dataset + '-%A_%a.err')),
                   '--wrap=exec ' + shlex.join(wrap)]
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        manifest['jobs'].append(dict(dataset=dataset, job=job, indices=indices,
                                     gpu='l40s', early_stopping=EARLY_STOPPING))
        write_json(manifest_path, manifest)
    jobs = [item['job'] for item in manifest['jobs']]
    aggregate = subprocess.check_output([
        'sbatch', '--parsable', '--account=system',
        '--job-name=mask55-clinical-es-results', '--partition=cpu_short,cpu_long',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G', '--time=00:30:00',
        '--dependency=afterany:' + ':'.join(jobs),
        '--output=' + str(logs / 'aggregate-%j.out'),
        '--error=' + str(logs / 'aggregate-%j.err'),
        '--wrap=exec ' + shlex.join([sys.executable,
            str(campaign / 'source/scripts/submit_mask55_clinical_hp_grid.py'),
            'aggregate', '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest.update(aggregate_job=aggregate, status='submitted')
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), jobs=manifest['jobs'],
                          aggregate_job=aggregate), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'submit'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if args.campaign is None:
            parser.error('--campaign required for submit')
        submit(args.campaign)
