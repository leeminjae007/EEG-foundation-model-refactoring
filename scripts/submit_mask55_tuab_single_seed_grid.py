"""Submit ordered, one-seed TUAB mask-55 hyperparameter checks on A100."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import digest, write_json

BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1348-gr2-d2-mask55-tuab-a100')
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
SEED = 42
PYTHON = Path('/gpfs/data/oermannlab/users/ml10266/.conda/envs/eeg-foundation-model-cu118/bin/python')
BAD_NODES = ('a100-4011', 'a100-4018', 'a100-4024', 'a100-4028', 'a100-4034', 'a100-4045')
EARLY_STOPPING = dict(monitor='balanced_accuracy', min_epochs=4, patience=10, min_delta=0.0)

# Strict dispatch order: higher dropout first, then higher WD. At WD .05
# check the anchor LR before the larger-LR probes.
CANDIDATES = (
    (1e-4, .05, .3), (2e-4, .05, .3), (5e-4, .05, .3),
    (1e-4, .02, .3), (1e-4, .01, .3),
    (1e-4, .05, .2), (1e-4, .02, .2), (1e-4, .01, .2),
    (1e-4, .05, .1), (1e-4, .02, .1), (1e-4, .01, .1),
)


def prepare():
    if len(CANDIDATES) != 11 or len(set(CANDIDATES)) != 11:
        raise ValueError('Expected eleven distinct ordered TUAB candidates')
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if (not verified.get('strict_load') or verified.get('partial_epoch_smoke')
            or not checkpoint.is_file() or digest(checkpoint) != verified['sha256']):
        raise ValueError('TUAB mask-55 pretrain did not pass strict verification')
    original = yaml.safe_load((BASE / f'configs/downstream/tuab_seed{SEED}.yaml').read_text())
    if original['seed'] != SEED or original['data']['dataset'] != 'tuab':
        raise ValueError('Wrong frozen TUAB seed/config')
    if not Path(original['data']['dataset_dir']).is_dir():
        raise FileNotFoundError(original['data']['dataset_dir'])
    if any('warmup' in str(key).lower() for key in original['optimization']):
        raise ValueError('Downstream warmup is forbidden')
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)

    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M%S')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-tuab-single-seed42-b64x8-a100'
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT, campaign / 'source', ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth',
        '.pytest-ascii-temp'))
    write_json(campaign / 'pretrain/verified.json', verified)
    entries = []
    for index, (lr, wd, dropout) in enumerate(CANDIDATES):
        config = json.loads(json.dumps(original))
        opt = config['optimization']
        for key in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
            opt[key] = lr
        opt.update(weight_decay=wd, batch_size_per_gpu=64,
                   gradient_accumulation_steps=8, epochs=20,
                   early_stopping=dict(EARLY_STOPPING))
        config['model'].update(head_dropout=dropout, checkpoint=str(checkpoint))
        config['data']['num_workers'] = 2
        output = campaign / 'runs' / f'{index:02d}-seed-{SEED}'
        config['runtime']['output'] = str(output)
        config_path = campaign / 'configs' / f'{index:02d}-seed{SEED}.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        entries.append(dict(index=index, dataset='tuab', seed=SEED,
                            hp=dict(learning_rate=lr, weight_decay=wd, dropout=dropout),
                            config=str(config_path), config_sha256=digest(config_path),
                            output=str(output)))
    write_json(campaign / 'downstream_entries.json', entries)
    write_json(campaign / 'manifest.json', dict(
        status='prepared', created_at_new_york=stamp, base_experiment=str(BASE),
        checkpoint=str(checkpoint), checkpoint_sha256=verified['sha256'],
        python=str(PYTHON), seed=SEED, candidates=len(entries),
        batch_size_per_gpu=64, gradient_accumulation_steps=8, effective_batch_size=512,
        epochs=20, early_stopping=EARLY_STOPPING,
        selection='Per-candidate validation-BAcc checkpoint; test is reporting-only',
        resources=dict(gpu='a100', partitions='a100_short,a100_long',
                       time='24:00:00', concurrency=1, excluded_nodes=list(BAD_NODES)),
        jobs=[], finalizer_job=None))
    return campaign


def submit(campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    jobs = manifest['jobs']
    if len(jobs) > len(entries) or manifest.get('finalizer_job'):
        raise ValueError('Campaign already finalized or manifest corrupted')
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    previous = jobs[-1]['job'] if jobs else None
    for entry in entries[len(jobs):]:
        index = entry['index']
        command = [
            'sbatch', '--parsable', '--account=system',
            '--job-name=mask55-tuab-seed42-grid',
            '--partition=' + manifest['resources']['partitions'],
            '--nodes=1', '--ntasks=1', '--gpus-per-task=a100:1',
            '--cpus-per-task=2', '--mem=32G', '--time=' + manifest['resources']['time'],
            '--array=' + str(index),
            '--exclude=' + ','.join(manifest['resources']['excluded_nodes']),
            '--chdir=' + str(campaign / 'source'),
            '--output=' + str(logs / 'tuab-%A_%a.out'),
            '--error=' + str(logs / 'tuab-%A_%a.err'),
        ]
        if previous:
            command.append('--dependency=afterany:' + previous)
        command.append('--wrap=exec ' + shlex.join([
            'srun', '--ntasks=1', '--gpus-per-task=a100:1',
            '--gpu-bind=single:1', '--kill-on-bad-exit=1', manifest['python'],
            str(campaign / 'source/scripts/downstream_experiment_worker.py'),
            '--experiment', str(campaign), '--source', str(campaign / 'source')]))
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        jobs.append(dict(index=index, job=job, dependency=previous, command=command))
        manifest.update(jobs=jobs, status='submitting')
        write_json(manifest_path, manifest)
        previous = job
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=system',
        '--job-name=mask55-tuab-seed42-grid-results',
        '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=1', '--mem=4G', '--time=00:30:00',
        '--dependency=afterany:' + previous,
        '--output=' + str(logs / 'aggregate-%j.out'),
        '--error=' + str(logs / 'aggregate-%j.err'),
        '--wrap=exec ' + shlex.join([
            manifest['python'],
            str(campaign / 'source/scripts/submit_mask55_tuab_single_seed_grid.py'),
            'aggregate', '--campaign', str(campaign)]),
    ], text=True).strip().split(';', 1)[0]
    manifest.update(finalizer_job=finalizer, status='submitted')
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign),
                          ordered_jobs=[item['job'] for item in jobs],
                          finalizer=finalizer), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    rows, missing = [], []
    for entry in entries:
        try:
            if digest(Path(entry['config'])) != entry['config_sha256']:
                raise ValueError('Frozen config checksum changed')
            output = Path(entry['output'])
            result = json.loads((output / 'result.json').read_text())['balanced_accuracy']
            score = float(result['selection']['score'])
            epoch = int(result['selection']['epoch'])
            test = {metric: float(result['test'][metric])
                    for metric in ('balanced_accuracy', 'auroc', 'auprc')}
            if (not math.isfinite(score) or not 1 <= epoch <= 20
                    or not all(math.isfinite(value) for value in test.values())
                    or not (output / 'best-balanced_accuracy.pth').is_file()):
                raise ValueError('Invalid validation-selected result/checkpoint')
            rows.append(dict(**entry, validation_bacc=score, best_epoch=epoch, test=test))
        except Exception as error:
            missing.append(dict(index=entry['index'], error=repr(error)))
    ranked = sorted(rows, key=lambda row: (-row['validation_bacc'], row['index']))
    complete = len(rows) == len(entries) == len(CANDIDATES)
    summary = dict(status='complete' if complete else 'incomplete',
                   expected=len(entries), readable=len(rows), missing=missing,
                   ranked_by_validation=ranked,
                   validation_winner=ranked[0] if complete else None,
                   test_used_for_selection=False)
    write_json(campaign / 'summary.json', summary)
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = summary['status']
    write_json(manifest_path, manifest)
    print(json.dumps(dict(status=summary['status'], readable=len(rows),
                          missing=missing), indent=2))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'launch', 'aggregate'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action in ('prepare', 'launch'):
        campaign = prepare()
        print(campaign)
        if args.action == 'launch':
            submit(campaign)
    elif args.campaign is None:
        parser.error('submit/aggregate require --campaign')
    elif args.action == 'submit':
        submit(args.campaign.resolve())
    else:
        return aggregate(args.campaign.resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())
