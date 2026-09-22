"""Submit five TUAB seeds at LR 1e-4, WD 5e-5, dropout 0.3 on A100."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import digest, write_json

BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1348-gr2-d2-mask55-tuab-a100')
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
SEEDS = (42, 696, 1001, 1234, 3407)
HP = dict(learning_rate=1e-4, weight_decay=5e-5, dropout=0.3)
EARLY_STOPPING = dict(monitor='balanced_accuracy', min_epochs=4, patience=10, min_delta=0.0)
BAD_NODES = ('a100-4011', 'a100-4018', 'a100-4024', 'a100-4028', 'a100-4034', 'a100-4045')


def prepare():
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M%S')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-tuab-lr1e4-wd5e5-drop03-five-a100'
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if (not verified.get('strict_load') or verified.get('partial_epoch_smoke') or
            not checkpoint.is_file() or digest(checkpoint) != verified['sha256']):
        raise ValueError('Verified mask55 pretrain checkpoint failed strict verification')
    configs = []
    for seed in SEEDS:
        original = BASE / f'configs/downstream/tuab_seed{seed}.yaml'
        config = yaml.safe_load(original.read_text())
        if config['seed'] != seed or config['data']['dataset'] != 'tuab':
            raise ValueError(f'Unexpected frozen TUAB template: {original}')
        if not Path(config['data']['dataset_dir']).is_dir():
            raise FileNotFoundError(config['data']['dataset_dir'])
        if any('warmup' in str(key).lower() for key in config['optimization']):
            raise ValueError('Downstream warmup is forbidden')
        configs.append(config)
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT, campaign / 'source', ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    write_json(campaign / 'pretrain/verified.json', verified)
    entries = []
    for index, (seed, config) in enumerate(zip(SEEDS, configs)):
        opt = config['optimization']
        for key in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
            opt[key] = HP['learning_rate']
        opt.update(weight_decay=HP['weight_decay'], batch_size_per_gpu=64,
                   gradient_accumulation_steps=1, epochs=20,
                   early_stopping=dict(EARLY_STOPPING))
        config['model'].update(head_dropout=HP['dropout'], checkpoint=str(checkpoint))
        config['data']['num_workers'] = 2
        output = campaign / 'runs' / f'seed-{seed}'
        config['runtime']['output'] = str(output)
        path = campaign / 'configs' / f'tuab_seed{seed}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        entries.append(dict(index=index, dataset='tuab', seed=seed, config=str(path),
                            config_sha256=digest(path), output=str(output)))
    write_json(campaign / 'downstream_entries.json', entries)
    write_json(campaign / 'manifest.json', dict(
        status='prepared', created_at_new_york=stamp, base_experiment=str(BASE),
        checkpoint=str(checkpoint), checkpoint_sha256=verified['sha256'],
        python=sys.executable, seeds=list(SEEDS), fixed_hp=HP,
        batch_size_per_gpu=64, gradient_accumulation_steps=1, epochs=20,
        early_stopping=EARLY_STOPPING,
        selection='Per-seed best validation balanced accuracy; test reporting only',
        resources=dict(gpu='a100', partitions='a100_short,a100_long',
                       time='24:00:00', concurrency=5, excluded_nodes=list(BAD_NODES)),
        jobs=[]))
    return campaign


def submit(campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['jobs']:
        raise ValueError('Already submitted')
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    resources = manifest['resources']
    command = ['sbatch', '--parsable', '--account=system',
               '--job-name=mask55-tuab-five-fixed',
               '--partition=' + resources['partitions'], '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G',
               '--time=' + resources['time'], '--array=0-4%5',
               '--exclude=' + ','.join(resources['excluded_nodes']),
               '--chdir=' + str(campaign / 'source'),
               '--output=' + str(logs / 'tuab-%A_%a.out'),
               '--error=' + str(logs / 'tuab-%A_%a.err'),
               '--wrap=exec ' + shlex.join([
                   'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                   '--kill-on-bad-exit=1', manifest['python'],
                   str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                   '--experiment', str(campaign), '--source', str(campaign / 'source')])]
    job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    manifest['jobs'].append(dict(job=job, indices=list(range(5)), command=command))
    manifest['status'] = 'submitted'
    write_json(manifest_path, manifest)
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=system', '--job-name=mask55-tuab-five-results',
        '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--mem=4G', '--time=00:30:00', '--dependency=afterany:' + job,
        '--output=' + str(logs / 'aggregate-%j.out'),
        '--error=' + str(logs / 'aggregate-%j.err'),
        '--wrap=exec ' + shlex.join([manifest['python'],
                                    str(campaign / 'source/scripts/submit_mask55_tuab_fixed_five.py'),
                                    'aggregate', '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest['finalizer_job'] = finalizer
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), job=job, finalizer=finalizer), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    rows, missing = [], []
    for entry in entries:
        try:
            if digest(entry['config']) != entry['config_sha256']:
                raise ValueError('Frozen config changed')
            output = Path(entry['output'])
            result = json.loads((output / 'result.json').read_text())
            selected = result['balanced_accuracy']
            score = float(selected['selection']['score'])
            epoch = int(selected['selection']['epoch'])
            test = {metric: float(selected['test'][metric])
                    for metric in ('balanced_accuracy', 'auroc', 'auprc')}
            if (not math.isfinite(score) or not 1 <= epoch <= 20 or
                    not all(math.isfinite(value) for value in test.values()) or
                    not (output / 'best-balanced_accuracy.pth').is_file() or
                    not (output / 'last.pth').is_file()):
                raise ValueError('Incomplete or non-finite validation-selected result')
            rows.append(dict(seed=entry['seed'], selected_epoch=epoch,
                             validation_bacc=score, test=test))
        except Exception as exc:
            missing.append(dict(seed=entry['seed'], error=repr(exc)))
    complete = len(rows) == 5 and {row['seed'] for row in rows} == set(SEEDS)
    summary = dict(status='complete' if complete else 'incomplete', rows=rows, missing=missing,
                   test_used_for_selection=False)
    if complete:
        summary['test_mean_sd'] = {
            metric: dict(mean=statistics.mean(row['test'][metric] for row in rows),
                         sd=statistics.pstdev(row['test'][metric] for row in rows))
            for metric in ('balanced_accuracy', 'auroc', 'auprc')}
    write_json(campaign / 'summary.json', summary)
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = summary['status']
    write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'aggregate'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'launch':
        submit(prepare())
    elif args.campaign:
        return aggregate(args.campaign)
    else:
        parser.error('aggregate requires --campaign')
    return 0


if __name__ == '__main__':
    sys.exit(main())
