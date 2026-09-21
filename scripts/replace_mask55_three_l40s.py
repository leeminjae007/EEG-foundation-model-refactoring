"""Prepare/submit validated L40S replacement for the three mask55 datasets.

No GPU smoke jobs are scheduled. `prepare` checks every LMDB record and writes
frozen configs; `submit` must be called only after the old A100 jobs are gone.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import digest, write_json
from scripts.submit_mask55_missing_three import BASE, DATA, DATASETS, RESULTS, SEEDS, config_for, train_counts
from src.data.datasets.processed_dataset import _loads
from src.data.datasets.registry import get_dataset_spec

OLD_JOBS = (27670907, 27670908, 27670909, 27670910,
            27670911, 27670912, 27670913)
PATHS = {
    'mumtaz': DATA / 'mumtaz',
    'bcic2020_3': DATA / 'bcic2020-3/processed',
    'bciciv2a': DATA / 'bcic-iv-2a/processed_inde_avg_filter',
}
EXPECTED_SPLITS = {
    'mumtaz': (4891, 1041, 1211),
    'bcic2020_3': (4500, 750, 750),
}


def audit_all_records():
    audit = {}
    for dataset, path in PATHS.items():
        spec = get_dataset_spec(dataset)
        per_split, all_keys = {}, set()
        for position, split in enumerate(('train', 'val', 'test')):
            reader = spec.dataset_class(path, split)
            database = reader._database()
            with database.begin() as transaction:
                keys = reader.keys
                if len(keys) != len(reader) or len(keys) != len(set(keys)):
                    raise ValueError(f'{dataset}/{split}: inconsistent or duplicate keys')
                if all_keys.intersection(keys):
                    raise ValueError(f'{dataset}/{split}: key appears in another split')
                all_keys.update(keys)
                if dataset in EXPECTED_SPLITS and len(keys) != EXPECTED_SPLITS[dataset][position]:
                    raise ValueError(f'{dataset}/{split}: unexpected split size {len(keys)}')
                if not bool(reader.channel_validity.all()):
                    raise ValueError(f'{dataset}/{split}: inactive channel')
                counts = Counter()
                for key in keys:
                    record = _loads(transaction.get(key.encode()))
                    signal = np.asarray(record['sample'])
                    if tuple(signal.shape) != tuple(reader.stored_shape):
                        raise ValueError(f'{dataset}/{split}/{key}: shape {signal.shape}')
                    if not np.isfinite(signal).all():
                        raise ValueError(f'{dataset}/{split}/{key}: non-finite signal')
                    label = int(record['label'])
                    if label < 0 or label >= reader.num_outputs:
                        raise ValueError(f'{dataset}/{split}/{key}: label {label}')
                    counts[label] += 1
                if set(counts) != set(range(reader.num_outputs)):
                    raise ValueError(f'{dataset}/{split}: class missing: {counts}')
                per_split[split] = dict(samples=len(keys), labels=dict(sorted(counts.items())),
                                        shape=list(reader.stored_shape),
                                        first_key=keys[0], last_key=keys[-1])
            reader.database.close()
        audit[dataset] = per_split
    return audit


def prepare():
    audit = audit_all_records()
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if not checkpoint.is_file() or digest(checkpoint) != verified['sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    if not verified['strict_load'] or verified['partial_epoch_smoke'] or verified['epoch'] != 40:
        raise ValueError('Pretrain is not a strict completed epoch-40 checkpoint')
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-three-l40s-no-smoke'
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    (campaign / 'pretrain').mkdir()
    write_json(campaign / 'pretrain/verified.json', verified)
    write_json(campaign / 'data_audit.json', audit)
    counts = train_counts(DATA / 'mumtaz')
    entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            output = campaign / 'downstream' / dataset / f'seed-{seed}'
            config = config_for(dataset, seed, source, checkpoint, output, counts)
            if config['optimization']['epochs'] != 50 or config['seed'] != seed:
                raise ValueError(f'{dataset}/{seed}: incorrect epoch/seed')
            if any('warmup' in str(k).lower() for k in config['optimization']):
                raise ValueError('Unexpected warmup configuration')
            path = campaign / 'configs/downstream' / f'{dataset}_seed{seed}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            entries.append(dict(dataset=dataset, seed=seed, config=str(path),
                                config_sha256=digest(path), output=str(output)))
    write_json(campaign / 'downstream_entries.json', entries)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    manifest = dict(experiment=campaign.name, preset='gr2-d2-patch-dimension-mask55',
                    created_at_new_york=stamp, source_commit=commit,
                    checkpoint=str(checkpoint), checkpoint_sha256=verified['sha256'],
                    gpu='l40s', smoke=False, datasets=DATASETS, seeds=SEEDS,
                    replaced_a100_jobs=OLD_JOBS,
                    selection='Per-seed validation balanced accuracy; test reporting only',
                    audit='Every LMDB sample: shape, finiteness, label, split keys; strict checkpoint SHA256',
                    downstream_jobs=[], status='prepared')
    write_json(campaign / 'manifest.json', manifest)
    print(json.dumps(dict(campaign=str(campaign), audit=audit, entries=len(entries)), indent=2))


def submit(campaign):
    campaign = campaign.resolve()
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'prepared' or manifest['downstream_jobs']:
        raise ValueError('Campaign was already submitted')
    old_active = subprocess.check_output(['squeue', '-h', '-j', ','.join(map(str, OLD_JOBS)),
                                          '-o', '%i'], text=True).strip()
    if old_active:
        raise RuntimeError('Old A100 jobs still active: ' + old_active)
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    source = campaign / 'source'
    verified = json.loads((campaign / 'pretrain/verified.json').read_text())
    if digest(verified['checkpoint']) != verified['sha256']:
        raise ValueError('Checkpoint changed before submission')
    for entry in entries:
        if digest(entry['config']) != entry['config_sha256']:
            raise ValueError('Frozen downstream config changed: ' + entry['config'])
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    for dataset in DATASETS:
        indices = [i for i, entry in enumerate(entries) if entry['dataset'] == dataset]
        wrap = ['srun', '--ntasks=1', '--gpus-per-task=l40s:1', '--gpu-bind=single:1',
                '--kill-on-bad-exit=1', sys.executable,
                str(source / 'scripts/downstream_experiment_worker.py'),
                '--experiment', str(campaign), '--source', str(source)]
        command = ['sbatch', '--parsable', '--account=system',
                   '--job-name=mask55-three-l40s-' + dataset,
                   '--partition=gl40s_dev,gl40s_short,gl40s_long', '--nodes=1', '--ntasks=1',
                   '--gpus-per-task=l40s:1', '--cpus-per-task=2', '--mem=32G',
                   '--time=04:00:00', '--array=' + ','.join(map(str, indices)) + '%5',
                   '--chdir=' + str(source),
                   '--output=' + str(logs / (dataset + '-%A_%a.out')),
                   '--error=' + str(logs / (dataset + '-%A_%a.err')),
                   '--wrap=exec ' + shlex.join(wrap)]
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        manifest['downstream_jobs'].append(dict(dataset=dataset, job=job, indices=indices,
                                                gpu='l40s', time='04:00:00',
                                                partitions='gl40s_dev,gl40s_short,gl40s_long'))
        write_json(manifest_path, manifest)
    jobs = [item['job'] for item in manifest['downstream_jobs']]
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=system', '--job-name=mask55-three-l40s-results',
        '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--mem=4G', '--time=00:30:00', '--dependency=afterany:' + ':'.join(jobs),
        '--output=' + str(logs / 'results-%j.out'), '--error=' + str(logs / 'results-%j.err'),
        '--wrap=exec ' + shlex.join([sys.executable,
            str(source / 'scripts/finalize_mask55_missing_three.py'), '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest.update(finalizer_job=finalizer, status='submitted')
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), downstream_jobs=manifest['downstream_jobs'],
                          finalizer_job=finalizer), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'submit'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if args.campaign is None:
            parser.error('--campaign is required for submit')
        submit(args.campaign)
