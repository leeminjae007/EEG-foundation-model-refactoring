"""Five-seed A100 rerun of Mumtaz, BCIC2020-3, and BCIC-IV-2a.

Each dataset first runs a separate one-batch smoke job. Its production array
has an afterok dependency on that smoke job, so a failed adapter cannot launch
five production runs. Checkpoints are selected by validation BAcc only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
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
from src.data.datasets.processed_dataset import _loads

BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55')
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
DATA = Path('/gpfs/data/oermannlab/users/ml10266/Data/eeg_foundation_downstream')
SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ('mumtaz', 'bcic2020_3', 'bciciv2a')


def train_counts(path):
    import lmdb
    database = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with database.begin() as transaction:
        keys = _loads(transaction.get(b'__keys__'))['train']
        counts = Counter(int(_loads(transaction.get(key.encode()))['label']) for key in keys)
    database.close()
    if set(counts) != {0, 1}:
        raise ValueError(f'Unexpected Mumtaz labels: {counts}')
    return [counts[0], counts[1]]


def config_for(dataset, seed, source, checkpoint, output, mumtaz_counts):
    template = source / f'configs/downstream/gr9-1_bciciv2a_seed{seed}.yaml'
    config = yaml.safe_load(template.read_text())
    config['model']['checkpoint'] = str(checkpoint)
    config['runtime']['output'] = str(output)
    config['data']['num_workers'] = 2
    if dataset == 'mumtaz':
        config['data'].update(dataset=dataset, dataset_dir=str(DATA / 'mumtaz'))
        config['model']['head_dropout'] = 0.1
        config['optimization'].update(label_smoothing=0.0, class_counts=mumtaz_counts)
    elif dataset == 'bcic2020_3':
        config['data'].update(dataset=dataset, dataset_dir=str(DATA / 'bcic2020-3/processed'))
        config['model']['head_dropout'] = 0.1
        for name in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
            config['optimization'][name] = 5e-5
    elif dataset != 'bciciv2a':
        raise ValueError(dataset)
    return config


def submit_worker(campaign, source, dataset, indices, smoke, dependency, account, excluded):
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    experiment = campaign / 'smoke' if smoke else campaign
    worker = source / 'scripts/downstream_experiment_worker.py'
    wrap = ['srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
            '--kill-on-bad-exit=1', sys.executable, str(worker),
            '--experiment', str(experiment), '--source', str(source)]
    if smoke:
        wrap.extend(['--smoke-batches', '1'])
    command = ['sbatch', '--parsable', '--account=' + account,
               '--job-name=mask55-three-' + dataset + ('-smoke' if smoke else ''),
               '--partition=a100_dev,a100_short,a100_long', '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G',
               '--time=' + ('01:00:00' if smoke else '12:00:00'),
               '--array=' + ','.join(map(str, indices)) + ('%1' if smoke else '%5'),
               '--exclude=' + ','.join(excluded), '--chdir=' + str(source),
               '--output=' + str(logs / (dataset + ('-smoke' if smoke else '') + '-%A_%a.out')),
               '--error=' + str(logs / (dataset + ('-smoke' if smoke else '') + '-%A_%a.err')),
               '--wrap=exec ' + shlex.join(wrap)]
    if dependency:
        command.extend(['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes'])
    return subprocess.check_output(command, text=True).strip().split(';', 1)[0]


def main():
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if not checkpoint.is_file() or digest(checkpoint) != verified['sha256'] or not verified['strict_load']:
        raise ValueError('Pretrain checkpoint failed verification')
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-mumtaz-bcic3-bcic2a-a100'
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    (campaign / 'pretrain').mkdir()
    write_json(campaign / 'pretrain/verified.json', verified)
    (campaign / 'smoke/pretrain').mkdir(parents=True)
    write_json(campaign / 'smoke/pretrain/verified.json', verified)
    counts = train_counts(DATA / 'mumtaz')
    entries, smoke_entries = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            output = campaign / 'downstream' / dataset / f'seed-{seed}'
            config = config_for(dataset, seed, source, checkpoint, output, counts)
            path = campaign / 'configs/downstream' / f'{dataset}_seed{seed}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            entries.append(dict(dataset=dataset, seed=seed, config=str(path), output=str(output)))
            if seed == SEEDS[0]:
                smoke_output = campaign / 'smoke/downstream' / dataset / f'seed-{seed}'
                smoke_config = config_for(dataset, seed, source, checkpoint, smoke_output, counts)
                smoke_path = campaign / 'smoke/configs' / f'{dataset}.yaml'
                smoke_path.parent.mkdir(parents=True, exist_ok=True)
                smoke_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False))
                smoke_entries.append(dict(dataset=dataset, seed=seed, config=str(smoke_path), output=str(smoke_output)))
    write_json(campaign / 'downstream_entries.json', entries)
    write_json(campaign / 'smoke/downstream_entries.json', smoke_entries)
    cluster = yaml.safe_load((source / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    excluded = cluster['pretrain']['excluded_nodes']
    manifest = dict(experiment=campaign.name, preset='gr2-d2-patch-dimension-mask55',
                    created_at_new_york=stamp, checkpoint=str(checkpoint),
                    checkpoint_sha256=verified['sha256'], seeds=SEEDS, datasets=DATASETS,
                    mumtaz_train_class_counts=counts,
                    selection='Per-seed best validation balanced accuracy; test reporting only',
                    resources=dict(gpu='a100', partitions='a100_dev,a100_short,a100_long',
                                   time='12:00:00', excluded_nodes=excluded),
                    smoke_jobs=[], downstream_jobs=[], status='submitting')
    write_json(campaign / 'manifest.json', manifest)
    for index, dataset in enumerate(DATASETS):
        smoke_job = submit_worker(campaign, source, dataset, [index], True, None,
                                  cluster['account'], excluded)
        manifest['smoke_jobs'].append(dict(dataset=dataset, job=smoke_job))
        write_json(campaign / 'manifest.json', manifest)
        indices = list(range(index * 5, index * 5 + 5))
        job = submit_worker(campaign, source, dataset, indices, False, smoke_job,
                            cluster['account'], excluded)
        manifest['downstream_jobs'].append(dict(dataset=dataset, job=job, indices=indices,
                                                dependency=smoke_job))
        write_json(campaign / 'manifest.json', manifest)
    jobs = [entry['job'] for entry in manifest['downstream_jobs']]
    report = source / 'scripts/finalize_mask55_missing_three.py'
    logs = campaign / 'logs'
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=' + cluster['account'],
        '--job-name=mask55-three-results', '--partition=cpu_short,cpu_long',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G', '--time=00:30:00',
        '--dependency=afterany:' + ':'.join(jobs),
        '--output=' + str(logs / 'results-%j.out'), '--error=' + str(logs / 'results-%j.err'),
        '--wrap=exec ' + shlex.join([sys.executable, str(report), '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest.update(finalizer_job=finalizer, status='submitted')
    write_json(campaign / 'manifest.json', manifest)
    print(json.dumps(dict(campaign=str(campaign), smoke_jobs=manifest['smoke_jobs'],
                          downstream_jobs=manifest['downstream_jobs'], finalizer_job=finalizer), indent=2))


if __name__ == '__main__':
    main()
