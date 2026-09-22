"""Run TUAB first, then six fixed-HP mask-55 downstream datasets on A100."""
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

from submit_mask55_hp_grid import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55')
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ('tuab', 'tuev', 'tusz', 'siena', 'seedv', 'isruc', 'hmc')
HP = dict(learning_rate=1e-4, weight_decay=0.02, dropout=0.1)
BAD_A100_NODES = ('a100-4011', 'a100-4018', 'a100-4024', 'a100-4028', 'a100-4045')


def early_stopping(dataset):
    return dict(monitor='balanced_accuracy', min_epochs=4 if dataset == 'tuab' else 15,
                patience=4 if dataset == 'tuab' else 10, min_delta=0.0)


def prepare():
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-fixed-hp-seven-a100'
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if not verified.get('strict_load') or not checkpoint.is_file() or digest(checkpoint) != verified['sha256']:
        raise ValueError('Mask-55 pretrained checkpoint is not strictly verified')
    write_json(campaign / 'pretrain/verified.json', verified)
    entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            template = BASE / f'configs/downstream/{dataset}_seed{seed}.yaml'
            cfg = yaml.safe_load(template.read_text())
            if cfg['data']['dataset'] != dataset or cfg['seed'] != seed:
                raise ValueError(f'Unexpected frozen template: {template}')
            if not Path(cfg['data']['dataset_dir']).is_dir():
                raise FileNotFoundError(cfg['data']['dataset_dir'])
            opt = cfg['optimization']
            if any('warmup' in str(key).lower() for key in opt):
                raise ValueError(f'Downstream warmup is forbidden: {dataset}')
            for key in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
                opt[key] = HP['learning_rate']
            opt['weight_decay'] = HP['weight_decay']
            opt['early_stopping'] = early_stopping(dataset)
            if dataset == 'tuab':
                opt['epochs'] = 10
            cfg['model']['head_dropout'] = HP['dropout']
            cfg['model']['checkpoint'] = str(checkpoint)
            cfg['data']['num_workers'] = 2
            output = campaign / 'runs' / dataset / f'seed-{seed}'
            cfg['runtime']['output'] = str(output)
            path = campaign / 'configs' / dataset / f'seed-{seed}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            entries.append(dict(index=len(entries), dataset=dataset, seed=seed,
                                config=str(path), config_sha256=digest(path), output=str(output)))
    if len(entries) != 35:
        raise AssertionError('Expected seven datasets by five seeds')
    write_json(campaign / 'downstream_entries.json', entries)
    manifest = dict(status='prepared', created_at_new_york=stamp,
                    base_experiment=str(BASE), checkpoint=str(checkpoint),
                    checkpoint_sha256=verified['sha256'], python=sys.executable,
                    datasets=list(DATASETS), seeds=list(SEEDS), fixed_hp=HP,
                    tuab_first='All six other arrays depend on successful completion of all five TUAB seeds',
                    early_stopping={d: early_stopping(d) for d in DATASETS},
                    selection='Per-seed best validation balanced accuracy; test reporting only',
                    resources=dict(gpu='a100', partitions='a100_dev,a100_short,a100_long',
                                   time='04:00:00', excluded_nodes=list(BAD_A100_NODES)),
                    jobs=[])
    write_json(campaign / 'manifest.json', manifest)
    return campaign


def submit(campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['jobs'] or manifest.get('finalizer_job'):
        raise ValueError('Campaign already submitted')
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    tuab_job = None
    for dataset in DATASETS:
        indices = [i for i, entry in enumerate(json.loads((campaign / 'downstream_entries.json').read_text()))
                   if entry['dataset'] == dataset]
        command = ['sbatch', '--parsable', '--account=system',
                   '--job-name=mask55-fixed-' + dataset,
                   '--partition=a100_dev,a100_short,a100_long', '--nodes=1', '--ntasks=1',
                   '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G', '--time=04:00:00',
                   '--array=' + ','.join(map(str, indices)) + '%5',
                   '--exclude=' + ','.join(BAD_A100_NODES),
                   '--chdir=' + str(campaign / 'source'),
                   '--output=' + str(logs / (dataset + '-%A_%a.out')),
                   '--error=' + str(logs / (dataset + '-%A_%a.err')),
                   '--wrap=exec ' + shlex.join([
                       'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                       '--kill-on-bad-exit=1', manifest['python'],
                       str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                       '--experiment', str(campaign), '--source', str(campaign / 'source')])]
        if dataset != 'tuab':
            command[-1:-1] = ['--dependency=afterok:' + tuab_job, '--kill-on-invalid-dep=yes']
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        if dataset == 'tuab':
            tuab_job = job
        manifest['jobs'].append(dict(dataset=dataset, job=job, indices=indices,
                                     dependency=tuab_job if dataset != 'tuab' else None,
                                     command=command))
        write_json(manifest_path, manifest)
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=system', '--job-name=mask55-fixed-seven-results',
        '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--mem=4G', '--time=00:30:00',
        '--dependency=afterany:' + ':'.join(job['job'] for job in manifest['jobs']),
        '--output=' + str(logs / 'aggregate-%j.out'),
        '--error=' + str(logs / 'aggregate-%j.err'),
        '--wrap=exec ' + shlex.join([
            manifest['python'], str(campaign / 'source/scripts/submit_mask55_fixed_hp_seven.py'),
            'aggregate', '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest.update(status='submitted', finalizer_job=finalizer)
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), jobs=manifest['jobs'],
                          finalizer_job=finalizer), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    rows, missing = [], []
    for entry in entries:
        try:
            if digest(entry['config']) != entry['config_sha256']:
                raise ValueError('Frozen config changed')
            result = json.loads((Path(entry['output']) / 'result.json').read_text())
            chosen = result['balanced_accuracy']
            validation = float(chosen['selection']['score'])
            test = {key: float(value) for key, value in chosen['test'].items()
                    if key in ('balanced_accuracy', 'weighted_f1', 'kappa', 'auroc', 'auprc')}
            if not math.isfinite(validation) or not all(math.isfinite(value) for value in test.values()):
                raise ValueError('Non-finite metric')
            rows.append(dict(**entry, validation_bacc=validation,
                             selected_epoch=int(chosen['selection']['epoch']), test=test))
        except Exception as exc:
            missing.append(dict(dataset=entry['dataset'], seed=entry['seed'], error=repr(exc)))
    summary = {}
    for dataset in DATASETS:
        group = [row for row in rows if row['dataset'] == dataset]
        if len(group) != 5 or {row['seed'] for row in group} != set(SEEDS):
            continue
        metrics = sorted(set.intersection(*(set(row['test']) for row in group)))
        summary[dataset] = dict(n=5, validation_bacc_mean=statistics.mean(row['validation_bacc'] for row in group),
            validation_bacc_sd=statistics.pstdev(row['validation_bacc'] for row in group),
            test={metric: dict(mean=statistics.mean(row['test'][metric] for row in group),
                               sd=statistics.pstdev(row['test'][metric] for row in group)) for metric in metrics},
            seeds=[dict(seed=row['seed'], validation_bacc=row['validation_bacc'],
                        selected_epoch=row['selected_epoch'], test=row['test']) for row in group])
    status = 'complete' if len(rows) == 35 and len(summary) == 7 and not missing else 'incomplete'
    write_json(campaign / 'fixed_hp_summary.json',
               dict(status=status, expected_runs=35, readable_runs=len(rows), missing=missing,
                    fixed_hp=HP, results=summary, test_used_for_selection=False))
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = status
    write_json(manifest_path, manifest)
    print(json.dumps(dict(status=status, readable_runs=len(rows), missing=missing,
                          results=summary), indent=2))
    return 0 if status == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'launch', 'aggregate'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action in ('prepare', 'launch'):
        campaign = prepare()
        if args.action == 'launch':
            submit(campaign)
        else:
            print(campaign)
    else:
        if args.campaign is None:
            parser.error(f'{args.action} requires --campaign')
        campaign = args.campaign.resolve()
        if args.action == 'submit':
            submit(campaign)
        else:
            return aggregate(campaign)
    return 0


if __name__ == '__main__':
    sys.exit(main())
