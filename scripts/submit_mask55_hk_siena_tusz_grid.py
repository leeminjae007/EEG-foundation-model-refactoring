"""Submit hk4935's independent mask-55 Siena/TUSZ five-seed A100 HP grid.

After activating the collaborator environment in the owner's checkout:
    python scripts/submit_mask55_hk_siena_tusz_grid.py launch
"""

from __future__ import annotations

import argparse
import getpass
import itertools
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import candidate_id, digest, write_json

BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55')
CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1456-mask55-d2-patchdim-shared-pretrain/accounts/hk4935/hp-grid-siena-tusz-a100')
EXPECTED_SHA256 = 'a5ab908bde519241b199efcc701ea1352b3fcd9da068f600549845bef65babca'
DATASETS = ('siena', 'tusz')
SEEDS = (42, 696, 1001, 1234, 3407)
LEARNING_RATES = (2.5e-5, 5e-5, 1e-4, 2e-4)
WEIGHT_DECAYS = (.005, .01, .02)
DROPOUTS = (.1, .2, .3)
TIME_LIMITS = {'siena': '04:00:00', 'tusz': '04:00:00'}
EARLY_STOPPING = dict(monitor='balanced_accuracy', patience=10, min_epochs=15, min_delta=0.0)


def candidates():
    return [dict(learning_rate=lr, weight_decay=wd, dropout=dropout)
            for lr, wd, dropout in itertools.product(LEARNING_RATES, WEIGHT_DECAYS, DROPOUTS)]


def prepare():
    if CAMPAIGN.exists():
        raise FileExistsError(f'Campaign already exists; refusing duplicate: {CAMPAIGN}')
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if (not verified.get('strict_load') or verified.get('partial_epoch_smoke') or
            verified.get('sha256') != EXPECTED_SHA256 or digest(checkpoint) != EXPECTED_SHA256):
        raise ValueError('The verified mask-55 depth-2 patch-dimension checkpoint does not match')
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    source = CAMPAIGN / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    (CAMPAIGN / 'pretrain').mkdir()
    write_json(CAMPAIGN / 'pretrain/verified.json', verified)
    entries = []
    for dataset in DATASETS:
        for hp in candidates():
            cid = candidate_id(hp)
            for seed in SEEDS:
                config = yaml.safe_load((BASE / 'configs/downstream' / f'{dataset}_seed{seed}.yaml').read_text())
                if config['data']['dataset'] != dataset or config['seed'] != seed:
                    raise ValueError(f'Frozen base config mismatch: {dataset}, {seed}')
                optimization = config['optimization']
                if any('warmup' in str(key).lower() for key in optimization):
                    raise ValueError('Downstream warmup is forbidden')
                for key in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
                    optimization[key] = hp['learning_rate']
                optimization['weight_decay'] = hp['weight_decay']
                optimization['early_stopping'] = EARLY_STOPPING.copy()
                config['model']['head_dropout'] = hp['dropout']
                config['model']['checkpoint'] = str(checkpoint)
                output = CAMPAIGN / 'runs' / dataset / cid / f'seed-{seed}'
                config['runtime']['output'] = str(output)
                path = CAMPAIGN / 'configs' / dataset / f'{cid}_seed{seed}.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
                entries.append(dict(index=len(entries), dataset=dataset, seed=seed,
                                    candidate=cid, hp=hp, config=str(path),
                                    config_sha256=digest(path), output=str(output)))
    if len(candidates()) != 36 or len(entries) != 360:
        raise AssertionError('Expected two datasets x 36 candidates x five seeds')
    write_json(CAMPAIGN / 'downstream_entries.json', entries)
    cluster = yaml.safe_load((source / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    write_json(CAMPAIGN / 'manifest.json', dict(
        owner='hk4935', status='prepared', base_experiment=str(BASE),
        checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED_SHA256,
        datasets=list(DATASETS), seeds=list(SEEDS), learning_rates=list(LEARNING_RATES),
        weight_decays=list(WEIGHT_DECAYS), dropouts=list(DROPOUTS),
        early_stopping=EARLY_STOPPING, total_runs=360, jobs=[],
        selection='Five-seed mean validation balanced accuracy only',
        test_policy='Test BAcc/AUROC/AUPRC reporting only; no test-driven selection',
        resources=dict(gpu='a100', partitions='a100_dev,a100_short,a100_long',
                       concurrency_per_dataset=5, time_limits=TIME_LIMITS,
                       excluded_nodes=cluster['pretrain']['excluded_nodes'])))
    return CAMPAIGN


def submit(campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] not in ('prepared', 'submitting'):
        raise ValueError('Campaign is already submitted')
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    resources = manifest['resources']
    for dataset in DATASETS:
        if any(job['dataset'] == dataset for job in manifest['jobs']):
            continue
        indices = [entry['index'] for entry in entries if entry['dataset'] == dataset]
        if len(indices) != 180:
            raise ValueError(f'{dataset}: expected 180 frozen entries')
        command = ['sbatch', '--parsable', '--account=system',
                   '--job-name=mask55-hk-grid-' + dataset,
                   '--partition=' + resources['partitions'], '--nodes=1', '--ntasks=1',
                   '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G',
                   '--time=' + resources['time_limits'][dataset],
                   '--array=' + ','.join(map(str, indices)) + '%5',
                   '--chdir=' + str(campaign / 'source'),
                   '--output=' + str(logs / f'{dataset}-%A_%a.out'),
                   '--error=' + str(logs / f'{dataset}-%A_%a.err')]
        if resources.get('excluded_nodes'):
            command.append('--exclude=' + ','.join(resources['excluded_nodes']))
        command.append('--wrap=exec ' + shlex.join([
                       'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                       '--kill-on-bad-exit=1', sys.executable,
                       str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                       '--experiment', str(campaign), '--source', str(campaign / 'source')]))
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        manifest['jobs'].append(dict(dataset=dataset, job=job, indices=indices))
        manifest['status'] = 'submitting'
        write_json(manifest_path, manifest)
    if not manifest.get('aggregate_job'):
        dependency = ':'.join(job['job'] for job in manifest['jobs'])
        command = ['sbatch', '--parsable', '--account=system',
                   '--job-name=mask55-hk-grid-results', '--partition=cpu_short,cpu_long',
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                   '--time=00:30:00', '--dependency=afterany:' + dependency,
                   '--output=' + str(logs / 'aggregate-%j.out'),
                   '--error=' + str(logs / 'aggregate-%j.err'),
                   '--wrap=exec ' + shlex.join([sys.executable,
                       str(campaign / 'source/scripts/submit_mask55_hk_siena_tusz_grid.py'),
                       'aggregate', '--campaign', str(campaign)])]
        manifest['aggregate_job'] = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    manifest['status'] = 'submitted'
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), jobs=manifest['jobs'],
                          aggregate_job=manifest['aggregate_job']), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    readable, missing = [], []
    for entry in entries:
        try:
            if digest(entry['config']) != entry['config_sha256']:
                raise ValueError('Frozen config changed')
            output = Path(entry['output'])
            payload = json.loads((output / 'result.json').read_text())
            selected = payload['balanced_accuracy']
            validation = float(selected['selection']['score'])
            epoch = int(selected['selection']['epoch'])
            test = {key: float(selected['test'][key])
                    for key in ('balanced_accuracy', 'auroc', 'auprc')}
            if (not math.isfinite(validation) or epoch < 1 or
                    not all(math.isfinite(value) for value in test.values()) or
                    not (output / 'best-balanced_accuracy.pth').is_file() or
                    not (output / 'last.pth').is_file()):
                raise ValueError('Incomplete or non-finite validation-selected result')
            readable.append(dict(**entry, validation_bacc=validation,
                                 selected_epoch=epoch, test=test))
        except Exception as exc:
            missing.append(dict(dataset=entry['dataset'], candidate=entry['candidate'],
                                seed=entry['seed'], error=repr(exc)))
    summaries = []
    for dataset in DATASETS:
        for hp in candidates():
            cid = candidate_id(hp)
            group = [row for row in readable if row['dataset'] == dataset and row['candidate'] == cid]
            if len(group) != 5 or {row['seed'] for row in group} != set(SEEDS):
                continue
            summaries.append(dict(dataset=dataset, candidate=cid, hp=hp,
                validation_bacc_mean=statistics.mean(row['validation_bacc'] for row in group),
                validation_bacc_sd=statistics.pstdev(row['validation_bacc'] for row in group),
                test={metric: dict(mean=statistics.mean(row['test'][metric] for row in group),
                                   sd=statistics.pstdev(row['test'][metric] for row in group))
                      for metric in ('balanced_accuracy', 'auroc', 'auprc')}))
    winners = {}
    for dataset in DATASETS:
        complete = [row for row in summaries if row['dataset'] == dataset]
        if complete:
            winners[dataset] = min(complete, key=lambda row: (
                -row['validation_bacc_mean'], row['validation_bacc_sd'], row['candidate']))
    status = 'complete' if not missing and len(readable) == 360 and len(summaries) == 72 else 'incomplete'
    write_json(campaign / 'grid_summary.json', dict(
        status=status, expected_runs=360, readable_runs=len(readable), missing=missing,
        candidate_summaries=summaries, winner_by_validation=winners,
        test_used_for_selection=False))
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = status
    write_json(manifest_path, manifest)
    print(json.dumps(dict(status=status, readable_runs=len(readable),
                          missing_runs=len(missing), winners=winners), indent=2))
    return 0 if status == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'aggregate'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'launch':
        if getpass.getuser() != 'hk4935':
            parser.error('Only hk4935 may launch this campaign')
        campaign = CAMPAIGN if CAMPAIGN.exists() else prepare()
        submit(campaign)
        return 0
    if args.campaign is None:
        parser.error('aggregate requires --campaign')
    return aggregate(args.campaign.resolve())


if __name__ == '__main__':
    sys.exit(main())
