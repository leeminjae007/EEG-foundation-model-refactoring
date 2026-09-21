"""One-factor-at-a-time TUAB search on one seed, selected by validation BAcc."""
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

from submit_mask55_hp_grid import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1348-gr2-d2-mask55-tuab-a100')
SEED = 42
BASE_HP = dict(learning_rate=1e-5, weight_decay=5e-5, dropout=0.1)
AXES = dict(learning_rate=(5e-6, 2e-5, 5e-5),
            weight_decay=(0.0, 5e-4, 5e-3, 1e-2),
            dropout=(0.0, 0.2, 0.3, 0.4))
EARLY_STOPPING = dict(monitor='balanced_accuracy', min_epochs=4, patience=4, min_delta=0.0)


def candidates():
    result = [dict(axis='baseline', value=None, hp=dict(BASE_HP))]
    for axis, values in AXES.items():
        for value in values:
            hp = dict(BASE_HP)
            hp[axis] = value
            result.append(dict(axis=axis, value=value, hp=hp))
    assert len(result) == 12 and len({tuple(x['hp'].items()) for x in result}) == 12
    return result


def prepare(output_root):
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M')
    campaign = output_root.resolve() / (stamp + '-gr2-d2-mask55-tuab-onefactor-seed42')
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if not checkpoint.is_file() or digest(checkpoint) != verified['sha256']:
        raise ValueError('Pretrain checkpoint is absent or differs from verified SHA256')
    write_json(campaign / 'pretrain/verified.json', verified)
    base = yaml.safe_load((BASE / f'configs/downstream/tuab_seed{SEED}.yaml').read_text())
    if base['seed'] != SEED or base['data']['dataset'] != 'tuab':
        raise ValueError('Unexpected TUAB base configuration')
    opt = base['optimization']
    if (opt['tokenizer_learning_rate'], opt['encoder_learning_rate'], opt['head_learning_rate'],
            opt['weight_decay'], base['model']['head_dropout']) != (
            BASE_HP['learning_rate'],) * 3 + (BASE_HP['weight_decay'], BASE_HP['dropout']):
        raise ValueError('TUAB baseline hyperparameters changed')
    if any('warmup' in str(key).lower() for key in opt):
        raise ValueError('Downstream warmup is forbidden')
    entries = []
    for index, item in enumerate(candidates()):
        cfg = json.loads(json.dumps(base))
        hp = item['hp']
        for key in ('tokenizer_learning_rate', 'encoder_learning_rate', 'head_learning_rate'):
            cfg['optimization'][key] = hp['learning_rate']
        cfg['optimization']['epochs'] = 10
        cfg['optimization']['weight_decay'] = hp['weight_decay']
        cfg['optimization']['early_stopping'] = dict(EARLY_STOPPING)
        cfg['model']['head_dropout'] = hp['dropout']
        cfg['model']['checkpoint'] = str(checkpoint)
        output = campaign / 'runs' / f'{index:02d}-{item["axis"]}'
        cfg['runtime']['output'] = str(output)
        path = campaign / 'configs' / f'{index:02d}-{item["axis"]}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        entries.append(dict(index=index, dataset='tuab', seed=SEED, axis=item['axis'],
                            value=item['value'], hp=hp, config=str(path),
                            config_sha256=digest(path), output=str(output)))
    write_json(campaign / 'downstream_entries.json', entries)
    manifest = dict(status='prepared', created_at_new_york=stamp, seed=SEED,
                    base_experiment=str(BASE), checkpoint=str(checkpoint),
                    checkpoint_sha256=verified['sha256'], python=sys.executable,
                    baseline=BASE_HP, axes=AXES, early_stopping=EARLY_STOPPING,
                    selection='Highest single-seed validation BAcc only; tie: lower index',
                    test_policy='Test is reporting-only, never used to choose settings',
                    resources=dict(gpu='a100', partitions='a100_dev,a100_short,a100_long',
                                   time='04:00:00', concurrency=5), jobs=[])
    write_json(campaign / 'manifest.json', manifest)
    return campaign


def submit(campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['jobs']:
        raise ValueError('Already submitted')
    logs = campaign / 'logs'
    logs.mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--account=system', '--job-name=mask55-tuab-onefactor',
               '--partition=a100_dev,a100_short,a100_long', '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G', '--time=04:00:00',
               '--array=0-11%5', '--chdir=' + str(campaign / 'source'),
               '--output=' + str(logs / 'tuab-%A_%a.out'),
               '--error=' + str(logs / 'tuab-%A_%a.err'),
               '--wrap=exec ' + shlex.join([
                   'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                   '--kill-on-bad-exit=1', manifest['python'],
                   str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                   '--experiment', str(campaign), '--source', str(campaign / 'source')])]
    job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    manifest['jobs'].append(dict(job=job, command=command))
    write_json(manifest_path, manifest)
    finalizer = subprocess.check_output([
        'sbatch', '--parsable', '--account=system', '--job-name=mask55-tuab-onefactor-results',
        '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--mem=4G', '--time=00:30:00', '--dependency=afterany:' + job,
        '--output=' + str(logs / 'aggregate-%j.out'),
        '--error=' + str(logs / 'aggregate-%j.err'),
        '--wrap=exec ' + shlex.join([
            manifest['python'], str(campaign / 'source/scripts/submit_mask55_tuab_onefactor.py'),
            'aggregate', '--campaign', str(campaign)])
    ], text=True).strip().split(';', 1)[0]
    manifest.update(finalizer_job=finalizer, status='submitted')
    write_json(manifest_path, manifest)
    print(json.dumps(dict(campaign=str(campaign), job=job, finalizer=finalizer), indent=2))


def aggregate(campaign):
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    rows, missing = [], []
    for entry in entries:
        try:
            if digest(entry['config']) != entry['config_sha256']:
                raise ValueError('Frozen config changed')
            result = json.loads((Path(entry['output']) / 'result.json').read_text())
            selected = result['balanced_accuracy']
            validation = float(selected['selection']['score'])
            test = {key: float(value) for key, value in selected['test'].items()
                    if isinstance(value, (int, float))}
            if not math.isfinite(validation) or not all(math.isfinite(value) for value in test.values()):
                raise ValueError('Non-finite result')
            rows.append(dict(**entry, validation_bacc=validation,
                             selected_epoch=selected['selection']['epoch'], test=test))
        except Exception as exc:
            missing.append(dict(index=entry['index'], error=repr(exc)))
    ranked = sorted(rows, key=lambda row: (-row['validation_bacc'], row['index']))
    status = 'complete' if len(rows) == len(entries) == 12 else 'incomplete'
    write_json(campaign / 'onefactor_summary.json',
               dict(status=status, expected=12, readable=len(rows), missing=missing,
                    validation_winner=ranked[0] if ranked and status == 'complete' else None,
                    ranked_by_validation=ranked))
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['status'] = status
    write_json(manifest_path, manifest)
    print(json.dumps(dict(status=status, readable=len(rows), missing=missing,
                          validation_winner=ranked[0] if ranked and status == 'complete' else None), indent=2))
    return 0 if status == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'aggregate'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'aggregate':
        if args.campaign is None:
            parser.error('aggregate requires --campaign')
        return aggregate(args.campaign.resolve())
    campaign = prepare(args.output_root)
    print(campaign)
    if args.action == 'launch':
        submit(campaign)
    return 0


if __name__ == '__main__':
    sys.exit(main())
