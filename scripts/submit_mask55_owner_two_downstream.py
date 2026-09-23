"""Submit ten final-default five-seed datasets for verified owner mask55 arms.

TUAB is submitted separately by submit_mask55_tuab_comparison.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import submit_mask55_final_comparison as common
from scripts.submit_experiment_downstream import prepare

OWNER4 = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260923-0033-mask55-d2-patchdim-owner-encoder-four-pretrain')
ARMS = ('ours-lite', 'enc-t2s-6stage')
DATASETS = common.DATASETS
HP = {name: {'learning_rate': lr, 'weight_decay': wd, 'head_dropout': drop}
      for name, (lr, wd, drop) in common.HP.items()}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def prepare_stage(original: Path, arm: str) -> tuple[Path, list[dict]]:
    checkpoint, proof = common.verify_checkpoint(original, .55)
    stage = original / 'final_default_downstream'
    manifest_path = stage / 'manifest.json'
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (manifest.get('source_pretrain') != str(original)
                or manifest.get('checkpoint_sha256') != proof['sha256']
                or manifest.get('hyperparameters') !=
                {name: list(value) for name, value in common.HP.items()}):
            raise ValueError(f'Existing stage provenance differs: {stage}')
    else:
        stage.mkdir(parents=True, exist_ok=False)
        (stage / 'source').symlink_to(ROOT, target_is_directory=True)
        (stage / 'pretrain').mkdir()
        (stage / 'pretrain/checkpoint-epoch-0040.pth').symlink_to(checkpoint)
        stage_proof = dict(proof, checkpoint=str(stage / 'pretrain/checkpoint-epoch-0040.pth'))
        (stage / 'pretrain/verified.json').write_text(json.dumps(stage_proof, indent=2) + '\n')
        (stage / 'configs').mkdir()
        shutil.copy2(original / 'configs/pretrain.yaml', stage / 'configs/pretrain.yaml')
        manifest = {
            'experiment': f'mask55-{arm}-ten-default', 'preset': arm, 'owner': 'ml10266',
            'source_pretrain': str(original), 'checkpoint_sha256': proof['sha256'],
            'hyperparameters': {name: list(value) for name, value in common.HP.items()},
            'created_at_new_york': datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M'),
            'publication_root': str(stage / 'outputs/results'),
            'selection': 'validation balanced_accuracy', 'test_role': 'reporting only',
            'downstream_jobs': [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    entries = prepare(stage, stage / 'pretrain/checkpoint-epoch-0040.pth', DATASETS, arm, HP)
    if len(entries) != 50 or {entry['seed'] for entry in entries} != set(common.SEEDS):
        raise ValueError('Expected ten datasets and five seeds each')
    for entry in entries:
        cfg = yaml.safe_load(Path(entry['config']).read_text(encoding='utf-8'))
        lr, wd, drop = common.HP[entry['dataset']]
        if (any(abs(cfg['optimization'][name + '_learning_rate'] - lr) > 1e-12
                for name in ('tokenizer', 'encoder', 'head'))
                or abs(cfg['optimization']['weight_decay'] - wd) > 1e-12
                or abs(cfg['model']['head_dropout'] - drop) > 1e-12):
            raise ValueError(f'Frozen downstream HP differs: {entry["config"]}')
    return stage, entries


def submit_stage(stage: Path, entries: list[dict], arm: str) -> list[tuple[str, str]]:
    manifest_path = stage / 'manifest.json'
    manifest = read_json(manifest_path)
    registered = {job['dataset'] for job in manifest['downstream_jobs']}
    submitted = []
    for dataset in DATASETS:
        if dataset in registered:
            continue
        missing = [index for index, entry in enumerate(entries)
                   if entry['dataset'] == dataset
                   and not common.complete(Path(entry['output']) / 'result.json')]
        if not missing:
            continue
        policy = common.resources('gl40s', dataset)
        logs = stage / 'downstream/logs'
        logs.mkdir(parents=True, exist_ok=True)
        wrap = shlex.join([
            'srun', '--ntasks=1', '--gpus-per-task=l40s:1', '--gpu-bind=single:1',
            '--kill-on-bad-exit=1', sys.executable,
            str(ROOT / 'scripts/downstream_experiment_worker.py'), '--experiment', str(stage),
        ])
        command = [
            'sbatch', '--parsable', '--account=system',
            '--job-name=mask55-' + arm + '-' + dataset,
            '--partition=' + policy['partitions'], '--nodes=1', '--ntasks=1',
            '--gpus-per-task=l40s:1', '--cpus-per-task=4', '--mem=32G',
            '--time=' + policy['time'], '--array=' + ','.join(map(str, missing)) + '%5',
            '--output=' + str(logs / '%A_%a.out'),
            '--error=' + str(logs / '%A_%a.err'), '--wrap=' + wrap,
        ]
        if policy['excluded_nodes']:
            command.append('--exclude=' + ','.join(policy['excluded_nodes']))
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
        if not re.fullmatch(r'\d+', job):
            raise ValueError(f'Unexpected downstream job: {job}')
        manifest['downstream_jobs'].append({
            'dataset': dataset, 'job': job, 'indices': missing, 'resources': policy,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        submitted.append((dataset, job))
    if manifest['downstream_jobs'] and not manifest.get('finalizer_job'):
        subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                        '--experiment', str(stage), '--submit'], check=True)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if getpass.getuser() != 'ml10266':
        parser.error('Only ml10266 may submit owner ablation downstream jobs')
    for arm in (args.arm,) if args.arm else ARMS:
        original = OWNER4 / arm
        if not (original / 'pretrain/verified.json').is_file():
            raise FileNotFoundError(f'No verified pretrain for {arm}: {original}')
        stage, entries = prepare_stage(original, arm)
        submitted = submit_stage(stage, entries, arm)
        print(f'{arm}: jobs={submitted}; results={stage}', flush=True)


if __name__ == '__main__':
    main()
