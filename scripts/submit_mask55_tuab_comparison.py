"""Submit only mask-55 TUAB five-seed downstream comparisons from the assigned account.

The owner updates the shared checkout. Collaborators activate their own CUDA
environment and run --account hk4935/yc8820 with the agreed, common TUAB HP.
Only verified owner pretrains are eligible for immediate submission.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import hashlib
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

SHARED = common.SHARED
STAGE3 = common.STAGE3
OWNER4 = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260923-0033-mask55-d2-patchdim-owner-encoder-four-pretrain')
ARMS = {
    'hk4935': {
        'pe-ch_order': (SHARED / 'accounts/hk4935/pe-ch_order', 'gl40s'),
        'pe-acpe': (SHARED / 'accounts/hk4935/pe-acpe', 'gl40s'),
        'pe-4dREVE': (SHARED / 'accounts/hk4935/pe-4dREVE', 'gl40s'),
    },
    'yc8820': {
        'cbramod': (SHARED / 'accounts/yc8820/cbramod', 'gl40s'),
        'csbrain': (SHARED / 'accounts/yc8820/csbrain', 'gl40s'),
        'labram': (SHARED / 'accounts/yc8820/labram', 'gl40s'),
        'enc-s2t-3stage': (STAGE3 / 'accounts/yc8820/s2t3', 'gl40s'),
        'enc-t2s-3stage': (STAGE3 / 'accounts/yc8820/t2s3', 'gl40s'),
    },
    'ml10266': {
        'ours-lite': (OWNER4 / 'ours-lite', 'gl40s'),
        'enc-t2s-6stage': (OWNER4 / 'enc-t2s-6stage', 'gl40s'),
    },
}
EARLY_STOPPING = {'monitor': 'balanced_accuracy', 'min_epochs': 4,
                  'patience': 10, 'min_delta': 0.0}


def checked_hp(lr: float, wd: float, dropout: float) -> dict:
    if not 0 < lr <= 1e-2 or not 0 <= wd < 1 or not 0 <= dropout < 1:
        raise ValueError('Invalid TUAB learning rate, weight decay, or dropout')
    return {'learning_rate': lr, 'weight_decay': wd,
            'head_dropout': dropout, 'early_stopping': dict(EARLY_STOPPING)}


def prepare_stage(original: Path, arm: str, account: str, hp: dict) -> tuple[Path, list[dict]]:
    checkpoint, proof = common.verify_checkpoint(original, .55)
    stage = original / 'tuab_final_comparison'
    manifest_path = stage / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest['checkpoint_sha256'] != proof['sha256']
                or manifest['hyperparameters'] != hp
                or manifest['source_pretrain'] != str(original)):
            raise ValueError(f'Existing TUAB comparison provenance differs: {stage}')
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
            'experiment': f'tuab-mask55-{arm}', 'preset': arm, 'owner': account,
            'source_pretrain': str(original), 'checkpoint_sha256': proof['sha256'],
            'created_at_new_york': datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M'),
            'publication_root': str(stage / 'outputs/results'),
            'hyperparameters': hp, 'selection': 'validation balanced_accuracy',
            'test_role': 'reporting only', 'downstream_jobs': [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    entries = prepare(stage, stage / 'pretrain/checkpoint-epoch-0040.pth',
                      ('tuab',), arm, {'tuab': hp})
    if len(entries) != 5 or {entry['seed'] for entry in entries} != set(common.SEEDS):
        raise ValueError('Expected exactly five TUAB seeds')
    for entry in entries:
        config = yaml.safe_load(Path(entry['config']).read_text(encoding='utf-8'))
        opt = config['optimization']
        if (opt['epochs'] != 20 or opt['batch_size_per_gpu'] != 64
                or opt['gradient_accumulation_steps'] != 8
                or opt['early_stopping'] != EARLY_STOPPING):
            raise ValueError('Frozen TUAB epoch/batch/early-stop settings differ')
    return stage, entries


def submit_stage(stage: Path, entries: list[dict], arm: str, gpu: str) -> str:
    manifest_path = stage / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['downstream_jobs']:
        if not manifest.get('finalizer_job'):
            subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                            '--experiment', str(stage), '--submit'], check=True)
        return str(manifest['downstream_jobs'][0]['job'])
    policy = common.resources(gpu, 'tuab')
    policy['partitions'] = 'a100_short,a100_long' if gpu == 'a100' else 'gl40s_long'
    policy['time'] = '24:00:00'
    # BigPurple names the L40S GRES ``l40s``, even though its partition is gl40s_*.
    slurm_gpu = 'l40s' if gpu == 'gl40s' else gpu
    missing = [index for index, entry in enumerate(entries)
               if not common.complete(Path(entry['output']) / 'result.json')]
    if not missing:
        return 'already_complete'
    logs = stage / 'downstream/logs'
    logs.mkdir(parents=True, exist_ok=True)
    wrap = shlex.join([
        'srun', '--ntasks=1', f'--gpus-per-task={slurm_gpu}:1',
        '--gpu-bind=single:1', '--kill-on-bad-exit=1', sys.executable,
        str(ROOT / 'scripts/downstream_experiment_worker.py'), '--experiment', str(stage),
    ])
    command = [
        'sbatch', '--parsable', '--account=system', '--job-name=tuab55-' + arm,
        '--partition=' + policy['partitions'], '--nodes=1', '--ntasks=1',
        f'--gpus-per-task={slurm_gpu}:1', '--cpus-per-task=4', '--mem=32G',
        '--time=' + policy['time'],
        '--array=' + ','.join(map(str, missing)) + '%5',
        '--output=' + str(logs / '%A_%a.out'),
        '--error=' + str(logs / '%A_%a.err'), '--wrap=' + wrap,
    ]
    if policy['excluded_nodes']:
        command.append('--exclude=' + ','.join(policy['excluded_nodes']))
    job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    if not re.fullmatch(r'\d+', job):
        raise ValueError(f'Unexpected sbatch response: {job}')
    manifest['downstream_jobs'].append({'dataset': 'tuab', 'job': job,
                                        'indices': missing, 'resources': policy})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                    '--experiment', str(stage), '--submit'], check=True)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True, choices=ARMS)
    parser.add_argument('--arm', choices=sorted({arm for arms in ARMS.values() for arm in arms}))
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--wd', required=True, type=float)
    parser.add_argument('--dropout', required=True, type=float)
    args = parser.parse_args()
    if getpass.getuser() != args.account:
        parser.error(f'Must submit from {args.account}')
    hp = checked_hp(args.lr, args.wd, args.dropout)
    arms = ARMS[args.account]
    if args.arm and args.arm not in arms:
        parser.error(f'{args.arm} is not assigned to {args.account}')
    selected = [args.arm] if args.arm else list(arms)
    for arm in selected:
        original, gpu = arms[arm]
        if not (original / 'pretrain/verified.json').is_file():
            raise FileNotFoundError(f'No verified pretrain for {arm}: {original}')
        stage, entries = prepare_stage(original, arm, args.account, hp)
        print(f'{arm}: TUAB array {submit_stage(stage, entries, arm, gpu)}; results {stage}', flush=True)


if __name__ == '__main__':
    main()
