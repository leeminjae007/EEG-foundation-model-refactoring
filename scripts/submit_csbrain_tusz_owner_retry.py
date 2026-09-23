"""Retry CSBrain TUSZ on ml10266 with yc8820's frozen five-seed settings.

The collaborator's failed outputs are left untouched. This creates a separate
owner-writable stage and refuses to duplicate active or completed seeds.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_final_comparison import verify_checkpoint

RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
SHARED = RESULTS / '260921-1456-mask55-d2-patchdim-shared-pretrain'
ORIGINAL = SHARED / 'accounts/yc8820/csbrain'
FROZEN = SHARED / 'accounts/yc8820/final_default_downstream/csbrain'
STAGE = RESULTS / '260923-csbrain-tusz-owner-retry'
SEEDS = (42, 696, 1001, 1234, 3407)
HP = (2e-4, .005, .3)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def complete(path: Path) -> bool:
    try:
        result = read_json(path)
        metric = result['balanced_accuracy']
        return (isinstance(metric['selection']['score'], (int, float))
                and isinstance(metric['test']['balanced_accuracy'], (int, float)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def live(job: str) -> bool:
    query = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                           text=True, capture_output=True)
    if query.returncode:
        if 'Invalid job id specified' in query.stderr:
            return False
        raise RuntimeError(f'squeue {job}: {query.stderr.strip()}')
    return bool(query.stdout.strip())


def config_without_paths(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    config['model'].pop('checkpoint')
    config['runtime'].pop('output')
    return config


def prepare() -> tuple[list[dict], dict]:
    checkpoint, proof = verify_checkpoint(ORIGINAL, .55)
    frozen_manifest = read_json(FROZEN / 'manifest.json')
    if (frozen_manifest.get('checkpoint_sha256') != proof['sha256']
            or frozen_manifest.get('hyperparameters', {}).get('tusz') != list(HP)):
        raise ValueError('CSBrain TUSZ frozen provenance or hyperparameters differ')
    old_jobs = [str(job['job']) for job in frozen_manifest['downstream_jobs']
                if job['dataset'] == 'tusz']
    if len(old_jobs) != 1 or any(live(job) for job in old_jobs):
        raise RuntimeError(f'Original TUSZ array is still active or ambiguous: {old_jobs}')
    STAGE.mkdir(parents=True, exist_ok=True)
    source_link = STAGE / 'source'
    if not source_link.exists():
        source_link.symlink_to(ROOT, target_is_directory=True)
    elif source_link.resolve() != ROOT:
        raise ValueError('Stage source points to unexpected code')
    (STAGE / 'pretrain').mkdir(exist_ok=True)
    stage_checkpoint = STAGE / 'pretrain/checkpoint-epoch-0040.pth'
    if not stage_checkpoint.exists():
        stage_checkpoint.symlink_to(checkpoint)
    if stage_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError('Stage checkpoint differs from verified source')
    stage_proof = dict(proof, checkpoint=str(stage_checkpoint))
    proof_path = STAGE / 'pretrain/verified.json'
    if proof_path.exists() and read_json(proof_path) != stage_proof:
        raise ValueError('Existing stage verification differs')
    proof_path.write_text(json.dumps(stage_proof, indent=2) + '\n')
    (STAGE / 'configs/downstream').mkdir(parents=True, exist_ok=True)
    pretrain_config = STAGE / 'configs/pretrain.yaml'
    if not pretrain_config.exists():
        shutil.copy2(FROZEN / 'configs/pretrain.yaml', pretrain_config)
    entries = []
    for seed in SEEDS:
        frozen = yaml.safe_load((FROZEN / f'configs/downstream/tusz_seed{seed}.yaml')
                                .read_text(encoding='utf-8'))
        opt = frozen['optimization']
        if (any(abs(opt[f'{name}_learning_rate'] - HP[0]) > 1e-12
                for name in ('tokenizer', 'encoder', 'head'))
                or abs(opt['weight_decay'] - HP[1]) > 1e-12
                or abs(frozen['model']['head_dropout'] - HP[2]) > 1e-12):
            raise ValueError(f'Frozen TUSZ hyperparameters differ for seed {seed}')
        output = STAGE / f'downstream/tusz/seed-{seed}'
        config_path = STAGE / f'configs/downstream/tusz_seed{seed}.yaml'
        frozen['model']['checkpoint'] = str(stage_checkpoint)
        frozen['runtime']['output'] = str(output)
        if config_path.exists():
            current = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            if current != frozen:
                raise ValueError(f'Existing TUSZ config differs for seed {seed}')
        else:
            config_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding='utf-8')
        prior = yaml.safe_load((FROZEN / f'configs/downstream/tusz_seed{seed}.yaml')
                               .read_text(encoding='utf-8'))
        if config_without_paths(prior) != config_without_paths(frozen):
            raise ValueError(f'Non-path configuration changed for seed {seed}')
        entries.append(dict(dataset='tusz', seed=seed, config=str(config_path), output=str(output)))
    entries_path = STAGE / 'downstream_entries.json'
    if entries_path.exists() and read_json(entries_path) != entries:
        raise ValueError('Existing downstream entries differ')
    entries_path.write_text(json.dumps(entries, indent=2) + '\n')
    manifest_path = STAGE / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get('checkpoint_sha256') != proof['sha256']
                or manifest.get('source_pretrain') != str(ORIGINAL)
                or manifest.get('hyperparameters') != {'tusz': list(HP)}):
            raise ValueError('Existing retry manifest differs')
    else:
        manifest = dict(preset='csbrain', owner='ml10266', experiment=STAGE.name,
                        source_pretrain=str(ORIGINAL), frozen_source=str(FROZEN),
                        prior_failed_job=old_jobs[0], checkpoint_sha256=proof['sha256'],
                        hyperparameters={'tusz': list(HP)},
                        selection='validation balanced_accuracy', test_role='reporting only',
                        downstream_jobs=[])
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return entries, manifest


def main() -> None:
    if getpass.getuser() != 'ml10266':
        raise PermissionError('Run this retry only as ml10266')
    entries, manifest = prepare()
    missing = [index for index, entry in enumerate(entries)
               if not complete(Path(entry['output']) / 'result.json')]
    print(f'CSBrain TUSZ: {5 - len(missing)}/5 complete; missing={missing}', flush=True)
    if not missing:
        return
    if any(live(str(job['job'])) for job in manifest['downstream_jobs']):
        raise RuntimeError('Existing owner retry job remains active; refusing duplicate submission')
    logs = STAGE / 'downstream/logs'
    logs.mkdir(parents=True, exist_ok=True)
    frozen_jobs = [job for job in read_json(FROZEN / 'manifest.json')['downstream_jobs']
                   if job['dataset'] == 'tusz']
    policy = frozen_jobs[0]['resources']
    wrap = shlex.join([
        'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
        '--kill-on-bad-exit=1', sys.executable,
        str(ROOT / 'scripts/downstream_experiment_worker.py'), '--experiment', str(STAGE),
    ])
    command = [
        'sbatch', '--parsable', '--account=system', '--job-name=csbrain-tusz-ml-retry',
        '--partition=' + policy['partitions'], '--nodes=1', '--ntasks=1',
        '--gpus-per-task=a100:1', '--cpus-per-task=' + str(policy['cpus_per_task']),
        '--mem=' + policy['memory'], '--time=' + policy['time'],
        '--array=' + ','.join(map(str, missing)) + '%5',
        '--output=' + str(logs / '%A_%a.out'),
        '--error=' + str(logs / '%A_%a.err'), '--wrap=' + wrap,
    ]
    if policy['excluded_nodes']:
        command.append('--exclude=' + ','.join(policy['excluded_nodes']))
    job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    if not re.fullmatch(r'\d+', job):
        raise RuntimeError(f'Unexpected job ID: {job}')
    manifest['downstream_jobs'].append(dict(dataset='tusz', job=job,
                                            indices=missing, resources=policy))
    (STAGE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'submitted={job}; results={STAGE}', flush=True)


if __name__ == '__main__':
    main()
