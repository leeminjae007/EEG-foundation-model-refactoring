"""Submit fixed-HP HMC and Siena five-seed L40S jobs for yc8820 encoder arms."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1456-mask55-d2-patchdim-shared-pretrain')
OWNER = 'yc8820'
ARMS = {'cbramod': ('cbramod', 'shpe'),
        'csbrain': ('csbrain', 'shpe'),
        'labram': ('labram', 'shpe')}
DATASETS = ('hmc', 'siena')
FIXED = {
    'hmc': dict(candidate_id='f80de04f7907', learning_rate=5e-5,
                weight_decay=.005, head_dropout=.3),
    'siena': dict(candidate_id='d9c26d7257ff', learning_rate=2.5e-5,
                  weight_decay=.005, head_dropout=.1),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dependency(original):
    upstream = read_json(original / 'manifest.json')
    job = str(upstream['pretrain_entries'][0]['job'])
    state = subprocess.check_output(
        ['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State'], text=True)
    matches = [line.split('|')[1].split()[0] for line in state.splitlines()
               if line.split('|')[0] == job]
    if len(matches) != 1:
        raise RuntimeError(f'{original}: cannot resolve pretrain state for {job}')
    verified = original / 'pretrain/verified.json'
    if matches[0] == 'COMPLETED' and verified.is_file():
        proof = read_json(verified)
        if not proof.get('strict_load') or proof.get('partial_epoch_smoke'):
            raise ValueError(f'{original}: pretrain verification is not strict')
        return None, upstream
    if matches[0] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
        return job, upstream
    raise RuntimeError(f'{original}: pretrain state is {matches[0]}')


def stage_arm(campaign, arm, expected_encoder, expected_position):
    original = campaign / 'accounts' / OWNER / arm
    wait_for, upstream = dependency(original)
    config = yaml.safe_load((original / 'configs/pretrain.yaml').read_text())
    if (config['masking']['mask_ratio'] != .55 or
            config['mae']['decoder_depth'] != 2 or
            config['ablation']['encoder'] != expected_encoder or
            config['ablation']['position'] != expected_position):
        raise ValueError(f'Unexpected mask55/depth2/arm configuration: {original}')
    stage = campaign / 'accounts' / OWNER / 'hmc_siena_downstream' / arm
    if not stage.exists():
        stage.mkdir(parents=True)
        (stage / 'configs').mkdir()
        (stage / 'source').symlink_to(ROOT, target_is_directory=True)
        (stage / 'pretrain').symlink_to(original / 'pretrain', target_is_directory=True)
        shutil.copy2(original / 'configs/pretrain.yaml', stage / 'configs/pretrain.yaml')
        manifest = dict(
            experiment=stage.name, preset=arm, owner=OWNER,
            source_pretrain=str(original), pretrain_job=upstream['pretrain_entries'][0]['job'],
            pretrain_source_commit=upstream['source_commit'], datasets=list(DATASETS),
            fixed_hyperparameters=FIXED,
            selection='Per-seed validation balanced accuracy; test reporting only',
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = read_json(stage / 'manifest.json')
        if (manifest.get('source_pretrain') != str(original) or
                manifest.get('fixed_hyperparameters') != FIXED):
            raise ValueError(f'Existing stage has different provenance or HP: {stage}')
    return stage, wait_for


def submit(stage, wait_for):
    manifest = read_json(stage / 'manifest.json')
    jobs = manifest.get('downstream_jobs', [])
    if jobs and len(jobs) != len(DATASETS):
        raise RuntimeError(f'Partial submission requires inspection: {stage}')
    if not jobs:
        command = [sys.executable, str(ROOT / 'scripts/submit_experiment_downstream.py'),
                   '--experiment', str(stage), '--datasets', *DATASETS]
        if wait_for:
            command += ['--dependency', wait_for]
        subprocess.run(command, check=True)
    manifest = read_json(stage / 'manifest.json')
    if not manifest.get('finalizer_job'):
        subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                        '--experiment', str(stage), '--submit'], check=True)
    current = read_json(stage / 'manifest.json')
    print(json.dumps(dict(arm=stage.name,
                          jobs={row['dataset']: row['job'] for row in current['downstream_jobs']},
                          finalizer=current.get('finalizer_job')), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    args = parser.parse_args()
    if getpass.getuser() != OWNER:
        parser.error('This launcher must be run by yc8820')
    for arm, (encoder, position) in ARMS.items():
        stage, wait_for = stage_arm(args.campaign.resolve(), arm, encoder, position)
        submit(stage, wait_for)


if __name__ == '__main__':
    main()
