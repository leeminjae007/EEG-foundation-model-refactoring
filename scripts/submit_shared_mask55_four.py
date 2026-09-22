"""Submit the four selected downstream datasets for one shared pretrain group.

Run from a collaborator's activated environment in ml10266's shared checkout:
    python scripts/submit_shared_mask55_four.py pe
    python scripts/submit_shared_mask55_four.py encoder
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1456-mask55-d2-patchdim-shared-pretrain')
DATASETS = ('chb', 'faced', 'physionet_mi', 'mentalarithmetic')
GROUPS = {'pe': (('hk4935', 'pe-ch_order', 'mjde', 'channel_id'),
                 ('hk4935', 'pe-acpe', 'mjde', 'acpe'),
                 ('hk4935', 'pe-4dREVE', 'mjde', 'reve4d')),
          'encoder': (('yc8820', 'cbramod', 'cbramod', 'shpe'),
                      ('yc8820', 'csbrain', 'csbrain', 'shpe'),
                      ('yc8820', 'labram', 'labram', 'shpe'))}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pretrain_dependency(original):
    manifest = read_json(original / 'manifest.json')
    if (original / 'pretrain/verified.json').is_file():
        return None, manifest
    job = manifest['pretrain_entries'][0]['job']
    state = subprocess.check_output(
        ['sacct', '-X', '-j', job, '--format=JobIDRaw,State', '-P', '-n'], text=True)
    matching = [line.split('|')[1].split()[0] for line in state.splitlines()
                if line.split('|')[0] == job]
    if len(matching) != 1 or matching[0] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
        raise RuntimeError('%s: pretrain is not active and has no verified checkpoint: %s' % (original, state))
    return job, manifest


def prepare_stage(campaign, owner, arm, encoder, position):
    original = campaign / 'accounts' / owner / arm
    dependency, upstream = pretrain_dependency(original)
    config = yaml.safe_load((original / 'configs/pretrain.yaml').read_text(encoding='utf-8'))
    if (config['masking']['mask_ratio'] != 0.55 or
            config['ablation']['encoder'] != encoder or
            config['ablation']['position'] != position):
        raise ValueError('Unexpected mask/architecture/PE setting: ' + str(original))
    stage = campaign / 'accounts' / owner / 'fourtask_downstream' / arm
    if stage.exists():
        staged = read_json(stage / 'manifest.json')
        if staged['source_pretrain'] != str(original):
            raise ValueError('Existing stage points to another pretrain: ' + str(stage))
        return stage, dependency
    stage.mkdir(parents=True)
    (stage / 'configs').mkdir()
    (stage / 'source').symlink_to(ROOT, target_is_directory=True)
    (stage / 'pretrain').symlink_to(original / 'pretrain', target_is_directory=True)
    shutil.copy2(original / 'configs/pretrain.yaml', stage / 'configs/pretrain.yaml')
    manifest = dict(experiment=stage.name, preset=arm,
                    created_at_new_york=campaign.name[:11],
                    publication_root=str(campaign / 'outputs/results'),
                    source_pretrain=str(original), pretrain_job=upstream['pretrain_entries'][0]['job'],
                    pretrain_source_commit=upstream['source_commit'],
                    downstream_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return stage, dependency


def submit(stage, dependency):
    manifest = read_json(stage / 'manifest.json')
    jobs = manifest.get('downstream_jobs', [])
    if jobs and len(jobs) != len(DATASETS):
        raise RuntimeError('Partial submission; inspect the recorded jobs before retrying: ' + str(stage))
    if not jobs:
        command = [sys.executable, str(ROOT / 'scripts/submit_experiment_downstream.py'),
                   '--experiment', str(stage), '--datasets', *DATASETS]
        if dependency:
            command += ['--dependency', dependency]
        subprocess.run(command, check=True)
    manifest = read_json(stage / 'manifest.json')
    if not manifest.get('finalizer_job'):
        subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                        '--experiment', str(stage), '--submit'], check=True)
    print('%s: %s' % (stage.name, ', '.join(
        '%s=%s' % (entry['dataset'], entry['job']) for entry in read_json(stage / 'manifest.json')['downstream_jobs'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    args = parser.parse_args()
    for owner, arm, encoder, position in GROUPS[args.group]:
        stage, dependency = prepare_stage(args.campaign, owner, arm, encoder, position)
        submit(stage, dependency)


if __name__ == '__main__':
    main()
