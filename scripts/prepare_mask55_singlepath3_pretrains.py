"""Owner-prepared, yc8820-submitted mask-55 S2T/T2S three-stage pretrains."""

from __future__ import annotations

import argparse
from copy import deepcopy
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_shared_mask55_pretrains as shared

RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
SUFFIX = '-mask55-d2-singlepath3-pretrain'
ACCOUNT = 'yc8820'
ARMS = {'s2t3': 'mjde_s2t3', 't2s3': 'mjde_t2s3'}


def resolved_configs():
    from ablation.config import resolve_ablation

    base = yaml.safe_load((ROOT / 'configs/pretrain_gr2_patch_feature.yaml').read_text())
    configs = {}
    for slug, encoder in ARMS.items():
        config = deepcopy(base)
        config['masking']['mask_ratio'] = .55
        config['mae']['decoder_depth'] = 2
        # A single path has no S2T/T2S fusion operation. Keep every other
        # baseline setting, but mark the former patch-feature gate as N/A.
        config['encoder']['fusion_gate'] = 'static_feature'
        config['ablation'] = dict(
            name='mask55_d2_patchdim_' + slug, encoder=encoder,
            position='shpe', pe_scope='both',
            reference_fusion_gate='patch_feature',
            fusion_gate_applicability='not_applicable_single_path')
        config = resolve_ablation(config)
        assert config['masking']['policy'] == 'geometry_tubelet'
        assert config['masking']['mask_ratio'] == .55
        assert config['mae']['decoder_depth'] == 2
        assert config['optimization']['epochs'] == 40
        configs[slug] = config
    return configs


def prepare(results_root=RESULTS):
    if getpass.getuser() != shared.OWNER:
        raise PermissionError('Only ml10266 prepares the owner-hosted campaign')
    matches = sorted(results_root.glob('*' + SUFFIX))
    if matches:
        raise FileExistsError('Matching campaign already exists: ' + ', '.join(map(str, matches)))
    configs = resolved_configs()
    policy = yaml.safe_load((ROOT / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    commit = shared.source_commit()
    campaign = results_root / (shared.new_york_stamp() + SUFFIX)
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / 'source'
    shared.snapshot(source)
    entries = []
    for slug, config in configs.items():
        arm = campaign / 'accounts' / ACCOUNT / slug
        for name in ('configs', 'pretrain', 'logs'):
            (arm / name).mkdir(parents=True, exist_ok=True)
        (arm / 'source').symlink_to(source, target_is_directory=True)
        config['runtime']['output'] = str(arm / 'pretrain')
        config_path = arm / 'configs/pretrain.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        entry = dict(kind='pretrain', arm='mask55-' + slug, config=str(config_path),
                     config_sha256=shared.digest(config_path), source=str(source),
                     result_dir=str(arm / 'pretrain'), job=None, job_history=[], retries=0,
                     last_resume_epoch=0, max_timeout_resumes=40,
                     time_limit=policy['pretrain']['time'],
                     gpu_partitions=policy['pretrain']['partitions'],
                     timeout_continuation=False)
        manifest = dict(experiment='mask55-d2-singlepath3-' + slug,
                        owner=shared.OWNER, assigned_account=ACCOUNT,
                        source_commit=commit, python=None,
                        created_at_new_york=campaign.name[:11],
                        preset='mask55-d2-singlepath3-' + slug,
                        pretrain_launcher='slurm_flexible',
                        pretrain_entries=[entry],
                        pretrain_excluded_nodes=policy['pretrain']['excluded_nodes'],
                        resource_policy=policy['pretrain'], auto_resume=False,
                        verify_before_success=True,
                        results_dir=str(arm / 'pretrain/report'), status='prepared',
                        downstream_submitted=False,
                        architecture='one original three-stage ' + slug.upper() + ' path; six blocks',
                        fusion_gate_applicability='not_applicable_single_path')
        shared.write_json(arm / 'manifest.json', manifest)
        worker = arm / 'worker.sh'
        worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                          'exec "${EEGFM_PYTHON:?activate the training environment before sbatch}" '
                          + str(source / 'scripts/gr2_pretrain_campaign.py')
                          + ' worker --folder ' + str(arm) + '\n', encoding='utf-8')
        worker.chmod(0o755)
        access = shared.grant_account(arm, ACCOUNT)
        manifest['collaboration_access'] = access
        shared.write_json(arm / 'manifest.json', manifest)
        (arm / 'manifest.json').chmod(0o666)
        entries.append(dict(slug=slug, assigned_account=ACCOUNT,
                            arm_folder=str(arm), config=str(config_path),
                            config_sha256=entry['config_sha256'],
                            collaboration_access=access))
    (campaign / 'accounts' / ACCOUNT).chmod(0o777)
    shared.write_json(campaign / 'manifest.json', dict(
        experiment=campaign.name, owner=shared.OWNER,
        source_commit=commit, source=str(source),
        assignments={ACCOUNT: list(ARMS)},
        settings=dict(mask_ratio=.55, decoder_depth=2,
                      encoder_path_stages=3, encoder_path_blocks=6,
                      fusion_gate_applicability='not_applicable_single_path',
                      epochs=40, seed=42, gpu='a100', tasks=4,
                      downstream=False), resources=policy['pretrain'],
        entries=entries, status='prepared'))
    return campaign


def submit(campaign):
    if getpass.getuser() != ACCOUNT:
        raise PermissionError('Only yc8820 submits these A100 pretrains')
    shared.require_environment()
    top = json.loads((campaign / 'manifest.json').read_text())
    if top['assignments'] != {ACCOUNT: list(ARMS)} or not shared.checkout_contains(top['source_commit']):
        raise ValueError('Campaign assignment or owner checkout commit differs')
    jobs = {}
    controller = campaign / 'source/scripts/gr2_pretrain_campaign.py'
    for slug in ARMS:
        arm = campaign / 'accounts' / ACCOUNT / slug
        manifest_path = arm / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        entry = manifest['pretrain_entries'][0]
        if entry['job']:
            jobs[slug] = entry['job']
            continue
        if (shared.digest(entry['config']) != entry['config_sha256'] or
                manifest['assigned_account'] != ACCOUNT or
                manifest['source_commit'] != top['source_commit']):
            raise ValueError('Frozen campaign mismatch: ' + slug)
        config = yaml.safe_load(Path(entry['config']).read_text())
        if (config['ablation']['encoder'] != ARMS[slug] or
                config['masking']['mask_ratio'] != .55 or
                config['mae']['decoder_depth'] != 2):
            raise ValueError('Frozen model config mismatch: ' + slug)
        manifest['python'] = sys.executable
        shared.write_json(manifest_path, manifest)
        environment = dict(os.environ, EEGFM_PYTHON=sys.executable)
        job = subprocess.check_output([sys.executable, str(controller), 'submit',
                                       '--folder', str(arm)], text=True,
                                      env=environment).strip().splitlines()[-1].split(';', 1)[0]
        jobs[slug] = job
    shared.write_json(campaign / 'accounts' / ACCOUNT / 'submission.json',
                      dict(account=ACCOUNT, python=sys.executable, jobs=jobs,
                           downstream_submitted=False))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit'))
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare())
        return
    if args.campaign is None:
        parser.error('submit requires --campaign')
    print(json.dumps(submit(args.campaign.resolve()), indent=2))


if __name__ == '__main__':
    main()
