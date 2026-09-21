"""Submit a 5-minute full-batch pretrain and 12 real-data downstream checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path, help='Isolated --prepare-only experiment')
    parser.add_argument('--gpu', choices=('a100', 'l40s'), default='a100', help='Diagnostic GPU type; production stays A100')
    args = parser.parse_args()
    folder = args.experiment.resolve()
    path = folder / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('pretrain_job') or manifest.get('smoke_seconds'):
        raise ValueError('Use a fresh prepare-only experiment for smoke checks')
    manifest.update(smoke_seconds=300, smoke_gpu=args.gpu, verify_before_success=False, auto_resume=False)
    entry = manifest['pretrain_entries'][0]
    entry.update(time_limit='00:20:00', timeout_continuation=False)
    if args.gpu == 'l40s':
        entry['gpu_partitions'] = 'gl40s_dev,gl40s_short,gl40s_long'
        manifest['pretrain_excluded_nodes'] = []
    cfg = yaml.safe_load(Path(entry['config']).read_text())
    profiles = {
        'gr2-d2-patch-dimension-mask60': (2, 'patch_feature', .6),
        'gr2-d4-patch-dimension-mask55': (4, 'patch_feature', .55),
        'gr2-d4-patch-dimension-mask60': (4, 'patch_feature', .6),
    }
    expected = profiles.get(manifest.get('preset'))
    if expected is None:
        raise ValueError('Smoke profile is not defined for preset: ' + str(manifest.get('preset')))
    depth, gate, ratio = expected
    assert cfg['mae']['decoder_depth'] == depth and cfg['encoder']['fusion_gate'] == gate
    assert cfg['masking']['mask_ratio'] == ratio and cfg['optimization']['batch_size_per_gpu'] == 128
    cfg['runtime']['log_every_steps'] = 5
    Path(entry['config']).write_text(yaml.safe_dump(cfg, sort_keys=False))
    entry['config_sha256'] = hashlib.sha256(Path(entry['config']).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    source = folder / 'source'
    for e in json.loads((folder / 'downstream_entries.json').read_text()):
        cpath = Path(e['config'])
        config = yaml.safe_load(cpath.read_text())
        config['model']['checkpoint'] = str(folder / 'pretrain/last.pth')
        config['runtime']['log_every_steps'] = 1
        cpath.write_text(yaml.safe_dump(config, sort_keys=False))
    job = subprocess.check_output([sys.executable, str(folder / 'controller/gr2_pretrain_campaign.py'),
                                   'submit', '--folder', str(folder)], text=True).strip().splitlines()[-1]
    logs = folder / 'downstream/logs'
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(source / 'scripts/downstream_experiment_worker.py'),
               '--experiment', str(folder), '--smoke-batches', '4']
    smoke_partition = 'a100_short,a100_long' if args.gpu == 'a100' else 'gl40s_short,gl40s_long'
    downstream = subprocess.check_output(['sbatch', '--parsable', '--account=system', '--job-name=patch-dim-ds-smoke',
        '--partition=' + smoke_partition, '--nodes=1', '--ntasks=1', '--gpus-per-task=' + args.gpu + ':1',
        '--cpus-per-task=2', '--mem=32G', '--time=00:30:00',
        '--array=' + ','.join(str(i) for i in range(0, 60, 5)) + '%4',
        '--dependency=afterok:' + job, '--kill-on-invalid-dep=yes',
        '--output=' + str(logs / 'smoke-%A_%a.out'), '--error=' + str(logs / 'smoke-%A_%a.err'),
        '--wrap=exec ' + shlex.join(['srun', '--ntasks=1', '--gpus-per-task=' + args.gpu + ':1', '--gpu-bind=single:1',
                                    '--kill-on-bad-exit=1'] + command)], text=True).strip().split(';')[0]
    manifest = json.loads(path.read_text())
    manifest['smoke_downstream_job'] = downstream
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(experiment=str(folder), pretrain=job, downstream=downstream)))


if __name__ == '__main__':
    main()
