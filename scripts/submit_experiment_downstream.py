"""Submit five-seed arrays immediately with native Slurm dependencies."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import yaml

SEEDS = (42, 696, 1001, 1234, 3407)
DATASETS = ('chb', 'siena', 'physionet_mi', 'tuev', 'tuab', 'faced', 'seedv',
            'mentalarithmetic', 'isruc', 'hmc', 'tusl', 'tusz')
TEMPLATE_PREFIX = {'tusz': 'nearest3_7', 'tusl': 'nearest3_7'}
MAIN_A100 = {'gr2-mjde-d4-geometry', 'gr2-d2-static', 'gr2-d2-patch-scalar', 'gr2-d2-patch-dimension',
             'gr2-d2-patch-dimension-mask60', 'gr2-d4-patch-dimension-mask60'}


def warmup(value):
    if isinstance(value, dict):
        return any('warmup' in str(k).lower() or warmup(v) for k, v in value.items())
    return any(warmup(v) for v in value) if isinstance(value, list) else False


def resource_policy(source, preset, dataset):
    cluster = yaml.safe_load((source / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    if preset in MAIN_A100:
        result = dict(cluster['downstream'])
        if dataset in ('chb', 'tuab', 'tuev', 'faced', 'isruc', 'hmc'):
            result.update(partitions='a100_short,a100_long', time='12:00:00')
    else:
        policy = yaml.safe_load((source / 'configs/cluster/downstream_l40s.yaml').read_text())
        result = {k: v for k, v in policy.items() if k != 'datasets'}
        result.update(policy['datasets'][dataset])
    return result


def prepare(experiment, checkpoint):
    source = experiment / 'source'
    config_dir = experiment / 'configs/downstream'
    config_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            prefix = TEMPLATE_PREFIX.get(dataset, 'gr9-1')
            template = source / ('configs/downstream/%s_%s_seed%d.yaml' % (prefix, dataset, seed))
            config = yaml.safe_load(template.read_text())
            if warmup(config):
                raise ValueError('Downstream warmup is forbidden: ' + str(template))
            opt = config['optimization']
            if len({opt[k + '_learning_rate'] for k in ('tokenizer', 'encoder', 'head')}) != 1:
                raise ValueError('Expected common downstream learning rate: ' + str(template))
            if dataset == 'tusz':
                opt['class_counts'] = [28670, 12842]
                opt['label_smoothing'] = 0.0
            output = experiment / 'downstream' / dataset / ('seed-%d' % seed)
            config['model']['checkpoint'] = str(checkpoint)
            config['runtime']['output'] = str(output)
            config['data']['num_workers'] = 2
            path = config_dir / ('%s_seed%d.yaml' % (dataset, seed))
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            entries.append(dict(dataset=dataset, seed=seed, config=str(path), output=str(output)))
    (experiment / 'downstream_entries.json').write_text(json.dumps(entries, indent=2) + '\n')
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--account')
    parser.add_argument('--dependency', help='Successful pretrain Slurm job ID')
    parser.add_argument('--hold', action='store_true', help='Legacy explicit hold; new launcher uses afterok')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    source = experiment / 'source'
    checkpoint = (args.checkpoint or experiment / 'pretrain/checkpoint-epoch-0040.pth').resolve()
    manifest_path = experiment / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('downstream_jobs') or manifest.get('downstream_job'):
        raise ValueError('Downstream already submitted; inspect manifest before retrying')
    if args.dependency and not re.fullmatch(r'\d+', args.dependency):
        raise ValueError('Expected numeric pretrain dependency job ID')
    if not (args.prepare_only or args.hold or args.dependency) and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    entries = prepare(experiment, checkpoint)
    if args.prepare_only:
        return
    cluster = yaml.safe_load((source / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    logs = experiment / 'downstream/logs'
    logs.mkdir(parents=True, exist_ok=True)
    manifest.update(downstream_jobs=[], downstream_dependency=args.dependency,
                    downstream_state='dependency' if args.dependency else 'held' if args.hold else 'submitted')
    # Record each ID before attempting the next submission.
    for dataset in DATASETS:
        policy = resource_policy(source, manifest.get('preset'), dataset)
        indices = [i for i, e in enumerate(entries) if e['dataset'] == dataset]
        command = ['sbatch', '--parsable', '--account=' + (args.account or cluster['account']),
                   '--job-name=' + experiment.name + '-' + dataset,
                   '--partition=' + policy['partitions'], '--nodes=1', '--ntasks=1',
                   '--gpus-per-task=' + policy['gpu'] + ':1', '--cpus-per-task=' + str(policy['cpus_per_task']),
                   '--mem=' + policy['memory'], '--time=' + policy['time'],
                   '--array=' + ','.join(map(str, indices)) + '%' + str(policy['array_parallelism']),
                   '--output=' + str(logs / '%A_%a.out'), '--error=' + str(logs / '%A_%a.err'),
                   '--wrap=exec ' + shlex.join(['srun', '--ntasks=1', '--gpus-per-task=' + policy['gpu'] + ':1',
                                              '--gpu-bind=single:1', '--kill-on-bad-exit=1', sys.executable, str(source / 'scripts/downstream_experiment_worker.py'),
                                              '--experiment', str(experiment)])]
        if policy.get('excluded_nodes'):
            command.append('--exclude=' + ','.join(policy['excluded_nodes']))
        if args.dependency:
            command += ['--dependency=afterok:' + args.dependency, '--kill-on-invalid-dep=yes']
        if args.hold:
            command.append('--hold')
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        manifest['downstream_jobs'].append(dict(job=job, dataset=dataset, indices=indices, resources=policy,
                                                duration_basis='scheduling_estimate_not_five_seed_mean'))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(job, flush=True)


if __name__ == '__main__':
    main()
