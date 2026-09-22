"""Retry only the failed shared mask-55 four-task downstream arrays.

Run from the collaborator's activated environment in the owner's checkout:
    python scripts/retry_shared_mask55_four.py pe-ch_order
    python scripts/retry_shared_mask55_four.py csbrain
"""

import argparse
import getpass
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

try:
    import fcntl
except ModuleNotFoundError:  # Windows unit tests; BigPurple always has fcntl.
    fcntl = None


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260921-1456-mask55-d2-patchdim-shared-pretrain')
ARMS = {'pe-ch_order': ('hk4935', ('chb', 'faced', 'physionet_mi', 'mentalarithmetic')),
        'csbrain': ('yc8820', ('chb',))}
SEEDS = {42, 696, 1001, 1234, 3407}
CONFIG_DATASETS = {'chb': 'chb', 'faced': 'faced',
                   'physionet_mi': 'physio', 'mentalarithmetic': 'stress'}
TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                   'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def complete(entry):
    output = Path(entry['output'])
    result = output / 'result.json'
    if not result.is_file():
        return False
    try:
        payload = read_json(result)
        selected = payload['balanced_accuracy']
        test = selected['test']
        metrics = ('balanced_accuracy', 'auroc', 'auprc') if 'auroc' in test else ('balanced_accuracy', 'weighted_f1', 'kappa')
        return (math.isfinite(selected['selection']['score']) and
                all(math.isfinite(test[name]) for name in metrics) and
                (output / 'best-balanced_accuracy.pth').is_file() and
                (output / 'last.pth').is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def require_original_finished(job, missing_indices):
    """squeue rejects expired IDs; sacct must prove every seed is terminal."""
    query = subprocess.run(['squeue', '-h', '-r', '-j', job, '-o', '%i|%T'],
                           capture_output=True, text=True)
    if query.returncode and 'Invalid job id specified' not in query.stderr:
        raise RuntimeError('Cannot inspect original Slurm job ' + job + ': ' + query.stderr.strip())
    active = {int(line.split('|')[0].rsplit('_', 1)[1]) for line in query.stdout.splitlines()
              if line.split('|')[0].rsplit('_', 1)[-1].isdigit()}
    if active.intersection(missing_indices):
        raise RuntimeError('Original job ' + job + ' still has active missing seeds')
    history = subprocess.check_output(
        ['sacct', '-X', '-n', '-P', '-j', job, '--format=JobID,State'], text=True)
    states = {}
    for line in history.splitlines():
        fields = line.split('|')
        if len(fields) < 2 or not fields[0].startswith(job + '_'):
            continue
        suffix = fields[0][len(job) + 1:]
        if suffix.isdigit():
            states[int(suffix)] = fields[1].split()[0]
    unsafe = {index: states.get(index, 'MISSING_FROM_SACCT') for index in missing_indices
              if states.get(index) not in TERMINAL_STATES}
    if unsafe:
        raise RuntimeError('Original Slurm tasks are not proven terminal: ' + repr(unsafe))


def submit(stage, arm, targets):
    manifest_path = stage / 'manifest.json'
    with (stage / 'retry-submit.lock').open('a+') as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_json(manifest_path)
        if manifest.get('source_pretrain') != str(CAMPAIGN / 'accounts' / ARMS[arm][0] / arm):
            raise ValueError('Stage points to another pretrain')
        verified = read_json(stage / 'pretrain/verified.json')
        if not Path(verified['checkpoint']).is_file():
            raise FileNotFoundError(verified['checkpoint'])
        entries = read_json(stage / 'downstream_entries.json')
        jobs = {job['dataset']: job for job in manifest['downstream_jobs']}
        if len(entries) != 20 or set(jobs) != {'chb', 'faced', 'physionet_mi', 'mentalarithmetic'}:
            raise ValueError('Expected the original four five-seed downstream arrays')
        repairs = manifest.setdefault('recovery_jobs', [])
        logs = stage / 'downstream/logs'
        logs.mkdir(parents=True, exist_ok=True)
        cluster = yaml.safe_load((ROOT / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
        for dataset in targets:
            rows = [(i, e) for i, e in enumerate(entries) if e['dataset'] == dataset]
            if len(rows) != 5 or {e['seed'] for _, e in rows} != SEEDS:
                raise ValueError(dataset + ': expected exactly five fixed seeds')
            if any(job['dataset'] == dataset for job in repairs):
                print(dataset + ': recovery already submitted; no duplicate')
                continue
            missing = [(i, e) for i, e in rows if not complete(e)]
            if not missing:
                print(dataset + ': already complete')
                continue
            require_original_finished(str(jobs[dataset]['job']), {i for i, _ in missing})
            for _, entry in missing:
                config = yaml.safe_load(Path(entry['config']).read_text(encoding='utf-8'))
                if (config['data']['dataset'] != CONFIG_DATASETS[dataset] or
                        config['seed'] != entry['seed'] or
                        Path(config['runtime']['output']) != Path(entry['output']) or
                        Path(config['model']['checkpoint']).resolve() != Path(verified['checkpoint']).resolve()):
                    raise ValueError(dataset + ': frozen config mismatch for seed ' + str(entry['seed']))
            resources = jobs[dataset]['resources']
            if resources['gpu'] != 'l40s':
                raise ValueError('Recovery must stay on L40S')
            indices = [i for i, _ in missing]
            command = ['sbatch', '--parsable', '--account=' + cluster['account'],
                       '--job-name=' + arm + '-retry-' + dataset,
                       '--partition=' + resources['partitions'], '--nodes=1', '--ntasks=1',
                       '--gpus-per-task=l40s:1', '--cpus-per-task=' + str(resources['cpus_per_task']),
                       '--mem=' + resources['memory'], '--time=' + resources['time'],
                       '--array=' + ','.join(map(str, indices)) + '%5',
                       '--output=' + str(logs / 'retry-%A_%a.out'),
                       '--error=' + str(logs / 'retry-%A_%a.err')]
            if resources.get('excluded_nodes'):
                command.append('--exclude=' + ','.join(resources['excluded_nodes']))
            command.append('--wrap=exec ' + shlex.join([
                'srun', '--ntasks=1', '--gpus-per-task=l40s:1', '--gpu-bind=single:1',
                '--kill-on-bad-exit=1', sys.executable,
                str(ROOT / 'scripts/downstream_experiment_worker.py'),
                '--experiment', str(stage), '--source', str(ROOT)]))
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            repairs.append({'dataset': dataset, 'job': job, 'indices': indices, 'resources': resources})
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            print(dataset + ': submitted ' + job + ' indices=' + ','.join(map(str, indices)), flush=True)
        if repairs and not manifest.get('recovery_finalizer_job'):
            dependency = ':'.join(job['job'] for job in repairs)
            command = ['sbatch', '--parsable', '--account=' + cluster['account'],
                       '--job-name=' + arm + '-retry-results', '--partition=cpu_short,cpu_long',
                       '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                       '--time=00:30:00', '--dependency=afterany:' + dependency,
                       '--output=' + str(logs / 'retry-results-%j.out'),
                       '--error=' + str(logs / 'retry-results-%j.err'),
                       '--wrap=export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec ' +
                       shlex.join([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                                   '--experiment', str(stage)])]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            manifest['recovery_finalizer_job'] = job
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            print('finalizer=' + job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=ARMS)
    args = parser.parse_args()
    owner, targets = ARMS[args.arm]
    if getpass.getuser() != owner:
        parser.error(args.arm + ' must be submitted by ' + owner)
    submit(CAMPAIGN / 'accounts' / owner / 'fourtask_downstream' / args.arm,
           args.arm, targets)


if __name__ == '__main__':
    main()
