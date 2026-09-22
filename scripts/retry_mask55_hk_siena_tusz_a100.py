"""Exclude unhealthy A100 nodes, probe CUDA, then retry failed HK grid seeds.

Run once from hk4935's activated environment in the owner's checkout. Re-run
only to capture additional original-array failures after the first invocation.
"""

from __future__ import annotations

try:
    import fcntl
except ModuleNotFoundError:  # Unit tests also run on Windows.
    fcntl = None
import getpass
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_shared_mask55_four import complete
from scripts.submit_mask55_hk_siena_tusz_grid import CAMPAIGN

TERMINAL = {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL',
            'PREEMPTED', 'DEADLINE', 'COMPLETED'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def task_states(job):
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
    query = subprocess.run(['squeue', '-h', '-r', '-j', job, '-o', '%i|%T'],
                           capture_output=True, text=True)
    if query.returncode and 'Invalid job id specified' not in query.stderr:
        raise RuntimeError('Cannot inspect Slurm job ' + job + ': ' + query.stderr.strip())
    for line in query.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 2 or not fields[0].startswith(job + '_'):
            continue
        suffix = fields[0][len(job) + 1:]
        if suffix.isdigit():
            states[int(suffix)] = fields[1]
    return states


def retryable_indices(entries, dataset, original_states, prior):
    selected = []
    for entry in entries:
        if entry['dataset'] != dataset or entry['index'] in prior or complete(entry):
            continue
        if original_states.get(entry['index']) in TERMINAL:
            selected.append(entry['index'])
    return selected


def exclude_pending_original(job, states, nodes):
    pending = [index for index, state in states.items() if state == 'PENDING']
    if not pending:
        return
    subprocess.run(['scontrol', 'update', 'JobId=' + job,
                    'ExcNodeList=' + ','.join(nodes)], check=True)
    # Verify one pending array element, rather than trusting only exit code.
    shown = subprocess.check_output(['scontrol', 'show', 'job',
                                     f'{job}_{pending[0]}'], text=True)
    if 'ExcNodeList=(null)' in shown or 'ExcNodeList=' not in shown:
        raise RuntimeError('Slurm did not retain the excluded nodes for ' + job)


def submit_probe(logs, nodes, account):
    code = ('import os,subprocess; from scripts.downstream_gpu_guard import allocated_gpu; '
            'query=subprocess.check_output(["nvidia-smi",'
            '"--query-gpu=uuid,memory.total,memory.free",'
            '"--format=csv,noheader,nounits"],text=True); '
            'health=allocated_gpu(query,12288); '
            'os.environ.update(CUDA_VISIBLE_DEVICES=health["gpu_uuid"],CUDA_DEVICE_ORDER="PCI_BUS_ID"); '
            'print(health,flush=True); import torch; '
            'torch.ones(1,device="cuda:0").sum().item(); '
            'torch.cuda.synchronize(); print("CUDA_PROBE_PASSED",flush=True)')
    command = ['sbatch', '--parsable', '--account=' + account,
               '--job-name=hk-mask55-a100-cuda-probe',
               '--partition=a100_dev,a100_short,a100_long', '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=8G',
               '--time=00:10:00', '--exclude=' + ','.join(nodes),
               '--chdir=' + str(ROOT),
               '--output=' + str(logs / 'cuda-probe-%j.out'),
               '--error=' + str(logs / 'cuda-probe-%j.err'),
               '--wrap=exec ' + shlex.join([
                   'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                   '--kill-on-bad-exit=1', sys.executable, '-c', code])]
    return subprocess.check_output(command, text=True).strip().split(';', 1)[0]


def submit_retry(campaign, dataset, indices, probe, nodes, resources, account):
    logs = campaign / 'logs'
    command = ['sbatch', '--parsable', '--account=' + account,
               '--job-name=mask55-hk-retry-' + dataset,
               '--partition=' + resources['partitions'], '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G',
               '--time=' + resources['time_limits'][dataset],
               '--array=' + ','.join(map(str, indices)) + '%5',
               '--dependency=afterok:' + probe, '--exclude=' + ','.join(nodes),
               '--chdir=' + str(campaign / 'source'),
               '--output=' + str(logs / (dataset + '-retry-%A_%a.out')),
               '--error=' + str(logs / (dataset + '-retry-%A_%a.err')),
               '--wrap=exec ' + shlex.join([
                   'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                   '--kill-on-bad-exit=1', sys.executable,
                   str(campaign / 'source/scripts/downstream_experiment_worker.py'),
                   '--experiment', str(campaign), '--source', str(campaign / 'source')])]
    return subprocess.check_output(command, text=True).strip().split(';', 1)[0]


def submit_finalizer(campaign, dependencies, account):
    logs = campaign / 'logs'
    command = ['sbatch', '--parsable', '--account=' + account,
               '--job-name=mask55-hk-retry-results', '--partition=cpu_short,cpu_long',
               '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
               '--time=00:30:00', '--dependency=afterany:' + ':'.join(dependencies),
               '--output=' + str(logs / 'retry-results-%j.out'),
               '--error=' + str(logs / 'retry-results-%j.err'),
               '--wrap=exec ' + shlex.join([sys.executable,
                   str(campaign / 'source/scripts/submit_mask55_hk_siena_tusz_grid.py'),
                   'aggregate', '--campaign', str(campaign)])]
    return subprocess.check_output(command, text=True).strip().split(';', 1)[0]


def main():
    if getpass.getuser() != 'hk4935':
        raise PermissionError('Only hk4935 can update and retry these jobs')
    campaign = CAMPAIGN
    manifest_path = campaign / 'manifest.json'
    with (campaign / 'recovery-submit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_json(manifest_path)
        if (manifest.get('owner') != 'hk4935' or
                manifest.get('checkpoint_sha256') !=
                'a5ab908bde519241b199efcc701ea1352b3fcd9da068f600549845bef65babca' or
                len(manifest['jobs']) != 2):
            raise ValueError('Unexpected campaign; refusing to modify Slurm jobs')
        jobs = {row['dataset']: str(row['job']) for row in manifest['jobs']}
        if set(jobs) != {'siena', 'tusz'}:
            raise ValueError('Expected exactly Siena and TUSZ arrays')
        entries = read_json(campaign / 'downstream_entries.json')
        if len(entries) != 360 or {row['index'] for row in entries} != set(range(360)):
            raise ValueError('Expected 360 frozen indexed runs')
        cluster = yaml.safe_load((ROOT / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
        nodes = sorted(set(cluster['pretrain']['excluded_nodes']))
        if not {'a100-4011', 'a100-4024', 'a100-4034', 'a100-4045'}.issubset(nodes):
            raise ValueError('Known CUDA-failing A100 nodes are not excluded')
        original_states = {dataset: task_states(job) for dataset, job in jobs.items()}
        for dataset, job in jobs.items():
            exclude_pending_original(job, original_states[dataset], nodes)
        previous = manifest.setdefault('recovery_jobs', [])
        prior_indices = {index for row in previous for index in row['indices']}
        missing = {dataset: retryable_indices(entries, dataset,
                                             original_states[dataset], prior_indices)
                   for dataset in jobs}
        if not any(missing.values()):
            print(json.dumps(dict(message='No new terminal missing seeds to retry',
                                  prior_recovery_jobs=previous), indent=2))
            return
        logs = campaign / 'logs'
        logs.mkdir(exist_ok=True)
        probe = manifest.get('recovery_probe_job')
        if probe:
            probe_states = subprocess.check_output(
                ['sacct', '-X', '-n', '-P', '-j', probe, '--format=JobID,State'],
                text=True)
            probe_state = next((line.split('|', 1)[1].split()[0]
                                for line in probe_states.splitlines()
                                if line.split('|', 1)[0] == probe), None)
            if probe_state in TERMINAL and probe_state != 'COMPLETED':
                probe = None
        if not probe:
            probe = submit_probe(logs, nodes, cluster['account'])
            manifest['recovery_probe_job'] = probe
            write_json(manifest_path, manifest)
        for dataset, indices in missing.items():
            if not indices:
                continue
            job = submit_retry(campaign, dataset, indices, probe, nodes,
                               manifest['resources'], cluster['account'])
            previous.append(dict(dataset=dataset, job=job, indices=indices,
                                 dependency='afterok:' + probe, excluded_nodes=nodes))
            write_json(manifest_path, manifest)
        dependencies = list(jobs.values()) + [row['job'] for row in previous]
        finalizer = submit_finalizer(campaign, dependencies, cluster['account'])
        manifest['recovery_finalizer_job'] = finalizer
        write_json(manifest_path, manifest)
        print(json.dumps(dict(cuda_probe_job=probe, retry_jobs=previous,
                              finalizer_job=finalizer, excluded_nodes=nodes), indent=2))


if __name__ == '__main__':
    main()
