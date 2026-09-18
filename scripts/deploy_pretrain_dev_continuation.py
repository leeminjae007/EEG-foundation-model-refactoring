"""One-time server migration of the two existing pretrain campaigns.

Change queued allocations in place and replace CPU controllers/callbacks only.
Training configurations, source snapshots, GPU job IDs and user holds are kept.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from datetime import datetime, timezone


ROOT = Path('/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model')
STAGE = ROOT / 'outputs/controller_update_20260917_2010'
PYTHON = str(ROOT / '.venv/bin/python')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def query_job(job):
    value = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9/:]*)=(\S*)', value))
    if fields.get('JobId') != str(job) or fields.get('UserId', '').split('(')[0] != 'ml10266':
        raise ValueError('Unexpected owner/job: ' + str(job))
    return fields


def stop_cpu(job, allowed_prefix):
    current = query_job(job)
    if not current.get('JobName', '').startswith(allowed_prefix):
        raise ValueError('Unexpected CPU job: ' + str(current))
    if 'gres/gpu' in current.get('ReqTRES', ''):
        raise ValueError('Refusing to cancel a GPU allocation')
    if current['JobState'] in ('PENDING', 'RUNNING', 'CONFIGURING'):
        subprocess.run(['scancel', str(job)], check=True)
    return current


def deploy(folder_name, module_name, monitor_key, monitor_prefix, monitor_memory):
    folder = ROOT / 'outputs' / folder_name
    backup = STAGE / ('before-' + module_name)
    backup.mkdir(exist_ok=True)
    report = {'campaign': str(folder), 'checked_utc': datetime.now(timezone.utc).isoformat(), 'entries': []}
    with (folder / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((folder / 'manifest.json').read_text())
        atomic_json(backup / 'manifest.json', manifest)
        # Install only orchestration files. Config/source fingerprints remain intact.
        files = [module_name + '.py']
        if module_name == 'mjde12_campaign':
            files.append('monitor_experiment_results.py')
        for name in files:
            target = folder / 'controller' / name
            (backup / name).write_bytes(target.read_bytes())
            pending = target.with_suffix('.deploy.tmp')
            pending.write_bytes((STAGE / 'scripts' / name).read_bytes())
            pending.replace(target)
        spec = importlib.util.spec_from_file_location(module_name, folder / 'controller' / (module_name + '.py'))
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)

        old_monitor = manifest[monitor_key]
        report['previous_monitor'] = stop_cpu(old_monitor, monitor_prefix)
        manifest.setdefault('retired_cpu_monitor_jobs', []).append(old_monitor)
        manifest['monitor_interval_seconds'] = 3600

        for entry in manifest['pretrain_entries']:
            before = query_job(entry['job'])
            if before['JobState'] != 'PENDING':
                raise ValueError('Expected queued pretrain: ' + str(before))
            if before.get('Command') != str(folder / 'worker.sh'):
                raise ValueError('Unexpected GPU command')
            subprocess.run(['scontrol', 'update', 'JobId=' + entry['job'],
                            'TimeLimit=04:00:00', 'Partition=a100_dev,a100_short,a100_long'], check=True)
            after = query_job(entry['job'])
            if before.get('Reason') == 'JobHeldUser' and after.get('Reason') != 'JobHeldUser':
                raise ValueError('User hold was lost')
            entry.update(gpu_partitions='a100_dev,a100_short,a100_long', time_limit='04:00:00',
                         timeout_continuation=True, max_timeout_resumes=40)
            previous_callback = entry.get('continuation_job')
            if previous_callback:
                stop_cpu(previous_callback, 'gr2-resume-' if module_name.startswith('gr2') else 'mj12-resume-')
                entry.setdefault('retired_continuation_jobs', []).append(previous_callback)
            entry.pop('continuation_for_job', None)
            entry.pop('continuation_job', None)
            # Persist the resource policy before registering the callback.
            controller.write_json(folder / 'manifest.json', manifest)
            callback = controller.attach_continuation(folder, manifest, entry)
            report['entries'].append({'arm': entry['arm'], 'job': entry['job'], 'before': before,
                                      'after': after, 'continuation_job': callback,
                                      'callback': query_job(callback)})

        command = shlex.join([PYTHON, str(folder / 'controller' / (module_name + '.py')),
                              'monitor', '--folder', str(folder)])
        job = subprocess.check_output([
            'sbatch', '--parsable', '--account=system', '--partition=cpu_long',
            '--job-name=' + monitor_prefix, '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
            '--mem=' + monitor_memory, '--time=28-00:00:00',
            '--output=' + str(folder / 'logs/controller-%j.out'),
            '--error=' + str(folder / 'logs/controller-%j.err'),
            '--wrap=export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1; exec ' + command,
        ], text=True).strip().split(';')[0]
        manifest[monitor_key] = job
        manifest['continuation_policy'] = {
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'target_epochs': 40, 'allocation_hours': 4,
            'partitions': 'a100_dev,a100_short,a100_long',
            'nodes': '1-4', 'total_gpus': 4, 'cpu_per_gpu': 4, 'ram_gib_per_gpu': 32,
            'terminal_states': ['TIMEOUT', 'PREEMPTED', 'COMPLETED'],
            'requires_committed_epoch_progress': True, 'max_timeout_resumes': 40,
            'afterany_cpu_callback': True, 'hourly_fallback': True,
            'preserve_existing_holds': True,
        }
        controller.write_json(folder / 'manifest.json', manifest)
        report['monitor_job'] = job
        atomic_json(folder / 'dev_continuation_migration.json', report)
        atomic_json(STAGE / (module_name + '-deployment.json'), report)
    return report


if __name__ == '__main__':
    # Configure the held ablations before GR2 can start and release their holds.
    results = [deploy('mjde12_pretrain_first_20260917_0600', 'mjde12_campaign',
                      'cpu_monitor_job', 'mj12-hour-watch', '8G')]
    results.append(deploy('mjde_gr2_geometry_20260917_171816', 'gr2_pretrain_campaign',
                          'monitor_job', 'gr2-monitor', '4G'))
    atomic_json(STAGE / 'deployment.json', results)
    print(json.dumps([{'campaign': r['campaign'], 'monitor_job': r['monitor_job'],
                      'entries': [{k: e[k] for k in ('arm', 'job', 'continuation_job')} for e in r['entries']]}
                     for r in results], indent=2))
