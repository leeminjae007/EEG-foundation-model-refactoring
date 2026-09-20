"""Resume unfinished TUAB seeds under the currently logged-in Slurm user.

Running source jobs remain in place. A single CPU result audit provides a stable
afterok dependency for the original owner's TUSZ arrays.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SEEDS = {42, 696, 1001, 1234, 3407}
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
EXTRA_EXCLUDED = {'a100-4035', 'a100-4046', 'a100-4047'}


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def user_name():
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def job_states(jobs):
    if not jobs:
        return {}
    bases = sorted({str(job).split('_')[0] for job in jobs})
    if not all(re.fullmatch(r'\d+', job) for job in bases):
        raise ValueError('Invalid source job ID')
    output = subprocess.check_output(['sacct', '--array', '-X', '-j', ','.join(bases),
                                      '--format=JobID%40,State%30', '-P', '-n'], text=True)
    return {parts[0]: parts[1].split()[0] for line in output.splitlines()
            if len(parts := line.strip().split('|')) >= 2 and parts[1]}


def decision(state, readable_result):
    if state in ACTIVE:
        return 'reuse_running'
    if state == 'PENDING':
        raise ValueError('Source seed is still pending; cancel it under its owner before transferring')
    if state not in TERMINAL:
        raise ValueError('Cannot establish whether the source seed is active: ' + str(state))
    return 'reuse_complete' if readable_result else 'resume'


def validate_resume_config(saved, desired):
    normalized = copy.deepcopy(desired)
    normalized['runtime']['output'] = saved['runtime']['output']
    if normalized != saved:
        raise ValueError('Resume would change training configuration; only runtime.output relocation is allowed')


def readable_result(output):
    path = Path(output) / 'result.json'
    if not path.exists():
        return False
    from scripts.monitor_experiment_results import selector_test
    selector_test(json.loads(path.read_text()))
    return True


def prepare(source_campaign, output_root):
    import torch
    import yaml
    if torch.__version__.split('+')[0] != '2.0.1':
        raise ValueError('Activate eeg-foundation-model-cu118 (PyTorch 2.0.1) before preparing a handoff')
    source_campaign = source_campaign.resolve()
    arms = json.loads((source_campaign / 'arms.json').read_text())
    if len(arms) != 4:
        raise ValueError('Expected the four GR2 TUAB campaigns')
    source_entries = {arm['alias']: json.loads((Path(arm['folder']) / 'entries.json').read_text()) for arm in arms}
    states = job_states([e['job'] for entries in source_entries.values() for e in entries])
    selected = []
    for alias, entries in source_entries.items():
        if len(entries) != 5 or {e['seed'] for e in entries} != SEEDS:
            raise ValueError(alias + ': expected the fixed five seeds')
        if not re.fullmatch(r'[a-z0-9-]+', alias):
            raise ValueError('Invalid campaign alias')
        for entry in entries:
            action = decision(states.get(entry['job']), readable_result(entry['output']))
            selected.append((alias, entry, action))
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M%S')
    experiment = output_root.resolve() / 'outputs/experiments' / (stamp + '-tuab-handoff')
    experiment.mkdir(parents=True, exist_ok=False)
    os.chmod(experiment, 0o755)
    source = experiment / 'source'
    source.mkdir()
    archive = experiment / 'source.tar'
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar', '--output=' + str(archive), 'HEAD'], check=True)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(source)], check=True)
    if not (source / 'scripts/run_tuab_resume_seed.py').exists():
        raise ValueError('Commit or pull the handoff launcher before using it')
    prepared = {alias: [] for alias in source_entries}
    for alias, original, action in selected:
        entry = dict(original, handoff_action=action, source_job=original['job'])
        folder = experiment / alias
        (folder / 'configs').mkdir(parents=True, exist_ok=True)
        (folder / 'logs').mkdir(exist_ok=True)
        if action != 'resume':
            # Early permission checks also cover previously private validation weights.
            for name in ['validation.jsonl', 'best-balanced_accuracy.pth', 'last.pth']:
                with (Path(entry['output']) / name).open('rb') as stream:
                    stream.read(1)
            prepared[alias].append(entry)
            continue
        old_output = Path(entry['output'])
        resume = old_output / 'last.pth'
        if not resume.exists():
            resume = Path(entry['resume'])
        config = yaml.safe_load(Path(entry['config']).read_text())
        saved = torch.load(resume, map_location='cpu')
        if config['data']['dataset'] != 'tuab' or config['seed'] != entry['seed']:
            raise ValueError('Source dataset/seed mismatch')
        validate_resume_config(saved['config'], config)
        if saved['extra'].get('partial_epoch_smoke') or not 17 <= saved['epoch'] <= 20:
            raise ValueError('Expected a complete TUAB epoch-17/18/19/20 checkpoint')
        if not saved['optimizer']['state'] or not saved['scheduler'] or not saved['rng_states']:
            raise ValueError('Checkpoint lacks complete optimizer/scheduler/RNG state')
        # Resume in place: the source campaign's downstream/tuab directory is
        # the authoritative location for its checkpoints, logs, and result.json.
        # The handoff folder holds only controller metadata/config snapshots.
        output = old_output
        if not output.is_dir():
            raise ValueError('Missing original TUAB output directory: ' + str(output))
        file_hashes = {}
        for name in ['initialization.json', 'best-balanced_accuracy.pth', 'best-auroc.pth']:
            path = old_output / name
            if not path.is_file():
                raise ValueError('Missing original TUAB artifact: ' + str(path))
            file_hashes[name] = digest(path)
        discarded = {}
        for name in ['metrics.jsonl', 'validation.jsonl']:
            rows = [json.loads(line) for line in (old_output / name).read_text().splitlines()]
            kept = [r for r in rows if r['epoch'] <= saved['epoch'] and r.get('step', 0) <= saved['extra']['step']]
            # Discard rows written after the last durable checkpoint before
            # resuming, so the original log remains internally consistent.
            (output / name).write_text(''.join(json.dumps(r) + '\n' for r in kept))
            discarded[name] = len(rows) - len(kept)
        config['runtime']['output'] = str(output)
        validate_resume_config(saved['config'], config)
        path = folder / 'configs' / ('tuab_seed%d.yaml' % entry['seed'])
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        entry.update(config=str(path), output=str(output), resume=str(resume), job=None,
                     resume_epoch=saved['epoch'], resume_step=saved['extra']['step'],
                     copied_file_sha256=file_hashes, discarded_uncommitted_log_rows=discarded)
        del saved
        prepared[alias].append(entry)
    for alias, entries in prepared.items():
        write(experiment / alias / 'entries.json', entries)
    tusz = json.loads((source_campaign / 'tusz_dependencies.json').read_text())
    owner_info = subprocess.check_output(['scontrol', 'show', 'job', tusz['jobs'][0], '-o'], text=True)
    match = re.search(r'UserId=([^\s(]+)\((\d+)\)', owner_info)
    if not match:
        raise ValueError('Could not establish TUSZ owner')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), created_at_new_york=stamp,
                    source_campaign=str(source_campaign), source_commit=commit, user=user_name(),
                    uid=os.getuid(), experiment=str(experiment), python=sys.executable,
                    publication_root=str(output_root.resolve() / 'outputs/results'),
                    aliases=list(prepared), status='prepared', downstream_jobs=[],
                    tusz_jobs=tusz['jobs'], tusz_owner=match.group(1), tusz_owner_uid=int(match.group(2)),
                    old_controller=tusz['controller_job'],
                    resume_seeds=sum(e['handoff_action'] == 'resume' for es in prepared.values() for e in es),
                    reused_seeds=sum(e['handoff_action'] != 'resume' for es in prepared.values() for e in es))
    write(experiment / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['experiment', 'user', 'resume_seeds', 'reused_seeds']}, indent=2), flush=True)
    return experiment


def submit(experiment):
    import yaml
    manifest_path = experiment / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['downstream_jobs'] or manifest.get('completion_job'):
        raise ValueError('Handoff already submitted; do not duplicate it')
    source = experiment / 'source'
    cluster = yaml.safe_load((source / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    exclude = sorted(set(cluster['downstream']['excluded_nodes']) | EXTRA_EXCLUDED)
    dependencies = set()
    for alias in manifest['aliases']:
        path = experiment / alias / 'entries.json'
        entries = json.loads(path.read_text())
        indices = [i for i, e in enumerate(entries) if e['handoff_action'] == 'resume']
        dependencies.update(e['job'] for e in entries if e['handoff_action'] == 'reuse_running')
        if not indices:
            continue
        # Recheck immediately before submission, including new job states since preparation.
        original_states = job_states([entries[i]['source_job'] for i in indices])
        if any(original_states.get(entries[i]['source_job']) not in TERMINAL for i in indices):
            raise ValueError('A source seed became active; refusing duplicate GPU submission')
        command = ['sbatch', '--parsable', '--hold', '--account=' + cluster['account'],
                   '--job-name=tuab-handoff-' + alias, '--partition=a100_dev,a100_short,a100_long',
                   '--nodes=1', '--ntasks=1', '--gpus-per-task=a100:1', '--cpus-per-task=2',
                   '--mem=32G', '--time=04:00:00', '--array=' + ','.join(map(str, indices)) + '%5',
                   '--exclude=' + ','.join(exclude), '--chdir=' + str(source),
                   '--output=' + str(experiment / alias / 'logs/%A_%a.out'),
                   '--error=' + str(experiment / alias / 'logs/%A_%a.err'),
                   '--wrap=exec ' + shlex.join(['srun', '--ntasks=1', '--gpus-per-task=a100:1',
                                               '--gpu-bind=single:1', '--kill-on-bad-exit=1', sys.executable,
                                               str(source / 'scripts/run_tuab_resume_seed.py'), str(experiment), alias])]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        for i in indices:
            entries[i]['job'] = job + '_' + str(i)
        write(path, entries)
        manifest['downstream_jobs'].append(dict(alias=alias, job=job, indices=indices, command=command))
        write(manifest_path, manifest)
        dependencies.add(job)
    audit_command = ['sbatch', '--parsable', '--account=' + cluster['account'], '--job-name=tuab-handoff-results',
                     '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
                     '--mem=4G', '--time=01:00:00', '--output=' + str(experiment / 'audit-%j.out'),
                     '--error=' + str(experiment / 'audit-%j.err'),
                     '--wrap=export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec ' +
                     shlex.join([sys.executable, str(source / 'scripts/launch_tuab_handoff.py'), 'audit', str(experiment)])]
    if dependencies:
        audit_command += ['--dependency=afterany:' + ':'.join(sorted(dependencies))]
    completion = subprocess.check_output(audit_command, text=True).strip().split(';')[0]
    manifest.update(completion_job=completion, completion_command=audit_command, status='submitted')
    write(manifest_path, manifest)
    for item in manifest['downstream_jobs']:
        subprocess.run(['scontrol', 'release', item['job']], check=True)
    print('Completion audit: ' + completion, flush=True)
    print('Run this command as ' + manifest['tusz_owner'] + ' to reconnect its TUSZ jobs:', flush=True)
    print(shlex.join(['python', str(REPO / 'scripts/launch_tuab_handoff.py'), 'attach-tusz', str(manifest_path)]), flush=True)


def attach_tusz(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if os.getuid() != manifest['tusz_owner_uid']:
        raise PermissionError('Run attach-tusz as ' + manifest['tusz_owner'] + '; Slurm cannot modify another user\'s jobs')
    completion = str(manifest['completion_job'])
    if not re.fullmatch(r'\d+', completion):
        raise ValueError('Invalid completion job ID')
    for job in manifest['tusz_jobs']:
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        if 'JobState=RUNNING' in info or 'JobState=PENDING' not in info:
            raise ValueError('TUSZ is no longer entirely pending: ' + job)
    for job in manifest['tusz_jobs']:
        subprocess.run(['scontrol', 'update', 'JobId=' + job, 'Dependency=afterok:' + completion,
                        'ArrayTaskThrottle=5'], check=True)
    old = subprocess.run(['scontrol', 'show', 'job', manifest['old_controller'], '-o'], text=True, capture_output=True)
    if old.returncode == 0 and 'JobState=PENDING' in old.stdout:
        subprocess.run(['scancel', manifest['old_controller']], check=True)
    origin = Path(manifest['source_campaign'])
    write(origin / 'handoff_attached.json', dict(manifest=str(manifest_path), completion_job=completion,
                                               tusz_jobs=manifest['tusz_jobs'], attached_utc=datetime.now(timezone.utc).isoformat()))
    print('TUSZ now waits for all handoff TUAB results: afterok:' + completion)


def audit(experiment):
    from scripts.audit_tuab_handoff import audit_and_publish
    return audit_and_publish(experiment)


def launch_once(source_campaign, destination, prepare_only=False):
    if prepare_only:
        return prepare(source_campaign, destination)
    import fcntl
    registry = destination.resolve() / 'outputs/handoffs'
    registry.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(source_campaign.resolve()).encode()).hexdigest()[:20]
    claim = registry / (key + '.json')
    with (registry / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if claim.exists():
            raise ValueError('This account already prepared/submitted this handoff. Inspect ' + str(claim))
        write(claim, dict(source_campaign=str(source_campaign.resolve()), status='preparing'))
        experiment = prepare(source_campaign, destination)
        write(claim, dict(source_campaign=str(source_campaign.resolve()), experiment=str(experiment), status='prepared'))
        submit(experiment)
        write(claim, dict(source_campaign=str(source_campaign.resolve()), experiment=str(experiment), status='submitted'))
    return experiment


def main():
    os.umask(0o022)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    launch = sub.add_parser('launch')
    launch.add_argument('source_campaign', type=Path)
    launch.add_argument('--output-root', type=Path)
    launch.add_argument('--prepare-only', action='store_true')
    attach = sub.add_parser('attach-tusz')
    attach.add_argument('manifest', type=Path)
    check = sub.add_parser('audit')
    check.add_argument('experiment', type=Path)
    args = parser.parse_args()
    if args.action == 'launch':
        destination = args.output_root or Path('/gpfs/data/oermannlab/users') / user_name() / 'workspace/eegfm'
        launch_once(args.source_campaign, destination, args.prepare_only)
    elif args.action == 'attach-tusz':
        attach_tusz(args.manifest)
    else:
        return audit(args.experiment)
    return 0


if __name__ == '__main__':
    sys.exit(main())
