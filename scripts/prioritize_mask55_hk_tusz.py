"""Cancel HK Siena and finish the existing TUSZ grid high-to-low, without lost results.

Run as hk4935 from the owner's checkout with the PyTorch environment active:
    python scripts/prioritize_mask55_hk_tusz.py submit
"""

from __future__ import annotations

import argparse
try:
    import fcntl
except ModuleNotFoundError:  # Pure priority-order tests run on Windows.
    fcntl = None
import getpass
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_shared_mask55_four import complete
from scripts.submit_mask55_hk_siena_tusz_grid import CAMPAIGN, EXPECTED_SHA256, SEEDS
from scripts.submit_mask55_hp_grid import digest, write_json


def priority_groups(entries):
    groups = {}
    for entry in entries:
        if entry['dataset'] == 'tusz':
            groups.setdefault(entry['candidate'], []).append(entry)
    if len(groups) != 36 or any(len(group) != 5 for group in groups.values()):
        raise ValueError('Expected 36 TUSZ candidates with five seeds each')
    return sorted(groups.items(), key=lambda item: (
        -item[1][0]['hp']['learning_rate'],
        -item[1][0]['hp']['weight_decay'],
        -item[1][0]['hp']['dropout'], item[0]))


def cancel_job(job, *, pending_only=False):
    command = ['scancel']
    if pending_only:
        command += ['--state=PENDING']
    command += [str(job)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode and 'Invalid job id specified' not in result.stderr:
        raise RuntimeError(f'Cannot cancel {job}: {result.stderr.strip()}')


def submit():
    if getpass.getuser() != 'hk4935':
        raise PermissionError('Only hk4935 can cancel or submit these jobs')
    manifest_path = CAMPAIGN / 'manifest.json'
    with (CAMPAIGN / 'tusz-priority.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        jobs = {row['dataset']: str(row['job']) for row in manifest['jobs']}
        if (manifest['owner'] != 'hk4935' or set(jobs) != {'siena', 'tusz'} or
                manifest['checkpoint_sha256'] != EXPECTED_SHA256):
            raise ValueError('Unexpected HK campaign; refusing Slurm changes')
        entries = json.loads((CAMPAIGN / 'downstream_entries.json').read_text())
        if len(entries) != 360 or {e['index'] for e in entries} != set(range(360)):
            raise ValueError('Frozen 360-entry grid changed')
        groups = priority_groups(entries)
        if digest(manifest['checkpoint']) != EXPECTED_SHA256:
            raise ValueError('Mask-55 depth-2 checkpoint hash changed')
        cluster = yaml.safe_load((ROOT / 'configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
        nodes = sorted(set(cluster['pretrain']['excluded_nodes']))
        if not {'a100-4011', 'a100-4024', 'a100-4034', 'a100-4045'}.issubset(nodes):
            raise ValueError('Known bad A100 nodes are not excluded')

        if not manifest.get('siena_priority_cancelled'):
            siena_jobs = [jobs['siena']] + [str(row['job']) for row in manifest.get('recovery_jobs', [])
                                               if row['dataset'] == 'siena']
            for job in siena_jobs:
                cancel_job(job)
            # Existing combined aggregators would incorrectly expect Siena to finish.
            for key in ('aggregate_job', 'recovery_finalizer_job'):
                if manifest.get(key):
                    cancel_job(manifest[key])
            manifest['siena_priority_cancelled'] = siena_jobs
            manifest['active_datasets'] = ['tusz']
            write_json(manifest_path, manifest)

        tusz_jobs = [jobs['tusz']] + [str(row['job']) for row in manifest.get('recovery_jobs', [])
                                          if row['dataset'] == 'tusz']
        if not manifest.get('tusz_pending_priority_cancelled'):
            for job in tusz_jobs:
                cancel_job(job, pending_only=True)
            manifest['tusz_pending_priority_cancelled'] = tusz_jobs
            write_json(manifest_path, manifest)

        existing = manifest.setdefault('tusz_priority_jobs', [])
        covered = {row['candidate'] for row in existing}
        dependency = str(existing[-1]['job']) if existing else ':'.join(tusz_jobs)
        logs = CAMPAIGN / 'logs'
        logs.mkdir(exist_ok=True)
        for rank, (candidate, group) in enumerate(groups, 1):
            if candidate in covered:
                continue
            pending = [e for e in group if not complete(e)]
            if not pending:
                continue
            # afterany enforces candidate waves in descending LR, WD, dropout.
            command = ['sbatch', '--parsable', '--account=' + cluster['account'],
                       f'--job-name=mask55-hk-tusz-prio-{rank:02d}',
                       '--partition=' + manifest['resources']['partitions'],
                       '--nodes=1', '--ntasks=1', '--gpus-per-task=a100:1',
                       '--cpus-per-task=2', '--mem=32G',
                       '--time=' + manifest['resources']['time_limits']['tusz'],
                       '--array=' + ','.join(str(e['index']) for e in pending) + '%5',
                       '--dependency=afterany:' + dependency,
                       '--exclude=' + ','.join(nodes),
                       '--chdir=' + str(CAMPAIGN / 'source'),
                       '--output=' + str(logs / 'tusz-priority-%A_%a.out'),
                       '--error=' + str(logs / 'tusz-priority-%A_%a.err'),
                       '--wrap=exec ' + shlex.join([
                           'srun', '--ntasks=1', '--gpus-per-task=a100:1',
                           '--gpu-bind=single:1', '--kill-on-bad-exit=1', sys.executable,
                           str(ROOT / 'scripts/run_priority_mask55_hk_tusz.py'),
                           '--campaign', str(CAMPAIGN)])]
            job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
            existing.append(dict(rank=rank, candidate=candidate, job=job,
                                 indices=[e['index'] for e in pending], dependency=dependency))
            dependency = job
            manifest['status'] = 'tusz_priority_submitting'
            write_json(manifest_path, manifest)

        if not manifest.get('tusz_priority_finalizer'):
            finalizer = subprocess.check_output([
                'sbatch', '--parsable', '--account=' + cluster['account'],
                '--job-name=mask55-hk-tusz-priority-results',
                '--partition=cpu_short,cpu_long', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=1', '--mem=4G', '--time=00:30:00',
                '--dependency=afterany:' + dependency,
                '--output=' + str(logs / 'tusz-priority-results-%j.out'),
                '--error=' + str(logs / 'tusz-priority-results-%j.err'),
                '--wrap=exec ' + shlex.join([sys.executable, str(ROOT / 'scripts/prioritize_mask55_hk_tusz.py'),
                                             'aggregate'])], text=True).strip().split(';', 1)[0]
            manifest['tusz_priority_finalizer'] = finalizer
        manifest['status'] = 'tusz_priority_submitted'
        write_json(manifest_path, manifest)
        print(json.dumps(dict(cancelled_siena=manifest['siena_priority_cancelled'],
                              preserved_completed_tusz=sum(complete(e) for e in entries if e['dataset'] == 'tusz'),
                              priority_first=[dict(candidate=cid, hp=group[0]['hp'])
                                              for cid, group in groups[:3]],
                              priority_jobs=len(existing), first_job=existing[0]['job'] if existing else None,
                              last_job=dependency,
                              finalizer=manifest['tusz_priority_finalizer']), indent=2))


def aggregate():
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    entries = json.loads((CAMPAIGN / 'downstream_entries.json').read_text())
    summaries, missing = [], []
    for candidate, group in priority_groups(entries):
        rows = []
        for entry in group:
            if not complete(entry) or digest(entry['config']) != entry['config_sha256']:
                missing.append(dict(candidate=candidate, seed=entry['seed']))
                continue
            result = json.loads((Path(entry['output']) / 'result.json').read_text())['balanced_accuracy']
            validation = float(result['selection']['score'])
            test = {name: float(result['test'][name]) for name in ('balanced_accuracy', 'auroc', 'auprc')}
            if not math.isfinite(validation) or not all(math.isfinite(v) for v in test.values()):
                missing.append(dict(candidate=candidate, seed=entry['seed']))
                continue
            rows.append(dict(seed=entry['seed'], validation_bacc=validation, test=test))
        if len(rows) == 5 and {row['seed'] for row in rows} == set(SEEDS):
            summaries.append(dict(candidate=candidate, hp=group[0]['hp'],
                validation_bacc_mean=statistics.mean(row['validation_bacc'] for row in rows),
                test={metric: dict(mean=statistics.mean(row['test'][metric] for row in rows),
                                   sd=statistics.pstdev(row['test'][metric] for row in rows))
                      for metric in ('balanced_accuracy', 'auroc', 'auprc')}))
    winner = max(summaries, key=lambda row: (row['validation_bacc_mean'], row['candidate'])) if summaries else None
    summary = dict(status='complete' if not missing and len(summaries) == 36 else 'incomplete',
                   completed_candidates=len(summaries), missing=missing,
                   winner_by_validation=winner, test_used_for_selection=False)
    write_json(CAMPAIGN / 'tusz_priority_summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'aggregate'))
    args = parser.parse_args()
    if args.action == 'submit':
        submit()
    else:
        return aggregate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
