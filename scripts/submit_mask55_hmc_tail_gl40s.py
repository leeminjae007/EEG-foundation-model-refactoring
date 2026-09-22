"""Submit the remaining HMC tail candidates on GL40S in user-specified reverse order."""

from __future__ import annotations

try:
    import fcntl
except ModuleNotFoundError:  # Ordering tests run on Windows.
    fcntl = None
import getpass
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/l40s-earlystop-replacements/260921-1738-gr2-d2-mask55-isruc-hmc-siena-grid')
ORDER = tuple(reversed((
    '6cbaf220df54', 'da808bb22b09', '8b32e74c89f0', '1bd05a85ab3c',
    '6336d1a757da', 'd38862e4af39', 'd39599feaab6', '155fee595931',
    'e442d48ce5d0', '4e1c6fe44d51', '7ebe26faae62',
)))
SEEDS = {42, 696, 1001, 1234, 3407}
BAD_NODES = ('gl40s-8013',)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def complete(entry):
    output = Path(entry['output'])
    try:
        selected = read_json(output / 'result.json')['balanced_accuracy']
        test = selected['test']
        return (math.isfinite(float(selected['selection']['score'])) and
                all(math.isfinite(float(test[name]))
                    for name in ('balanced_accuracy', 'weighted_f1', 'kappa')) and
                (output / 'best-balanced_accuracy.pth').is_file() and
                (output / 'last.pth').is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def submit():
    if getpass.getuser() != 'ml10266':
        raise PermissionError('This recovery submission belongs to ml10266')
    manifest_path = CAMPAIGN / 'manifest.json'
    with (CAMPAIGN / 'hmc-tail-gl40s.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_json(manifest_path)
        entries = read_json(CAMPAIGN / 'downstream_entries.json')
        groups = {cid: [row for row in entries
                        if row['dataset'] == 'hmc' and row['candidate'] == cid]
                  for cid in ORDER}
        for cid, rows in groups.items():
            if len(rows) != 5 or {row['seed'] for row in rows} != SEEDS:
                raise ValueError(f'{cid}: expected five frozen HMC seeds')
        existing = manifest.setdefault('hmc_tail_gl40s_jobs', [])
        covered = {row['candidate'] for row in existing}
        dependency = str(existing[-1]['job']) if existing else None
        logs = CAMPAIGN / 'logs'
        logs.mkdir(exist_ok=True)
        for rank, cid in enumerate(ORDER, 1):
            if cid in covered:
                continue
            pending = [row for row in groups[cid] if not complete(row)]
            if not pending:
                continue
            command = ['sbatch', '--parsable', '--account=system',
                       f'--job-name=mask55-hmc-tail-{rank:02d}',
                       '--partition=gl40s_long', '--nodes=1', '--ntasks=1',
                       '--gpus-per-task=l40s:1', '--cpus-per-task=2', '--mem=32G',
                       '--time=08:00:00',
                       '--array=' + ','.join(str(row['index']) for row in pending) + '%5',
                       '--exclude=' + ','.join(BAD_NODES),
                       '--chdir=' + str(CAMPAIGN / 'source'),
                       '--output=' + str(logs / 'hmc-tail-%A_%a.out'),
                       '--error=' + str(logs / 'hmc-tail-%A_%a.err')]
            if dependency:
                command.append('--dependency=afterany:' + dependency)
            command.append('--wrap=exec ' + shlex.join([
                'srun', '--ntasks=1', '--gpus-per-task=l40s:1', '--gpu-bind=single:1',
                '--kill-on-bad-exit=1', sys.executable,
                str(CAMPAIGN / 'source/scripts/downstream_experiment_worker.py'),
                '--experiment', str(CAMPAIGN), '--source', str(CAMPAIGN / 'source')]))
            job = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
            existing.append(dict(rank=rank, candidate=cid, hp=pending[0]['hp'], job=job,
                                 indices=[row['index'] for row in pending],
                                 dependency=('afterany:' + dependency) if dependency else None,
                                 resources=dict(gpu='l40s', partition='gl40s_long',
                                                time='08:00:00', excluded_nodes=list(BAD_NODES))))
            dependency = job
            write_json(manifest_path, manifest)
        if not manifest.get('hmc_tail_gl40s_finalizer'):
            finalizer = subprocess.check_output([
                'sbatch', '--parsable', '--account=system',
                '--job-name=mask55-hmc-tail-results', '--partition=cpu_short,cpu_long',
                '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                '--time=00:30:00', '--dependency=afterany:' + dependency,
                '--output=' + str(logs / 'hmc-tail-results-%j.out'),
                '--error=' + str(logs / 'hmc-tail-results-%j.err'),
                '--wrap=exec ' + shlex.join([sys.executable, str(ROOT / 'scripts/submit_mask55_hmc_tail_gl40s.py'),
                                             'aggregate'])], text=True).strip().split(';', 1)[0]
            manifest['hmc_tail_gl40s_finalizer'] = finalizer
        manifest['hmc_tail_gl40s_status'] = 'submitted'
        write_json(manifest_path, manifest)
        print(json.dumps(dict(order=list(ORDER), jobs=existing,
                              finalizer=manifest['hmc_tail_gl40s_finalizer']), indent=2))


def aggregate():
    entries = read_json(CAMPAIGN / 'downstream_entries.json')
    rows, missing = [], []
    for cid in ORDER:
        group = [entry for entry in entries
                 if entry['dataset'] == 'hmc' and entry['candidate'] == cid]
        completed = []
        for entry in group:
            if not complete(entry):
                missing.append(dict(candidate=cid, seed=entry['seed']))
                continue
            selected = read_json(Path(entry['output']) / 'result.json')['balanced_accuracy']
            completed.append(dict(
                seed=entry['seed'], validation_bacc=float(selected['selection']['score']),
                test={name: float(selected['test'][name])
                      for name in ('balanced_accuracy', 'weighted_f1', 'kappa')}))
        if len(completed) == 5:
            rows.append(dict(candidate=cid, hp=group[0]['hp'],
                validation_bacc_mean=statistics.mean(row['validation_bacc'] for row in completed),
                test={name: dict(mean=statistics.mean(row['test'][name] for row in completed),
                                 sd=statistics.pstdev(row['test'][name] for row in completed))
                      for name in ('balanced_accuracy', 'weighted_f1', 'kappa')}))
    summary = dict(status='complete' if not missing else 'incomplete',
                   candidate_summaries=rows, missing=missing,
                   winner_by_validation=(max(rows, key=lambda row: row['validation_bacc_mean'])
                                         if rows else None),
                   test_used_for_selection=False)
    write_json(CAMPAIGN / 'hmc_tail_gl40s_summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('submit', 'aggregate'):
        raise SystemExit('usage: submit_mask55_hmc_tail_gl40s.py {submit|aggregate}')
    sys.exit(submit() if sys.argv[1] == 'submit' else aggregate())
