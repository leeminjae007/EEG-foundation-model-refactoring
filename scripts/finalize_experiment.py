"""One-shot afterany audit/publication. No polling and no GPU allocation."""
import argparse
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import monitor_experiment_results as publisher

NAMES = dict(zip(('chb', 'siena', 'physionet_mi', 'tuev', 'tuab', 'faced', 'seedv', 'mentalarithmetic', 'isruc', 'hmc', 'tusl', 'tusz'),
                 ('CHB-MIT', 'SIENA', 'PHYSIONET-MI', 'TUEV', 'TUAB', 'FACED', 'SEED-V', 'Mental Arithmetic', 'ISRUC', 'HMC', 'TUSL', 'TUSZ')))


def collect(experiment):
    import torch
    import yaml
    entries = json.loads((experiment / 'downstream_entries.json').read_text())
    results, missing, timings = {}, [], []
    for entry in entries:
        output = Path(entry['output'])
        try:
            config = yaml.safe_load(Path(entry['config']).read_text())
            payload = json.loads((output / 'result.json').read_text())
            values = publisher.selector_test(payload)
            required = ('balanced_accuracy', 'auroc', 'auprc') if 'auroc' in values else ('balanced_accuracy', 'weighted_f1', 'kappa')
            if not all(k in values and math.isfinite(values[k]) for k in required):
                raise ValueError('Missing required finite test metrics')
            saved = torch.load(output / 'last.pth', map_location='cpu')
            if saved['extra'].get('partial_epoch_smoke') or saved['epoch'] != config['optimization']['epochs'] or saved['config'] != config:
                raise ValueError('Incomplete epochs, smoke checkpoint, or config mismatch')
            epoch = payload['balanced_accuracy']['selection']['epoch']
            validation = [json.loads(line) for line in (output / 'validation.jsonl').read_text().splitlines()]
            chosen = [r for r in validation if r['epoch'] == epoch]
            if len(chosen) != 1 or chosen[0]['balanced_accuracy'] != payload['balanced_accuracy']['selection']['score']:
                raise ValueError('Validation selector mismatch')
            weights = torch.load(output / 'best-balanced_accuracy.pth', map_location='cpu')
            if set(weights) != set(saved['model']) or any(weights[k].shape != v.shape or not torch.isfinite(weights[k]).all() for k, v in saved['model'].items()):
                raise ValueError('Invalid validation-selected weights')
            results[str(output).rstrip('/') + '/result.json'] = payload
            del saved, weights
            if (output / 'runtime.json').is_file():
                timings.append(json.loads((output / 'runtime.json').read_text()))
        except Exception as exc:
            missing.append(dict(dataset=entry['dataset'], seed=entry['seed'], error=repr(exc)))
    for dataset in sorted({e['dataset'] for e in entries}, key=publisher.dataset_sort_key):
        group = [e for e in entries if e['dataset'] == dataset]
        if len(group) != 5 or {e['seed'] for e in group} != publisher.SEEDS:
            missing.append(dict(dataset=dataset, error='Expected exactly the fixed five seeds'))
    return entries, results, missing, timings


def finalize(experiment):
    experiment = Path(experiment).resolve()
    manifest = json.loads((experiment / 'manifest.json').read_text())
    if manifest.get('smoke_seconds'):
        raise ValueError('Smoke campaigns cannot publish production results')
    entries, results, missing, timings = collect(experiment)
    state = dict(status='incomplete' if missing else 'complete', missing=missing,
                 readable_results=len(results), expected=len(entries))
    publisher.atomic_text(experiment / 'completion.json', json.dumps(state, indent=2) + '\n')
    runtime = []
    for dataset in sorted({e['dataset'] for e in entries}, key=publisher.dataset_sort_key):
        values = [r for r in timings if r['dataset'] == dataset and r['status'] == 'completed' and not r.get('smoke')]
        full = len(values) == 5 and {r['seed'] for r in values} == publisher.SEEDS
        runtime.append(dict(dataset=dataset, n=len(values), runs=values,
                            mean_seconds=statistics.mean(r['elapsed_seconds'] for r in values) if full else None,
                            basis='five_seed_observation' if full else 'insufficient_observations'))
    publisher.atomic_text(experiment / 'downstream_runtime.json', json.dumps(runtime, indent=2) + '\n')
    if missing:
        return state
    publisher.RESULTS_ROOT = Path(manifest['publication_root'])
    publisher.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    normalized = [dict(e, slug=e['dataset'], display_name=NAMES[e['dataset']]) for e in entries]
    # Serialize global index publication between independent experiment jobs.
    import fcntl
    with (publisher.RESULTS_ROOT / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = experiment / 'published.json'
        if not marker.exists():
            sections = publisher.publish(dict(submitted_entries=normalized, reused_entries=[]), results,
                                         manifest['created_at_new_york'][2:].replace('-', ''),
                                         manifest.get('preset') or manifest['experiment'],
                                         json.loads(publisher.LITERATURE.read_text()))
            publisher.atomic_text(marker, json.dumps(sections, indent=2) + '\n')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if not args.submit:
        state = finalize(experiment)
        print(json.dumps(state, indent=2))
        return 0 if state['status'] == 'complete' else 1
    import yaml
    manifest_path = experiment / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('finalizer_job'):
        raise ValueError('Finalizer already submitted')
    jobs = [j['job'] for j in manifest['downstream_jobs']]
    cluster = yaml.safe_load((experiment / 'source/configs/cluster/bigpurple_a100.yaml').read_text())['slurm']
    logs = experiment / 'downstream/logs'
    job = subprocess.check_output(['sbatch', '--parsable', '--account=' + cluster['account'],
        '--job-name=' + experiment.name + '-results', '--partition=cpu_short,cpu_long',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G', '--time=00:30:00',
        '--dependency=afterany:' + ':'.join(jobs), '--output=' + str(logs / 'results-%j.out'),
        '--error=' + str(logs / 'results-%j.err'), '--wrap=export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec ' +
        shlex.join([sys.executable, str(experiment / 'source/scripts/finalize_experiment.py'), '--experiment', str(experiment)])], text=True).strip().split(';')[0]
    manifest['finalizer_job'] = job
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('finalizer=' + job)
    return 0


if __name__ == '__main__':
    sys.exit(main())
