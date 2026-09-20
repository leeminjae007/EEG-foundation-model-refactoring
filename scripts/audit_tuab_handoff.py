"""Read-only completion checks followed by complete four-campaign publication."""
import json
import math
from pathlib import Path


def audit_and_publish(experiment):
    import torch
    import yaml
    from scripts import monitor_experiment_results as publisher
    from scripts.launch_tuab_handoff import write
    experiment = Path(experiment).resolve()
    manifest = json.loads((experiment / 'manifest.json').read_text())
    if len(set(manifest['aliases'])) != 4:
        raise ValueError('Expected four distinct TUAB campaigns')
    missing, groups = [], {}
    for alias in manifest['aliases']:
        entries = json.loads((experiment / alias / 'entries.json').read_text())
        if len(entries) != 5 or {e['seed'] for e in entries} != publisher.SEEDS:
            raise ValueError(alias + ': expected exactly five seeds')
        results = {}
        for entry in entries:
            output = Path(entry['output'])
            try:
                payload = json.loads((output / 'result.json').read_text())
                metrics = publisher.selector_test(payload)
                if not all(math.isfinite(metrics[k]) for k in ('balanced_accuracy', 'auroc', 'auprc')):
                    raise ValueError('Missing finite binary metrics')
                saved = torch.load(output / 'last.pth', map_location='cpu')
                if saved['epoch'] != 20 or saved['extra'].get('partial_epoch_smoke'):
                    raise ValueError('Incomplete downstream checkpoint')
                if saved['config'] != yaml.safe_load(Path(entry['config']).read_text()):
                    raise ValueError('Checkpoint configuration mismatch')
                selection = payload['balanced_accuracy']['selection']
                if selection != saved['extra']['best']['balanced_accuracy']:
                    raise ValueError('Checkpoint/result validation selector differs')
                rows = [json.loads(line) for line in (output / 'validation.jsonl').read_text().splitlines()]
                if not any(r['epoch'] == selection['epoch'] and r['balanced_accuracy'] == selection['score'] for r in rows):
                    raise ValueError('No matching validation record')
                weights = torch.load(output / 'best-balanced_accuracy.pth', map_location='cpu')
                if set(weights) != set(saved['model']) or any(weights[k].shape != v.shape or not torch.isfinite(weights[k]).all() for k, v in saved['model'].items()):
                    raise ValueError('Invalid validation-selected weights')
                results[str(entry['output']).rstrip('/') + '/result.json'] = payload
                del saved, weights
            except Exception as exc:
                missing.append(dict(alias=alias, seed=entry['seed'], job=entry['job'], error=repr(exc)))
        groups[alias] = entries, results
    status = dict(status='incomplete' if missing else 'complete', expected=20,
                  readable_results=sum(len(v[1]) for v in groups.values()), missing=missing)
    write(experiment / 'completion.json', status)
    print(json.dumps(status, indent=2), flush=True)
    if missing:
        return 1
    import fcntl
    publisher.RESULTS_ROOT = Path(manifest['publication_root'])
    publisher.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = manifest['created_at_new_york'][2:11].replace('-', '')
    with (publisher.RESULTS_ROOT / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for alias, (entries, results) in groups.items():
            marker = experiment / alias / 'published.json'
            if marker.exists():
                continue
            normalized = [dict(e, slug='tuab', display_name='TUAB') for e in entries]
            sections = publisher.publish(dict(submitted_entries=normalized, reused_entries=[]), results,
                                         stamp, alias, json.loads(publisher.LITERATURE.read_text()))
            write(marker, sections)
    write(experiment / 'published.json', dict(status='published', campaigns=4, seeds=20))
    return 0
