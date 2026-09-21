"""Audit 15 validation-selected results and summarize their three test metrics."""

import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    campaign = parser.parse_args().campaign
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    rows, missing = [], []
    for entry in entries:
        try:
            result = json.loads((Path(entry['output']) / 'result.json').read_text())
            selected = result['balanced_accuracy']
            val = float(selected['selection']['score'])
            metrics = selected['test']
            required = ('balanced_accuracy', 'auroc', 'auprc') if entry['dataset'] == 'mumtaz' else (
                'balanced_accuracy', 'weighted_f1', 'kappa')
            if not math.isfinite(val) or not all(math.isfinite(float(metrics[k])) for k in required):
                raise ValueError('Missing or non-finite validation/test metric')
            rows.append(dict(dataset=entry['dataset'], seed=entry['seed'], validation_bacc=val,
                             selected_epoch=selected['selection']['epoch'],
                             test={key: float(metrics[key]) for key in required}))
        except Exception as exc:
            missing.append(dict(dataset=entry['dataset'], seed=entry['seed'], error=repr(exc)))
    summary = {}
    for dataset in ('mumtaz', 'bcic2020_3', 'bciciv2a'):
        group = [row for row in rows if row['dataset'] == dataset]
        if len(group) != 5 or {row['seed'] for row in group} != {42, 696, 1001, 1234, 3407}:
            continue
        keys = group[0]['test']
        summary[dataset] = dict(validation_bacc_mean=statistics.mean(row['validation_bacc'] for row in group),
                                test={key: dict(mean=statistics.mean(row['test'][key] for row in group),
                                                sd=statistics.pstdev(row['test'][key] for row in group))
                                      for key in keys}, seeds=group)
    report = dict(status='complete' if len(rows) == 15 and not missing else 'incomplete',
                  expected=15, readable=len(rows), missing=missing, summary=summary)
    (campaign / 'three_dataset_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], readable=len(rows), missing=missing,
                          summary=summary), indent=2))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
