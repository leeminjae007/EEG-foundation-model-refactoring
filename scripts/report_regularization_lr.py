"""첫 LR 조정의 완료율과 validation BAcc만 비교한다. Test 값은 읽지 않는다."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/finetune_regularization_20260912'
entries = json.loads((CAMPAIGN / 'lr_array.json').read_text())
audit = json.loads((CAMPAIGN / 'audit.json').read_text())
report = {'test_metrics_used': False, 'tasks': []}
for task in audit['tasks']:
    completed = []
    pending = []
    for entry in entries:
        if entry['dataset'] != task['dataset']:
            continue
        path = Path(entry['result_dir']) / 'result.json'
        if not path.exists():
            pending.append(entry['seed'])
            continue
        result = json.loads(path.read_text())['balanced_accuracy']['selection']
        completed.append({'seed': entry['seed'], 'best_validation_bacc': result['score'],
                          'best_epoch': result['epoch'], 'path': str(path)})
    row = {'dataset': task['dataset'], 'completed': completed, 'missing_seeds': pending,
           'complete': not pending, 'baseline_validation_mean': task['selected_validation_bacc_mean']}
    if len(completed) == 5:
        row['adjusted_validation_mean'] = statistics.mean(x['best_validation_bacc'] for x in completed)
        row['adjusted_validation_sd'] = statistics.stdev(x['best_validation_bacc'] for x in completed)
        row['mean_difference'] = row['adjusted_validation_mean'] - row['baseline_validation_mean']
    report['tasks'].append(row)
(CAMPAIGN / 'validation_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
for task in report['tasks']:
    print(task['dataset'], str(len(task['completed'])) + '/5 completed',
          'delta=' + str(task.get('mean_difference', 'not yet compared')))
