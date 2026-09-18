"""GR9-1의 train/validation을 감사하고 단일 요인 fine-tune 후보를 만든다."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 1234, 696, 1001, 3407]
TASKS = [('mentalarithmetic', 'stress', 'MentalArithmetic', 20, 5, 1),
         ('bciciv2a', 'bciciv2a', 'BCIC-IV-2a', 22, 4, 4)]
ARMS = ['backbone_lr_x0p1', 'head_dropout_plus0p2', 'head_h2']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_parameters(channels, patches, hidden, outputs):
    width = hidden * 200
    return channels * patches * 200 * width + width + width * 200 + 200 + 200 * outputs + outputs


def prepare(original):
    campaign = ROOT / 'outputs/finetune_regularization_20260912'
    campaign.mkdir(exist_ok=True)
    if (campaign / 'submission.json').exists():
        print((campaign / 'submission.json').read_text())
        return
    destination = ROOT / 'configs/finetune_regularization'
    destination.mkdir(exist_ok=True)
    audit, entries = [], []
    trial = 'gr9_1_equal_shpe_gelu_rmsnorm_d200_e40_20260910'
    source = original / 'outputs/gr9_followup_20260912/downstream/warmup5' / trial
    for slug, dataset, display, channels, patches, outputs in TASKS:
        runs = []
        for seed in SEEDS:
            run_dir = source / dataset / 'eeg_mae/all_patch_reps' / f'seed-{seed}'
            result_path = run_dir / 'result.json'
            result = json.loads(result_path.read_text())
            assert result['seed'] == seed and result['selection_metric'] == 'balanced_accuracy'
            assert result['epochs_ran'] == 50
            logs = sorted({p.resolve() for p in run_dir.glob('wandb/*/files/output.log')})
            assert len(logs) == 1
            history = []
            for line in logs[0].read_text().splitlines():
                start = line.find('{')
                if start < 0:
                    continue
                try:
                    row = json.loads(line[start:])
                except ValueError:
                    continue
                if all(key in row for key in ['train_loss', 'validation', 'train', 'epoch']):
                    history.append({key: row[key] for key in ['epoch', 'train_loss', 'train', 'validation']})
            assert [row['epoch'] for row in history] == list(range(1, 51))
            last = history[-1]
            runs.append({'seed': seed, 'best_epoch': result['best_epoch'],
                         'best_validation': result['best_validation'],
                         'last': last, 'history': history,
                         'result_path': str(result_path), 'result_sha256': digest(result_path),
                         'log_path': str(logs[0]), 'log_sha256': digest(logs[0])})
            base_path = ROOT / f'configs/downstream/gr9-1_{slug}_seed{seed}.yaml'
            baseline = yaml.safe_load(base_path.read_text())
            assert 'warmup_epochs' not in baseline['optimization']
            for arm in ARMS:
                config = deepcopy(baseline)
                opt = config['optimization']
                changed = {}
                if arm == 'backbone_lr_x0p1':
                    for name in ['tokenizer_learning_rate', 'encoder_learning_rate']:
                        opt[name] = float(f'{opt[name] * .1:.8g}')
                        changed['optimization.' + name] = [baseline['optimization'][name], opt[name]]
                elif arm == 'head_dropout_plus0p2':
                    config['model']['head_dropout'] = round(config['model']['head_dropout'] + .2, 2)
                    changed['model.head_dropout'] = [baseline['model']['head_dropout'], config['model']['head_dropout']]
                else:
                    config['model']['head_hidden_tokens'] = 2
                    changed['model.head_hidden_tokens'] = [baseline['model']['head_hidden_tokens'], 2]
                config['runtime']['output'] = f'outputs/finetune_regularization_20260912/{arm}/{slug}_seed{seed}'
                # Only the declared scientific factor and output path may differ.
                restored = deepcopy(config)
                for name in changed:
                    section, key = name.split('.')
                    restored[section][key] = baseline[section][key]
                restored['runtime'] = baseline['runtime']
                assert restored == baseline
                path = destination / f'{arm}_{slug}_seed{seed}.yaml'
                content = yaml.safe_dump(config, sort_keys=False)
                if path.exists() and path.read_text() != content:
                    raise RuntimeError('refuse to overwrite a different experiment config: ' + str(path))
                path.write_text(content)
                entries.append({'dataset': display, 'slug': slug, 'seed': seed, 'arm': arm,
                                'config': str(path), 'sha256': digest(path),
                                'baseline_config': str(base_path), 'baseline_sha256': digest(base_path),
                                'changes': changed})
        audit.append({'dataset': display, 'runs': runs,
                      'last_train_bacc_mean': statistics.mean(r['last']['train']['balanced_accuracy'] for r in runs),
                      'last_validation_bacc_mean': statistics.mean(r['last']['validation']['balanced_accuracy'] for r in runs),
                      'selected_validation_bacc_mean': statistics.mean(r['best_validation']['balanced_accuracy'] for r in runs),
                      'best_epochs': [r['best_epoch'] for r in runs],
                      'head_parameters_default': head_parameters(channels, patches, patches, outputs),
                      'head_parameters_h2': head_parameters(channels, patches, 2, outputs)})
    report = {'baseline': 'Completed original GR9-1 epoch40 downstream, full 50 epochs, validation-BAcc selection',
              'test_metrics_used': False, 'seeds': SEEDS, 'tasks': audit}
    (campaign / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {'status': 'prepared_not_submitted', 'entries': entries,
                'priority': 'backbone_lr_x0p1', 'independent_arms': ARMS,
                'selection': 'Compare complete five-seed validation BAcc means separately for each task; no test-based adoption.',
                'preserved': 'GR9-1 checkpoint, 50 epochs, head LR, batch size, weight decay, loss, splits, all-patch pooling, dual selectors',
                'notes': ['Do not combine factors before their independent comparisons.',
                          'Existing original-project baselines can be reused as historical controls; new-engine comparisons remain exploratory until runtime comparability is checked.',
                          'The pending geometry pretrain is a different experiment and is not the source checkpoint for these configs.',
                          'Do not interpret a smaller train/validation gap alone as improvement; validation BAcc must improve.']}
    (campaign / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for task in audit:
        print(task['dataset'], {k:v for k,v in task.items() if k not in ['dataset','runs']})
    print('Prepared', len(entries), 'configs; no training submitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    prepare(parser.parse_args().original)
