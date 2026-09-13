"""완료된 geometry 실험의 검증 결과만 읽어 기본 마스킹 선택 근거를 저장한다."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ROSTER = [('tuab', 'TUAB'), ('tuev', 'TUEV'), ('chb', 'CHB-MIT'),
          ('seed-v', 'SEED-V'), ('seed-vig', 'SEED-VIG'), ('faced', 'FACED'),
          ('mumtaz', 'Mumtaz'), ('stress', 'MentalArithmetic'), ('bciciv2a', 'BCIC-IV-2a'),
          ('physio', 'PhysioNet-MI'), ('speech', 'BCIC2020-3'), ('isruc', 'ISRUC'),
          ('hmc', 'HMC'), ('siena', 'Siena')]
SEEDS = [42, 1234, 696, 1001, 3407]


def audit(original):
    campaigns = {
        'GR2-3': 'outputs/gr2_gr5_loso_20260906/downstream/gr2-3',
        'GR6-1': 'outputs/gr6_geometry_20260906/downstream_adopted/gr6-1',
        'GR6-2': 'outputs/gr6_geometry_20260906/downstream_adopted/gr6-2',
    }
    reports, settings = {}, {}
    for label, relative in campaigns.items():
        entries = {}
        for manifest in (original / relative).rglob('array.json'):
            for entry in json.loads(manifest.read_text()):
                key = (entry['dataset'], entry['seed'])
                if key in entries:
                    raise ValueError('duplicate manifest cell: ' + str(key))
                entries[key] = entry
        assert len(entries) == 70
        rows, settings[label] = [], {}
        for dataset, display in ROSTER:
            metric = 'r2' if dataset == 'seed-vig' else 'balanced_accuracy'
            values, sources = [], []
            for seed in SEEDS:
                entry = entries[dataset, seed]
                path = Path(entry['result_dir']) / 'result.json'
                content = path.read_bytes()
                result = json.loads(content)
                config = yaml.safe_load(Path(entry['config']).read_text())
                assert result['seed'] == seed and result['dataset'] == dataset
                assert result['selection_metric'] == metric
                assert result['epochs_ran'] == config['optimization']['epochs']
                values.append(result['best_validation'][metric])
                sources.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
                model = config['model'].copy()
                model.pop('checkpoint')
                settings[label][dataset, seed] = {'model': model, 'data': config['data'],
                                                  'optimization': config['optimization']}
            rows.append({'dataset': display, 'slug': dataset, 'metric': metric,
                         'seeds': values, 'mean': statistics.mean(values),
                         'sd': statistics.stdev(values), 'sources': sources})
        reports[label] = {'tasks': rows}
    matched = [ds for ds, _ in ROSTER if all(
        settings['GR2-3'][ds, seed] == settings['GR6-1'][ds, seed] == settings['GR6-2'][ds, seed]
        for seed in SEEDS)]
    assert len(matched) == 9
    assert settings['GR6-1'] == settings['GR6-2']
    for report in reports.values():
        rows = report['tasks']
        report['matched_8_classification_validation_bacc'] = statistics.mean(
            row['mean'] for row in rows if row['slug'] in matched and row['metric'] == 'balanced_accuracy')
        report['all_13_classification_validation_bacc'] = statistics.mean(
            row['mean'] for row in rows if row['metric'] == 'balanced_accuracy')
        report['seed_vig_validation_r2'] = next(row['mean'] for row in rows if row['metric'] == 'r2')
    selected = max(reports, key=lambda label: reports[label]['matched_8_classification_validation_bacc'])
    report = {'selected': selected, 'seeds': SEEDS, 'matched_tasks': matched,
              'criterion': 'Equal-task mean validation BAcc over 8 classification tasks with identical downstream settings across all three variants; SEED-VIG R2 reported separately.',
              'limitations': ['Exploratory selection from completed runs, not a preregistered estimate.',
                             'GR2-3 binary runs used CE; GR6 adopted binary runs used weighted CE. Those five tasks are excluded from the three-way score.',
                             'SEED-VIG validation R2 is higher for GR2-3.',
                             'These masking variants were trained on GR2 architecture. Transfer to GR9-1 PE has not been trained or evaluated.'],
              'test_metrics_used_for_selection': False, 'variants': reports}
    path = ROOT / 'docs/geometry_selection.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(selected)
    for label, item in reports.items():
        print(label, {k: v for k, v in item.items() if k != 'tasks'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    audit(parser.parse_args().original)
