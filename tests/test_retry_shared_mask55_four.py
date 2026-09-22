import json
from pathlib import Path

import yaml

from scripts import retry_shared_mask55_four as retry


def test_only_missing_seeds_are_resubmitted_once(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    stage = campaign / 'accounts/hk4935/fourtask_downstream/pe-ch_order'
    stage.mkdir(parents=True)
    checkpoint = tmp_path / 'pretrain.pth'
    checkpoint.write_bytes(b'checkpoint')
    (stage / 'pretrain').mkdir()
    (stage / 'pretrain/verified.json').write_text(json.dumps({'checkpoint': str(checkpoint)}))
    source_pretrain = campaign / 'accounts/hk4935/pe-ch_order'
    resources = dict(gpu='l40s', partitions='gl40s_long', cpus_per_task=2,
                     memory='32G', time='24:00:00', excluded_nodes=[])
    manifest = dict(source_pretrain=str(source_pretrain),
                    downstream_jobs=[dict(dataset=dataset, job=str(100 + group), resources=resources)
                                     for group, dataset in enumerate(retry.CONFIG_DATASETS)])
    (stage / 'manifest.json').write_text(json.dumps(manifest))
    entries = []
    for dataset in retry.CONFIG_DATASETS:
        for seed in sorted(retry.SEEDS):
            output = stage / 'downstream' / dataset / f'seed-{seed}'
            config = stage / 'configs' / f'{dataset}-{seed}.yaml'
            config.parent.mkdir(exist_ok=True)
            config.write_text(yaml.safe_dump(dict(data={'dataset': retry.CONFIG_DATASETS[dataset]},
                                                  seed=seed, runtime={'output': str(output)},
                                                  model={'checkpoint': str(checkpoint)})))
            entries.append(dict(dataset=dataset, seed=seed, config=str(config), output=str(output)))
    (stage / 'downstream_entries.json').write_text(json.dumps(entries))
    # One pre-existing successful seed must be preserved, not launched again.
    completed = Path(entries[0]['output'])
    completed.mkdir(parents=True)
    (completed / 'result.json').write_text(json.dumps({
        'balanced_accuracy': {'selection': {'score': 0.8},
                              'test': {'balanced_accuracy': 0.7, 'auroc': 0.8, 'auprc': 0.6}}}))
    (completed / 'best-balanced_accuracy.pth').write_bytes(b'best')
    (completed / 'last.pth').write_bytes(b'last')
    monkeypatch.setattr(retry, 'CAMPAIGN', campaign)
    calls = []

    def fake_check_output(command, text):
        calls.append(command)
        return '' if command[0] == 'squeue' else str(900 + len(calls))

    monkeypatch.setattr(retry.subprocess, 'check_output', fake_check_output)
    retry.submit(stage, 'pe-ch_order', ('chb',))
    saved = json.loads((stage / 'manifest.json').read_text())
    assert saved['recovery_jobs'][0]['indices'] == [1, 2, 3, 4]
    assert len([c for c in calls if c[0] == 'sbatch']) == 2  # array + finalizer
    retry.submit(stage, 'pe-ch_order', ('chb',))
    assert len([c for c in calls if c[0] == 'sbatch']) == 2
