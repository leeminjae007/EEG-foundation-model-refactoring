import json

import yaml

from scripts import submit_mask55_tuab_fixed_five as fixed


def test_prepare_and_submit_five_seeds(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    results = tmp_path / 'results'
    data = tmp_path / 'data'
    data.mkdir()
    checkpoint = tmp_path / 'pretrain.pth'
    checkpoint.write_bytes(b'verified checkpoint')
    (base / 'pretrain').mkdir(parents=True)
    (base / 'pretrain/verified.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), sha256=fixed.digest(checkpoint),
        strict_load=True, partial_epoch_smoke=False)))
    configs = base / 'configs/downstream'
    configs.mkdir(parents=True)
    for seed in fixed.SEEDS:
        config = dict(seed=seed, data=dict(dataset='tuab', dataset_dir=str(data), num_workers=4),
                      model=dict(checkpoint='old', head_dropout=.1),
                      optimization=dict(epochs=20, batch_size_per_gpu=64,
                                        gradient_accumulation_steps=8,
                                        tokenizer_learning_rate=1e-5,
                                        encoder_learning_rate=1e-5,
                                        head_learning_rate=1e-5, weight_decay=5e-5),
                      runtime=dict(output='old'))
        (configs / f'tuab_seed{seed}.yaml').write_text(yaml.safe_dump(config))
    monkeypatch.setattr(fixed, 'BASE', base)
    monkeypatch.setattr(fixed, 'RESULTS', results)
    monkeypatch.setattr(fixed.shutil, 'copytree',
                        lambda source, destination, ignore: destination.mkdir(parents=True))
    campaign = fixed.prepare()
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    assert len(entries) == 5
    assert {entry['seed'] for entry in entries} == set(fixed.SEEDS)
    for entry in entries:
        config = yaml.safe_load((campaign / 'configs' / f"tuab_seed{entry['seed']}.yaml").read_text())
        optimization = config['optimization']
        assert optimization['epochs'] == 20
        assert optimization['batch_size_per_gpu'] == 64
        assert optimization['gradient_accumulation_steps'] == 1
        assert optimization['early_stopping'] == fixed.EARLY_STOPPING
        assert [optimization[key] for key in ('tokenizer_learning_rate',
                                             'encoder_learning_rate',
                                             'head_learning_rate')] == [1e-4] * 3
        assert optimization['weight_decay'] == 5e-5
        assert config['model']['head_dropout'] == .3
        assert config['model']['checkpoint'] == str(checkpoint)
    calls = []

    def fake_submit(command, text):
        calls.append(command)
        return str(9000 + len(calls))

    monkeypatch.setattr(fixed.subprocess, 'check_output', fake_submit)
    fixed.submit(campaign)
    assert '--array=0-4%5' in calls[0]
    assert '--time=24:00:00' in calls[0]
    assert '--partition=a100_short,a100_long' in calls[0]
    assert any(arg.startswith('--exclude=') for arg in calls[0])
    assert '--dependency=afterany:9001' in calls[1]
