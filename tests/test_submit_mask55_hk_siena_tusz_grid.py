import json
from pathlib import Path

import yaml

from scripts import submit_mask55_hk_siena_tusz_grid as grid


def test_prepare_and_submit_freezes_binary_five_seed_grid(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    checkpoint = tmp_path / 'pretrain.pth'
    checkpoint.write_bytes(b'verified-pretrain')
    digest = grid.digest(checkpoint)
    (base / 'pretrain').mkdir(parents=True)
    (base / 'pretrain/verified.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), sha256=digest, strict_load=True,
        partial_epoch_smoke=False)))
    configs = base / 'configs/downstream'
    configs.mkdir(parents=True)
    for dataset in grid.DATASETS:
        for seed in grid.SEEDS:
            config = dict(seed=seed, data={'dataset': dataset}, model={'checkpoint': 'old'},
                          optimization=dict(epochs=50, tokenizer_learning_rate=1e-4,
                                            encoder_learning_rate=1e-4, head_learning_rate=1e-4,
                                            weight_decay=.01), runtime={'output': 'old'})
            (configs / f'{dataset}_seed{seed}.yaml').write_text(yaml.safe_dump(config))
    campaign = tmp_path / 'campaign'
    monkeypatch.setattr(grid, 'BASE', base)
    monkeypatch.setattr(grid, 'CAMPAIGN', campaign)
    monkeypatch.setattr(grid, 'EXPECTED_SHA256', digest)
    def fake_copytree(source, destination, ignore):
        cluster = Path(destination) / 'configs/cluster'
        cluster.mkdir(parents=True)
        (cluster / 'bigpurple_a100.yaml').write_text(yaml.safe_dump(
            {'slurm': {'pretrain': {'excluded_nodes': ['a100-4011', 'a100-4024']}}}))

    monkeypatch.setattr(grid.shutil, 'copytree', fake_copytree)
    grid.prepare()
    entries = json.loads((campaign / 'downstream_entries.json').read_text())
    assert len(entries) == 360
    assert {e['seed'] for e in entries} == set(grid.SEEDS)
    assert {e['dataset'] for e in entries} == set(grid.DATASETS)
    for index in (0, 179, 180, 359):
        entry = entries[index]
        config = yaml.safe_load(Path(entry['config']).read_text())
        assert config['optimization']['early_stopping'] == grid.EARLY_STOPPING
        assert config['model']['checkpoint'] == str(checkpoint)
        assert config['runtime']['output'] == entry['output']
    calls = []

    def fake_submit(command, text):
        calls.append(command)
        return str(9000 + len(calls))

    monkeypatch.setattr(grid.subprocess, 'check_output', fake_submit)
    grid.submit(campaign)
    assert len(calls) == 3  # Siena array, TUSZ array, CPU aggregate.
    assert all('--gpus-per-task=a100:1' in call for call in calls[:2])
    assert all('--exclude=a100-4011,a100-4024' in call for call in calls[:2])
    assert '--array=' + ','.join(map(str, range(180))) + '%5' in calls[0]
    assert '--array=' + ','.join(map(str, range(180, 360))) + '%5' in calls[1]
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert manifest['status'] == 'submitted'
    assert manifest['selection'] == 'Five-seed mean validation balanced accuracy only'
