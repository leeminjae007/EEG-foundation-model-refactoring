import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import submit_experiment_downstream as downstream
from scripts import submit_experiment_smoke as smoke
from scripts import finalize_experiment
from src.training.smoke import TimedSmoke, LimitedLoader

ROOT = Path(__file__).resolve().parents[1]


def experiment(tmp_path):
    folder = tmp_path / 'experiment'
    (folder / 'source').mkdir(parents=True)
    shutil.copytree(ROOT / 'configs', folder / 'source/configs')
    (folder / 'manifest.json').write_text(json.dumps(dict(preset='gr2-d4-patch-dimension-mask55')))
    return folder


def test_arrays_use_success_dependency_without_hold_or_polling(tmp_path, monkeypatch):
    folder = experiment(tmp_path)
    commands = []
    def submit(command, **kwargs):
        commands.append(command)
        return str(200 + len(commands)) + '\n'
    monkeypatch.setattr(downstream.subprocess, 'check_output', submit)
    monkeypatch.setattr(sys, 'argv', ['submit', '--experiment', str(folder), '--dependency', '123'])
    downstream.main()
    assert len(commands) == 12
    for command in commands:
        assert '--dependency=afterok:123' in command
        assert '--kill-on-invalid-dep=yes' in command
        assert '--hold' not in command
        assert '--gpus-per-task=a100:1' in command
        assert '--cpus-per-task=2' in command
        wrap = next(x for x in command if x.startswith('--wrap='))
        assert 'srun' in wrap and '--gpu-bind=single:1' in wrap
    saved = json.loads((folder / 'manifest.json').read_text())
    assert len(saved['downstream_jobs']) == 12
    assert [x['dataset'] for x in saved['downstream_jobs']] == list(downstream.DATASETS)
    assert saved['downstream_state'] == 'dependency'
    with pytest.raises(ValueError, match='already submitted'):
        downstream.main()


def test_all_60_configs_preserve_common_lr_and_base_settings(tmp_path):
    folder = experiment(tmp_path)
    entries = downstream.prepare(folder, folder / 'pretrain/checkpoint-epoch-0040.pth')
    assert len(entries) == 60
    for dataset in downstream.DATASETS:
        group = [e for e in entries if e['dataset'] == dataset]
        assert {e['seed'] for e in group} == set(downstream.SEEDS)
        for entry in group:
            cfg = yaml.safe_load(Path(entry['config']).read_text())
            assert not downstream.warmup(cfg)
            expected_batch = 2 if dataset == 'isruc' else 64
            assert cfg['optimization']['batch_size_per_gpu'] == expected_batch
            if dataset == 'isruc':
                assert cfg['optimization']['gradient_accumulation_steps'] == 32
            rates = [cfg['optimization'][k + '_learning_rate'] for k in ('tokenizer', 'encoder', 'head')]
            assert len(set(rates)) == 1
            if dataset == 'tusz':
                assert cfg['optimization']['class_counts'] == [28670, 12842]
                assert cfg['optimization']['label_smoothing'] == 0
    for dataset in ('chb', 'tuab', 'tuev'):
        assert downstream.resource_policy(folder / 'source', 'later-ablation', dataset)['partitions'] == 'gl40s_long'
        for preset in ('gr2-d4-patch-dimension-mask55', 'gr2-d4-patch-dimension-mask60'):
            policy = downstream.resource_policy(folder / 'source', preset, dataset)
            assert policy['partitions'] == 'a100_short,a100_long'
            assert policy['gpu'] == 'a100'
    assert downstream.resource_policy(folder / 'source', 'gr2-d2-static', 'tuab')['gpu'] == 'l40s'


def test_missing_or_smoke_results_never_publish(tmp_path, monkeypatch):
    folder = experiment(tmp_path)
    downstream.prepare(folder, folder / 'pretrain/checkpoint-epoch-0040.pth')
    monkeypatch.setattr(finalize_experiment.publisher, 'publish', lambda *a: pytest.fail('Published missing results'))
    state = finalize_experiment.finalize(folder)
    assert state['status'] == 'incomplete' and len(state['missing']) == 60
    assert not (folder / 'published.json').exists()
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['smoke_seconds'] = 300
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Smoke'):
        finalize_experiment.finalize(folder)


def test_timer_bounds_and_no_production_collective(monkeypatch):
    now = [0]
    monkeypatch.setattr('src.training.smoke.time.monotonic', lambda: now[0])
    timer = TimedSmoke(300, torch.device('cpu'), 1)
    timer.start()
    now[0] = 299
    assert not timer.finished()
    now[0] = 300
    assert timer.finished()
    assert not TimedSmoke(0, torch.device('cpu'), 4).finished()


def test_limited_loader_preserves_sampler_and_batch_size():
    loader = torch.utils.data.DataLoader(torch.arange(640), batch_size=64)
    limited = LimitedLoader(loader, 4)
    assert limited.sampler is loader.sampler
    assert len(limited) == 4
    assert [len(batch) for batch in limited] == [64] * 4


def test_mask55_smoke_requests_a100_for_both_stages(tmp_path, monkeypatch):
    folder = tmp_path / 'smoke'
    config_path = folder / 'configs/pretrain.yaml'
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        'mae': {'decoder_depth': 4},
        'encoder': {'fusion_gate': 'patch_feature'},
        'masking': {'mask_ratio': .55},
        'optimization': {'batch_size_per_gpu': 128},
        'runtime': {},
    }))
    downstream_config = folder / 'configs/downstream/tusz_seed42.yaml'
    downstream_config.parent.mkdir(parents=True)
    downstream_config.write_text(yaml.safe_dump({'model': {}, 'runtime': {}}))
    (folder / 'downstream_entries.json').write_text(json.dumps([{'config': str(downstream_config)}]))
    manifest = {
        'preset': 'gr2-d4-patch-dimension-mask55',
        'pretrain_entries': [{'config': str(config_path), 'gpu_partitions': 'a100_short,a100_long'}],
        'pretrain_excluded_nodes': [],
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return ('901\n' if len(calls) == 1 else '902\n')

    monkeypatch.setattr(smoke.subprocess, 'check_output', submit)
    monkeypatch.setattr(sys, 'argv', ['smoke', '--experiment', str(folder), '--gpu', 'a100'])
    smoke.main()
    assert len(calls) == 2
    downstream_command = calls[1]
    assert '--partition=a100_short,a100_long' in downstream_command
    assert '--gpus-per-task=a100:1' in downstream_command
    assert '--gpus-per-task=a100:1' in next(x for x in downstream_command if x.startswith('--wrap='))


def test_finalizer_publishes_only_all_five_and_is_idempotent(tmp_path, monkeypatch):
    folder = experiment(tmp_path)
    entries = downstream.prepare(folder, folder / 'pretrain/checkpoint-epoch-0040.pth')
    entries = [e for e in entries if e['dataset'] == 'tuab']
    (folder / 'downstream_entries.json').write_text(json.dumps(entries))
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest.update(publication_root=str(tmp_path / 'published'), created_at_new_york='260919-2100', experiment='mask60')
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    for entry in entries:
        out = Path(entry['output']); out.mkdir(parents=True)
        config = yaml.safe_load(Path(entry['config']).read_text())
        model = {'weight': torch.ones(2)}
        torch.save(dict(config=config, epoch=config['optimization']['epochs'], model=model, extra={'partial_epoch_smoke': False}), out / 'last.pth')
        torch.save(model, out / 'best-balanced_accuracy.pth')
        (out / 'validation.jsonl').write_text(json.dumps(dict(epoch=1, balanced_accuracy=.8)))
        (out / 'result.json').write_text(json.dumps({'balanced_accuracy': {'selection': {'epoch': 1, 'score': .8},
            'test': {'balanced_accuracy': .7, 'auroc': .8, 'auprc': .75}}}))
    if sys.platform == 'win32':
        monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(flock=lambda *a: None, LOCK_EX=1))
    assert finalize_experiment.finalize(folder)['status'] == 'complete'
    index = tmp_path / 'published/RESULTS.md'
    original = index.read_text(encoding='utf-8')
    assert finalize_experiment.finalize(folder)['status'] == 'complete'
    assert index.read_text(encoding='utf-8') == original
    assert len(list((tmp_path / 'published').glob('*/seed_results.csv'))) == 1
