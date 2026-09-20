"""Check real-data smoke artifacts without publishing scientific results."""
import argparse
import json
import math
from pathlib import Path
import sys


def audit(folder):
    import torch
    import yaml
    sys.path.insert(0, str(folder / 'source'))
    from ablation.bootstrap import ensure_data_imports
    ensure_data_imports()
    from ablation.models import build_pretrain
    from ablation.pretrain_resume import validate_resume
    saved = torch.load(folder / 'pretrain/last.pth', map_location='cpu')
    cfg = saved['config']
    assert cfg['mae']['decoder_depth'] == 2 and cfg['encoder']['fusion_gate'] == 'patch_feature'
    assert cfg['masking']['mask_ratio'] == .6
    assert cfg['optimization']['batch_size_per_gpu'] == 128
    assert saved['extra']['partial_epoch_smoke'] and saved['extra']['step'] >= 2
    assert len(saved['rng_states']) == 4 and saved['optimizer']['state']
    for key in ('torch', 'cuda'):
        assert len({r[key].numpy().tobytes() for r in saved['rng_states']}) == 4
    build_pretrain(cfg, torch.device('cpu')).load_state_dict(saved['model'], strict=True)
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    gate_keys = [k for k in saved['model'] if 'patch_gates' in k and k.endswith('weight')]
    assert len(gate_keys) == 3 and all(torch.count_nonzero(saved['model'][k]) > 0 for k in gate_keys)
    try:
        validate_resume(saved, cfg, 4)
    except ValueError as exc:
        assert 'smoke' in str(exc)
    else:
        raise AssertionError('Partial smoke was incorrectly accepted for resume')
    metrics = [json.loads(line) for line in (folder / 'pretrain/metrics.jsonl').read_text().splitlines()]
    assert all(math.isfinite(r['loss']) and math.isfinite(r['pre_clip_norm']) for r in metrics)
    assert all(r['masking/target_tokens'] == 342 and r['masking/context_tokens'] == 228 for r in metrics)
    health = [json.loads(p.read_text()) for p in (folder / 'pretrain').glob('gpu-health-*-rank*.json')]
    assert len(health) == 4 and all(h['cuda_probe'] == 'passed' for h in health)
    resources = [json.loads(p.read_text()) for p in (folder / 'pretrain').glob('smoke-resources-rank*.json')]
    assert len(resources) == 4 and min(r['elapsed_seconds'] for r in resources) >= 295
    report = dict(pretrain=dict(passed=True, steps=saved['extra']['step'], batch_per_gpu=128,
                                world_size=4, first_loss=metrics[0]['loss'], last_logged_loss=metrics[-1]['loss'],
                                target_tokens=342, context_tokens=228, trained_gate_weights=gate_keys,
                                partial_resume_rejected=True, health=health, resources=resources), downstream=[], problems=[])
    del saved
    for entry in json.loads((folder / 'downstream_entries.json').read_text()):
        if entry['seed'] != 42:
            continue
        try:
            out = Path(entry['output'])
            config = yaml.safe_load(Path(entry['config']).read_text())
            checkpoint = torch.load(out / 'last.pth', map_location='cpu')
            assert checkpoint['extra']['partial_epoch_smoke'] and checkpoint['extra']['step'] == 4
            assert not (out / 'result.json').exists()
            assert checkpoint['optimizer']['state']
            assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
            log = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
            rates = [config['optimization'][k + '_learning_rate'] for k in ('tokenizer', 'encoder', 'head')]
            assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(log[0]['lr_used'], rates))
            assert all(math.isfinite(row['loss']) and math.isfinite(row['pre_clip_norm']) for row in log)
            result = json.loads((out / 'smoke_result.json').read_text())
            selected = result['balanced_accuracy']
            assert selected['selection']['epoch'] == 1
            assert math.isfinite(selected['test']['balanced_accuracy'])
            secondary = 'auroc' if 'auroc' in selected['test'] else 'weighted_f1'
            assert math.isfinite(selected['test'][secondary])
            assert (out / 'best-balanced_accuracy.pth').is_file()
            assert (out / 'validation.jsonl').is_file()
            assert (out / 'gate_final').is_dir()
            timing = json.loads((out / 'runtime.json').read_text())
            assert timing['status'] == 'completed' and timing['smoke']
            report['downstream'].append(dict(dataset=entry['dataset'], passed=True,
                batch_size=config['optimization']['batch_size_per_gpu'], accumulation=config['optimization']['gradient_accumulation_steps'],
                updates=checkpoint['extra']['step'], first_lr=rates, final_train_loss=log[-1]['loss'],
                train_validation_test_and_checkpoint=True, elapsed_seconds=timing['elapsed_seconds']))
            del checkpoint
        except Exception as exc:
            report['problems'].append(dict(dataset=entry['dataset'], error=repr(exc)))
    report['passed'] = len(report['downstream']) == 12 and not report['problems']
    (folder / 'smoke_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    result = audit(parser.parse_args().experiment.resolve())
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)
