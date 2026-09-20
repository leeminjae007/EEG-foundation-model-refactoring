"""Verify the final checkpoint before Slurm reports pretraining success."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(experiment):
    import torch
    import yaml
    experiment = Path(experiment).resolve()
    sys.path.insert(0, str(experiment / 'source'))
    from ablation.bootstrap import ensure_data_imports
    ensure_data_imports()
    from ablation.models import build_pretrain
    from ablation.pretrain_resume import validate_resume
    config = yaml.safe_load((experiment / 'configs/pretrain.yaml').read_text())
    path = experiment / 'pretrain' / ('checkpoint-epoch-%04d.pth' % config['optimization']['epochs'])
    saved = torch.load(path, map_location='cpu')
    validate_resume(saved, config, 4)
    if saved['epoch'] != config['optimization']['epochs'] or saved['config'] != config:
        raise ValueError('Final checkpoint epoch/config mismatch')
    for key in ('model', 'optimizer', 'scheduler', 'rng_states'):
        if not saved.get(key):
            raise ValueError('Missing checkpoint state: ' + key)
    if saved['extra']['step'] < 1 or not saved['extra'].get('dataset_fingerprint'):
        raise ValueError('Missing optimizer progress or dataset fingerprint')
    model = build_pretrain(config, torch.device('cpu'))
    model.load_state_dict(saved['model'], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in saved['model'].values()):
        raise ValueError('Nonfinite checkpoint weights')
    for key in ('torch', 'cuda'):
        if len({r[key].numpy().tobytes() for r in saved['rng_states']}) != 4:
            raise ValueError('Expected independent four-rank RNG: ' + key)
    report = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  epoch=saved['epoch'], step=saved['extra']['step'], strict_load=True,
                  partial_epoch_smoke=False, world_size=4)
    (experiment / 'pretrain/verified.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().experiment), indent=2))
