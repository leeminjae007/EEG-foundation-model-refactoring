"""동일 device와 캡처한 동일 RNG에서 기본 모델과 GR9-1 공통 초기화를 비교한다."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.runtime import set_paths
set_paths()
import numpy as np
import torch
import yaml
from src.model import PretrainModel
from src.training.checkpoint import rng_state, restore_rng
from src.training.diagnostics import fingerprint


def check(config_path, baseline_path, output, device_name):
    config = yaml.safe_load(config_path.read_text())
    baseline = yaml.safe_load(baseline_path.read_text())
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    before = rng_state(device)
    actual = PretrainModel(config, device)
    actual_hashes = fingerprint(actual)['tensors']
    after = rng_state(device)
    restore_rng(before, device)
    reference = PretrainModel(baseline, device)
    reference_hashes = fingerprint(reference)['tensors']
    reference_after = rng_state(device)
    if actual_hashes != reference_hashes:
        raise RuntimeError('common initialization differs from GR9-1 on the same device')
    for key in ['torch'] + (['cuda'] if device.type == 'cuda' else []):
        if not torch.equal(after[key], reference_after[key]):
            raise RuntimeError('initialization RNG consumption differs: ' + key)
    report = {'comparison_available': True, 'all_common_tensors_equal': True,
              'tensor_count': len(actual_hashes), 'device': str(device), 'seed': seed,
              'comparison': 'Captured same-device CPU/CUDA RNG restored before GR9-1 factory replay',
              'config': str(config_path), 'baseline': str(baseline_path),
              'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'tensors': actual_hashes}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Common initialization verified:', len(actual_hashes), 'tensors on', device, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    check(args.config, args.baseline, args.output, args.device)
