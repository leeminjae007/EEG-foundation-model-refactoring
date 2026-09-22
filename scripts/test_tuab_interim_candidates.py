"""Snapshot the first five TUAB one-factor validation winners for interim test.

This is reporting only. Never rank hyperparameters or checkpoints by test data.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import yaml

CAMPAIGN = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/'
                '260922-0123-gr2-d2-mask55-tuab-onefactor-seed42-batch64-noaccum')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validation_best(raw):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(not math.isfinite(float(row['balanced_accuracy'])) for row in rows):
        raise ValueError('Missing or non-finite validation BAcc')
    return max(rows, key=lambda row: (float(row['balanced_accuracy']), -int(row['epoch'])))


def freeze(entry, stage):
    config = Path(entry['config'])
    if sha256(config) != entry['config_sha256']:
        raise ValueError('Frozen config hash mismatch')
    output = Path(entry['output'])
    validation = output / 'validation.jsonl'
    source = output / 'best-balanced_accuracy.pth'
    target = stage / ('best-%02d.pth' % entry['index'])
    for _ in range(3):
        before = validation.read_bytes()
        selected = validation_best(before)
        if (int(selected['epoch']) == len(before.splitlines()) and
                (output / 'last.pth').stat().st_mtime_ns < validation.stat().st_mtime_ns):
            time.sleep(2)
            continue
        stat = source.stat()
        shutil.copy2(source, target)
        if (validation.read_bytes() == before and source.stat().st_mtime_ns == stat.st_mtime_ns
                and source.stat().st_size == stat.st_size and sha256(source) == sha256(target)):
            return dict(index=entry['index'], config=str(config),
                        config_sha256=entry['config_sha256'], hp=entry['hp'],
                        checkpoint=str(target), checkpoint_sha256=sha256(target),
                        validation_epoch=int(selected['epoch']),
                        validation_bacc=float(selected['balanced_accuracy']),
                        validation_epochs_completed=len(before.splitlines()))
    raise RuntimeError('Checkpoint changed while freezing index %s' % entry['index'])


def submit():
    entries = json.loads((CAMPAIGN / 'downstream_entries.json').read_text())
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    if len(entries) != 12 or any(entries[i]['index'] != i or entries[i]['seed'] != 42
                                 for i in range(5)):
        raise ValueError('Unexpected first five one-factor entries')
    stage = CAMPAIGN / 'interim_tests' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage.mkdir(parents=True, exist_ok=False)
    frozen = [freeze(entries[index], stage) for index in range(5)]
    info = dict(campaign=str(CAMPAIGN), source=str(CAMPAIGN / 'source'), entries=frozen,
                policy='Interim test reporting only; validation BAcc selects checkpoint and HP')
    stage_file = stage / 'manifest.json'
    stage_file.write_text(json.dumps(info, indent=2) + '\n')
    resources = manifest['resources']
    command = ['sbatch', '--parsable', '--account=system',
               '--job-name=tuab-interim-best-test',
               '--partition=a100_dev,a100_short,a100_long', '--nodes=1', '--ntasks=1',
               '--gpus-per-task=a100:1', '--cpus-per-task=2', '--mem=32G',
               '--time=02:00:00', '--array=0-4%5',
               '--exclude=' + ','.join(resources['excluded_nodes']),
               '--chdir=' + str(CAMPAIGN / 'source'),
               '--output=' + str(stage / 'test-%A_%a.out'),
               '--error=' + str(stage / 'test-%A_%a.err'),
               '--wrap=exec ' + shlex.join([
                   'srun', '--ntasks=1', '--gpus-per-task=a100:1', '--gpu-bind=single:1',
                   '--kill-on-bad-exit=1', manifest['python'], str(Path(__file__).resolve()),
                   'worker', '--stage', str(stage)])]
    info['job'] = subprocess.check_output(command, text=True).strip().split(';', 1)[0]
    stage_file.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(dict(stage=str(stage), job=info['job'], entries=[
        {key: row[key] for key in ('index', 'hp', 'validation_epoch', 'validation_bacc')}
        for row in frozen]), indent=2))


def worker(stage):
    info = json.loads((stage / 'manifest.json').read_text())
    entry = info['entries'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    sys.path.insert(0, info['source'])
    from src.training.runtime import set_paths, setup
    set_paths()
    from scripts.downstream_gpu_guard import allocated_gpu
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    health = allocated_gpu(query, 12288)
    os.environ.update(CUDA_VISIBLE_DEVICES=health['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import torch
    from torch.utils.data import DataLoader
    from src.data.datasets.registry import get_dataset_spec
    from src.training.engine import build_finetune, evaluate
    config_path = Path(entry['config'])
    checkpoint = Path(entry['checkpoint'])
    if sha256(config_path) != entry['config_sha256'] or sha256(checkpoint) != entry['checkpoint_sha256']:
        raise ValueError('Frozen test input hash mismatch')
    config = yaml.safe_load(config_path.read_text())
    if config['data']['dataset'] != 'tuab' or config['seed'] != 42:
        raise ValueError('Unexpected test config')
    # Match the frozen run_finetune downstream setup, not pretrain runtime keys.
    device, _, world = setup('cuda', False, config['seed'], False, False, True)
    torch.ones(1, device=device).sum().item()
    model = build_finetune(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    spec = get_dataset_spec('tuab')
    dataset = spec.dataset_class(config['data']['dataset_dir'], 'test')
    dataset.enable_coordinate_only_channels()
    loader = DataLoader(dataset, batch_size=config['optimization']['batch_size_per_gpu'],
                        sampler=range(len(dataset)), num_workers=config['data']['num_workers'],
                        pin_memory=True)
    measured = evaluate(model, loader, spec.task, 'tuab', device, world)
    test = {key: float(measured[key]) for key in ('balanced_accuracy', 'auroc', 'auprc')}
    if not all(math.isfinite(value) for value in test.values()):
        raise ValueError('Non-finite test metric')
    result = dict(index=entry['index'], hp=entry['hp'],
                  validation_epoch=entry['validation_epoch'],
                  validation_bacc=entry['validation_bacc'], test=test,
                  checkpoint_sha256=entry['checkpoint_sha256'],
                  policy='Interim test reporting only')
    (stage / ('result-%02d.json' % entry['index'])).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'worker'))
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.action == 'submit':
        submit()
    elif args.stage:
        worker(args.stage)
    else:
        parser.error('worker requires --stage')


if __name__ == '__main__':
    main()
