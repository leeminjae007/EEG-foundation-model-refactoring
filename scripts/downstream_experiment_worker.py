"""Run one Slurm array entry, verify GPU binding and retain runtime evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--smoke-batches', type=int, default=0)
    args = parser.parse_args()
    entries = json.loads((args.experiment / 'downstream_entries.json').read_text())
    entry = entries[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    source = (args.source or args.experiment / 'source').resolve()
    sys.path.insert(0, str(source))
    import yaml
    cfg = yaml.safe_load(Path(entry['config']).read_text())
    output = Path(entry['output'])
    output.mkdir(parents=True, exist_ok=True)
    if not args.smoke_batches:
        verified = json.loads((args.experiment / 'pretrain/verified.json').read_text())
        checkpoint = Path(cfg['model']['checkpoint'])
        if str(checkpoint) != verified['checkpoint'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != verified['sha256']:
            raise ValueError('Downstream checkpoint differs from verified pretrain')
    from scripts.downstream_gpu_guard import allocated_gpu
    health = dict(job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                  cuda_probe='starting')
    try:
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.free',
                                         '--format=csv,noheader,nounits'], text=True)
        health.update(allocated_gpu(query, 12288))
        os.environ.update(CUDA_VISIBLE_DEVICES=health['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        import torch
        torch.ones(1, device='cuda:0').sum().item()
        torch.cuda.synchronize()
        health['cuda_probe'] = 'passed'
    except Exception as exc:
        health.update(cuda_probe='failed', error=repr(exc))
        raise
    finally:
        # Persist the diagnostic even when an allocated L40S is unhealthy.
        (output / 'gpu-health.json').write_text(json.dumps(health, indent=2))
    command = [sys.executable, str(source / 'scripts/run_downstream_with_results.py'),
               '--experiment', str(args.experiment), '--config', entry['config'],
               '--dataset', entry['dataset'], '--seed', str(entry['seed']), '--source', str(source),
               '--no-incremental-publish']
    if args.smoke_batches:
        command += ['--smoke-batches', str(args.smoke_batches)]
    elif (output / 'last.pth').exists() and not (output / 'result.json').exists():
        saved = torch.load(output / 'last.pth', map_location='cpu')
        if saved.get('extra', {}).get('partial_epoch_smoke') or saved['config'] != cfg:
            raise ValueError('Cannot resume partial smoke or mismatched downstream configuration')
        del saved
        command += ['--resume', str(output / 'last.pth')]
    started = time.monotonic()
    timing = dict(dataset=entry['dataset'], seed=entry['seed'], job=os.environ['SLURM_JOB_ID'],
                  started_utc=datetime.now(timezone.utc).isoformat(), gpu=health['gpu_uuid'],
                  node=health['node'], smoke=bool(args.smoke_batches), status='running')
    (output / 'runtime.json').write_text(json.dumps(timing, indent=2))
    try:
        subprocess.run(command, check=True)
        timing['status'] = 'completed'
    except Exception as exc:
        timing.update(status='failed', error=repr(exc))
        raise
    finally:
        timing.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        (output / 'runtime.json').write_text(json.dumps(timing, indent=2))


if __name__ == '__main__':
    main()
