"""Resume an audited TUAB checkpoint with explicit allocated-GPU binding."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

root = Path(sys.argv[1]).resolve()
alias = sys.argv[2]
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
entry = json.loads((root / alias / 'entries.json').read_text())[index]
source = root / 'source'
sys.path.insert(0, str(source))
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from scripts.downstream_gpu_guard import allocated_gpu

output = Path(entry['output'])
state = dict(job=os.environ['SLURM_JOB_ID'], dataset='tuab', seed=entry['seed'],
             node=os.environ.get('SLURMD_NODENAME'), started_utc=datetime.now(timezone.utc).isoformat(),
             status='starting', duration_scope='checkpoint_resume_and_evaluation')
started = time.monotonic()
try:
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True)
    health = allocated_gpu(query, 12288)
    os.environ.update(CUDA_VISIBLE_DEVICES=health['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    import torch
    import yaml
    torch.ones(1, device='cuda:0').sum().item()
    torch.cuda.synchronize()
    health.update(cuda_probe='passed', job=state['job'], node=state['node'])
    (output / 'gpu-health.json').write_text(json.dumps(health, indent=2))
    config = yaml.safe_load(Path(entry['config']).read_text())
    resume = output / 'last.pth'
    if not resume.exists():
        resume = Path(entry['resume'])
    saved = torch.load(resume, map_location='cpu')
    normalized = json.loads(json.dumps(config))
    normalized['runtime']['output'] = saved['config']['runtime']['output']
    assert normalized == saved['config'], 'Only relocation of runtime.output is allowed'
    assert not saved['extra'].get('partial_epoch_smoke')
    assert 17 <= saved['epoch'] <= config['optimization']['epochs'] == 20
    assert saved['optimizer']['state'] and saved['scheduler'] and saved['rng_states']
    state.update(status='running', resume_epoch=saved['epoch'], resume_step=saved['extra']['step'], gpu=health['gpu_uuid'])
    del saved
    (output / 'runtime.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    if not (output / 'result.json').exists():
        subprocess.run([sys.executable, str(source / 'finetune.py'), '--config', entry['config'], '--resume', str(resume)], cwd=source, check=True)
    json.loads((output / 'result.json').read_text())
    state['status'] = 'completed'
except BaseException as exc:
    state.update(status='failed', error=repr(exc))
    raise
finally:
    state.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    (output / 'runtime.json').write_text(json.dumps(state, indent=2))
