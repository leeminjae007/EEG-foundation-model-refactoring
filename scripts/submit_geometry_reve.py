"""승인된 geometry50/REVE3cm/2–15s 사전학습을 고정 소스로 한 번 제출한다."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/geometry50_reve3cm_t2_15_20260912'
EXCLUDE = 'a100-4004,a100-4007,a100-4011,a100-4012,a100-4021,a100-4023,a100-4028,a100-4032,a100-4033,a100-4037,a100-4042,a100-4044,a100-4045'


def atomic_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def submit():
    CAMPAIGN.mkdir(exist_ok=True)
    with (CAMPAIGN / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = CAMPAIGN / 'submission.json'
        if ledger.exists():
            print(ledger.read_text())
            return
        # unique job name is a recovery check if interrupted between sbatch and ledger write.
        job_name = 'eeg-geo50-r3-t2-15'
        active = subprocess.check_output(['squeue', '-h', '-u', str(ROOT.owner()),
                                          '--name', job_name, '-o', '%i'], text=True).strip()
        if active:
            raise RuntimeError('campaign job exists without a ledger; inspect before retrying: ' + active)
        if (CAMPAIGN / 'submission_intent.json').exists():
            raise RuntimeError('prior submission intent exists; reconcile sacct before retrying')
        source = CAMPAIGN / 'source'
        if source.exists():
            raise RuntimeError('source snapshot already exists; do not overwrite')
        source.mkdir()
        shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'scripts', source / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(ROOT / 'pretrain.py', source / 'pretrain.py')
        (source / 'configs').mkdir()
        shutil.copy2(ROOT / 'configs/gr9_1.yaml', source / 'configs/gr9_1.yaml')
        config = yaml.safe_load((ROOT / 'configs/pretrain.yaml').read_text())
        assert config['masking'] == {'policy': 'geometry_tubelet', 'mask_ratio': .5,
                                     'distance_metric': 'euclidean_m', 'radius_m': .03,
                                     'min_time_patches': 2, 'max_time_patches': 15}
        assert config['optimization']['epochs'] == 40
        assert config['optimization']['batch_size_per_gpu'] == 128
        assert config['seed'] == 42
        training = CAMPAIGN / 'training'
        training.mkdir()
        config['runtime']['output'] = str(training)
        config_path = source / 'configs/pretrain.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        python = ROOT / '.venv/bin/python'
        launch = source / 'launch.sh'
        launch.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
cd "{source}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NCCL_DEBUG=WARN
"{python}" scripts/check_pretrain_initialization.py --config configs/pretrain.yaml --baseline configs/gr9_1.yaml --output "{CAMPAIGN / 'initialization_gpu.json'}"
exec "{python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 pretrain.py --config configs/pretrain.yaml --distributed
''')
        files = [p for p in source.rglob('*') if p.is_file()]
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source_files': [{'path': str(p.relative_to(source)),
                                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)],
                    'config': config, 'global_batch': 512, 'gpus': 4, 'nodes': 1,
                    'initialization': 'fresh seed42; not resumed from GR9-1 epoch40',
                    'verification_job': '27393304', 'source_root': str(source),
                    'training_output': str(training),
                    'scientific_lineage': 'GR9-1 model with geometry union 50%, physical Euclidean radius 3cm, duration 2–15 seconds'}
        atomic_json(CAMPAIGN / 'manifest.json', manifest)
        for p in files:
            p.chmod(0o444)
        command = ['sbatch', '--parsable', '--job-name', job_name, '--account', 'system',
                   '--partition', 'a100_short,a100_long', '--nodes', '1', '--ntasks', '1',
                   '--gpus-per-task', 'a100:4', '--cpus-per-task', '32', '--mem', '128G',
                   '--time', '1-00:00:00', '--nice=0', '--exclude', EXCLUDE,
                   '--chdir', str(source), '--output', str(CAMPAIGN / 'slurm-%j.log'),
                   '--wrap', f'srun --ntasks=1 --gpus-per-task=a100:4 --cpu-bind=none --kill-on-bad-exit=1 bash {launch}']
        atomic_json(CAMPAIGN / 'submission_intent.json', {'command': command, 'job_name': job_name})
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            atomic_json(CAMPAIGN / 'submission_rejection.json', {
                'command': command, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr})
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('unexpected sbatch output: ' + result.stdout)
        report = {'job_id': job, 'submitted_utc': datetime.now(timezone.utc).isoformat(),
                  'command': command, 'manifest': str(CAMPAIGN / 'manifest.json'),
                  'source_root': str(source), 'training_output': str(training),
                  'stderr': result.stderr}
        atomic_json(ledger, report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    submit()
