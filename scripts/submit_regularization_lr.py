"""승인된 첫 LR 비교 15개를 한 번 제출한다. 후속 후보는 제출하지 않는다."""
import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/finetune_regularization_20260912'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def submit():
    with (CAMPAIGN / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = CAMPAIGN / 'submission.json'
        if ledger.exists():
            print(ledger.read_text())
            return
        intent = CAMPAIGN / 'submission_intent.json'
        if intent.exists():
            raise RuntimeError('prior submission intent exists; reconcile scheduler before retrying')
        source = CAMPAIGN / 'source'
        report = json.loads((CAMPAIGN / 'source_manifest.json').read_text())
        for item in report['files']:
            assert hashlib.sha256((source / item['path']).read_bytes()).hexdigest() == item['sha256']
        assert hashlib.sha256(Path(report['checkpoint']).read_bytes()).hexdigest() == report['checkpoint_sha256']
        smoke = json.loads((CAMPAIGN / 'smoke_validation.json').read_text())
        assert len(smoke) == 3 and all(item['passed'] for item in smoke)
        name = 'gr91-reg-backbone-lr01'
        active = subprocess.check_output(['squeue', '-h', '-u', ROOT.owner(), '--name', name, '-o', '%i'], text=True).strip()
        if active:
            raise RuntimeError('matching jobs already exist: ' + active)
        python = ROOT / '.venv/bin/python'
        wrapper = ('unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; '
                   f'exec srun --ntasks=1 --gpus-per-task=l40s:1 --cpu-bind=none --kill-on-bad-exit=1 {python} {source / "run_array.py"}')
        command = ['sbatch', '--parsable', '--job-name', name, '--account', 'system',
                   '--partition', 'gl40s_dev,gl40s_short,gl40s_long', '--array', '0-14%3',
                   '--nodes', '1', '--ntasks', '1', '--gpus-per-task', 'l40s:1',
                   '--cpus-per-task', '8', '--mem', '32G', '--time', '1-00:00:00', '--nice=0',
                   '--chdir', str(source), '--output', str(CAMPAIGN / 'slurm-%A_%a.log'), '--wrap', wrapper]
        write_json(intent, {'command': command})
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            write_json(CAMPAIGN / 'submission_rejection.json', {'stderr': result.stderr, 'stdout': result.stdout, 'returncode': result.returncode})
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit()
        value = {'job_id': job, 'array': '0-14%3', 'tasks': 15, 'arm': 'backbone_lr_x0p1',
                 'submitted_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                 'manifest': str(CAMPAIGN / 'lr_array.json'), 'verification_job': '27393550',
                 'other_arms': 'prepared only, not submitted'}
        write_json(ledger, value)
        manifest_path = CAMPAIGN / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['status'] = 'lr_arm_submitted_other_arms_prepared'
        manifest['submission'] = str(ledger)
        write_json(manifest_path, manifest)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    submit()
