"""첫 LR 비교의 실행 소스와 15개 설정을 고정한다. 이 단계는 제출하지 않는다."""
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/finetune_regularization_20260912'


def prepare():
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    array_path = CAMPAIGN / 'lr_array.json'
    if array_path.exists():
        print(array_path)
        return
    source = CAMPAIGN / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'finetune.py', source / 'finetune.py')
    (source / 'configs').mkdir()
    entries = []
    checkpoint = ROOT / 'outputs/gr9_1_epoch40.pth'
    for entry in manifest['entries']:
        if entry['arm'] != 'backbone_lr_x0p1':
            continue
        config = yaml.safe_load(Path(entry['config']).read_text())
        config['model']['checkpoint'] = str(checkpoint)
        output = CAMPAIGN / entry['arm'] / (entry['slug'] + '_seed' + str(entry['seed']))
        config['runtime']['output'] = str(output)
        path = source / 'configs' / Path(entry['config']).name
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        entries.append({**entry, 'config': str(path), 'result_dir': str(output),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(entries) == 15
    runner = source / 'run_array.py'
    runner.write_text('''import json, os, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent
entries = json.loads((root.parent / "lr_array.json").read_text())
entry = entries[int(os.environ["SLURM_ARRAY_TASK_ID"])]
subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                str(root / "finetune.py"), "--config", entry["config"], "--distributed"], check=True, cwd=root)
''')
    array_path.write_text(json.dumps(entries, indent=2) + '\n')
    files = [p for p in source.rglob('*') if p.is_file()]
    report = {'files': [{'path': str(p.relative_to(source)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(files)],
              'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'array': str(array_path), 'tasks': len(entries), 'max_concurrent_tasks': 3}
    (CAMPAIGN / 'source_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    for path in files:
        path.chmod(0o444)
    print(array_path)


if __name__ == '__main__':
    prepare()
