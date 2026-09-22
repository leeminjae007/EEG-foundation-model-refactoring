"""Launch a five-seed mask55 ISRUC or SEED-V fixed-HP experiment on GL40S."""

from __future__ import annotations

from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_mask55_hp_grid import digest

BASE = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-0342-gr2-d2-patch-dimension-mask55')
RESULTS = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
HP = dict(learning_rate=5e-4, weight_decay=.05, head_dropout=.3,
          early_stopping=dict(monitor='balanced_accuracy', min_epochs=15,
                              patience=10, min_delta=0.0))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('isruc', 'seedv'), default='isruc')
    args = parser.parse_args()
    dataset = args.dataset
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%y%m%d-%H%M%S')
    campaign = RESULTS / f'{stamp}-gr2-d2-mask55-{dataset}-lr5e-4-wd5e-2-drop03-five-gl40s'
    verified = json.loads((BASE / 'pretrain/verified.json').read_text())
    checkpoint = Path(verified['checkpoint'])
    if (not verified.get('strict_load') or verified.get('partial_epoch_smoke') or
            not checkpoint.is_file() or digest(checkpoint) != verified['sha256']):
        raise ValueError('Verified mask55 depth2 patch-dimension checkpoint failed')
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT, campaign / 'source', ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'outputs', 'results', '__pycache__', '*.pyc', '*.pth'))
    (campaign / 'pretrain').mkdir()
    write_json(campaign / 'pretrain/verified.json', verified)
    write_json(campaign / 'manifest.json', dict(
        experiment=campaign.name, preset='gr2-d2-patch-dimension-mask55',
        status='prepared', created_at_new_york=stamp, source_pretrain=str(BASE),
        publication_root=str(campaign / 'published'),
        checkpoint=str(checkpoint), checkpoint_sha256=verified['sha256'],
        datasets=[dataset], seeds=[42, 696, 1001, 1234, 3407],
        fixed_hyperparameters={dataset: HP},
        selection='Per-seed validation balanced accuracy; test reporting only'))
    subprocess.run([
        sys.executable, str(ROOT / 'scripts/submit_experiment_downstream.py'),
        '--experiment', str(campaign), '--checkpoint', str(checkpoint),
        '--datasets', dataset], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_experiment.py'),
                    '--experiment', str(campaign), '--submit'], check=True)
    manifest = json.loads((campaign / 'manifest.json').read_text())
    print(json.dumps(dict(campaign=str(campaign),
                          job=manifest['downstream_jobs'][0]['job'],
                          finalizer=manifest['finalizer_job'],
                          fixed_hp=HP), indent=2))


if __name__ == '__main__':
    main()
