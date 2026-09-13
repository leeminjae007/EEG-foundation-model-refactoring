#!/usr/bin/env bash
set -euo pipefail
source scripts/activate.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root=Path('outputs/finetune_regularization_20260912').resolve()
entries=json.loads((root/'lr_array.json').read_text())
for entry in entries:
    if entry['seed'] != 42:
        continue
    subprocess.run([sys.executable, str(root/'source/finetune.py'), '--config', entry['config'], '--device', 'cpu', '--smoke'], check=True, cwd=root/'source')
import torch,yaml
reports=[]
for entry in entries:
    if entry['seed'] != 42:
        continue
    config=yaml.safe_load(Path(entry['config']).read_text())
    payload=torch.load(root/'source/outputs/smoke'/config['data']['dataset']/'last.pth',map_location='cpu')
    assert payload['extra']['partial_epoch_smoke']
    assert payload['optimizer']['state']
    groups=payload['optimizer']['param_groups']
    assert [g['name'] for g in groups]==['tokenizer','encoder','head']
    assert abs(groups[0]['lr']/groups[2]['lr']-.1)<1e-10
    assert abs(groups[1]['lr']/groups[2]['lr']-.1)<1e-10
    assert all(torch.isfinite(v).all() for v in payload['model'].values())
    reports.append({'dataset':entry['dataset'],'passed':True,'optimizer_groups':[{'name':g['name'],'lr':g['lr']} for g in groups]})
(root/'smoke_validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print('All three actual-data LR-group smoke checks passed.')
PY
