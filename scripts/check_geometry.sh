#!/usr/bin/env bash
set -euo pipefail
source scripts/activate.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest tests/test_geometry.py tests/test_equivalence.py tests/test_schedule.py -q
python pretrain.py --device cpu --smoke
python - <<'PY'
import json
import torch
from pathlib import Path
p = Path('outputs/smoke/pretrain_geometry50_reve3cm_t2_15')
a = torch.load(p / 'last.pth', map_location='cpu')
assert a['config']['masking']['policy'] == 'geometry_tubelet'
assert a['config']['masking']['mask_ratio'] == .5
assert a['config']['masking']['radius_m'] == .03
assert a['config']['masking']['min_time_patches'] == 2
assert a['config']['masking']['max_time_patches'] == 15
assert a['extra']['partial_epoch_smoke']
assert a['optimizer']['state']
m = json.loads((p / 'metrics.jsonl').read_text().splitlines()[-1])
assert m['masking/target_tokens'] == 285
assert m['masking/context_tokens'] == 285
assert all(torch.isfinite(v).all() for v in a['model'].values())
print('Geometry actual-data optimizer smoke passed:', m['loss'])
PY
