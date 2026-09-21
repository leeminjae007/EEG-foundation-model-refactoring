"""Record the applied in-place Slurm partition/time updates in campaign manifests."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_mask55_hp_grid import write_json

ROOT = Path('/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results')
PARTITIONS = 'gl40s_dev,gl40s_short,gl40s_long'
FORMAL = ROOT / '260921-1733-gr2-d2-mask55-three-l40s-no-smoke/manifest.json'
CLINICAL = ROOT / 'l40s-earlystop-replacements/260921-1738-gr2-d2-mask55-isruc-hmc-siena-grid/manifest.json'


def main():
    now = datetime.now(timezone.utc).isoformat()
    formal = json.loads(FORMAL.read_text())
    for entry in formal['downstream_jobs']:
        if entry['dataset'] == 'mumtaz':
            entry['partitions'] = 'gl40s_short,gl40s_long'
            entry['scheduling_note'] = 'Five seeds were already running; allocation left unchanged.'
        else:
            entry['partitions'] = PARTITIONS
            entry['time'] = '04:00:00'
            entry['scheduling_note'] = 'Pending Slurm array updated in place; no resubmission.'
    formal['scheduling_updated_utc'] = now
    write_json(FORMAL, formal)
    clinical = json.loads(CLINICAL.read_text())
    clinical['resources']['partitions'] = PARTITIONS
    clinical['resources']['time_limits'] = {dataset: '04:00:00'
                                            for dataset in ('isruc', 'hmc', 'siena')}
    for entry in clinical['jobs']:
        entry['partitions'] = PARTITIONS
        entry['time'] = '04:00:00'
        entry['scheduling_note'] = 'Pending Slurm array updated in place; no resubmission.'
    clinical['scheduling_updated_utc'] = now
    write_json(CLINICAL, clinical)
    print(json.dumps(dict(formal=str(FORMAL), clinical=str(CLINICAL),
                          partitions=PARTITIONS, time='04:00:00',
                          mumtaz='running allocation unchanged'), indent=2))


if __name__ == '__main__':
    main()
