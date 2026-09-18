"""CPU-only local publisher for repaired TUAB and BCIC beam campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import monitor_experiment_results as publisher


REMOTE = '/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model'
ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'outputs/results/monitor-repaired-groups.json'


def read(ssh: list[str], relative: str):
    try:
        return publisher.remote_json(ssh, REMOTE + '/' + relative)
    except Exception:
        return None


def normalized(entries, alias):
    return {
        'submitted_entries': [
            {'slug': 'tuab' if alias.startswith('tuab-') else 'bciciv2a',
             'dataset': 'tuab' if alias.startswith('tuab-') else 'bciciv2a',
             'display_name': 'TUAB' if alias.startswith('tuab-') else 'BCIC-IV-2a',
             'seed': int(entry['seed']), 'output': entry['output']}
            for entry in entries
        ],
        'reused_entries': [],
    }


def publish_group(ssh, entries, alias, stamp, literature):
    manifest = normalized(entries, alias)
    if len(entries) != 5 or {int(e['seed']) for e in entries} != publisher.SEEDS:
        raise ValueError(alias + ': five fixed seeds required')
    results, missing = publisher.fetch_results(ssh, manifest['submitted_entries'])
    if missing:
        return {'status': 'waiting', 'missing': missing}
    folder = publisher.RESULTS_ROOT / f"{stamp}-{manifest['submitted_entries'][0]['slug']}-{alias}"
    if all((folder / name).is_file() for name in ('results.csv', 'results.md', 'seed_results.csv')):
        return {'status': 'complete', 'datasets': 1}
    sections = publisher.publish(manifest, results, stamp, alias, literature)
    return {'status': 'complete', 'datasets': len(sections)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    ssh = publisher.ssh_base('bigpurple.nyumc.org', '10.189.18.56', 'bigpurple.nyumc.org')
    literature = json.loads(publisher.LITERATURE.read_text(encoding='utf-8'))
    completed = set()
    while True:
        status = {}
        tuab = read(ssh, 'outputs/tuab_recovery_retry2_20260915/manifest.json')
        if tuab:
            for arm in ('backbone_lr_x0p1', 'head_first2_backbone_lr_x0p1', 'head_h4'):
                alias = 'tuab-' + arm.replace('_', '-')
                if alias in completed:
                    status[alias] = {'status': 'complete'}
                    continue
                entries = [e for e in tuab['entries'] if e['arm'] == arm]
                status[alias] = publish_group(ssh, entries, alias, '09160320', literature)
                if status[alias]['status'] == 'complete':
                    completed.add(alias)
        beam = read(ssh, 'outputs/bciciv2a_beam_w0_20260915/final_summary.json')
        if beam and 'bcbeam-final' not in completed:
            plan = read(ssh, 'outputs/bciciv2a_beam_w0_20260915/stages/final/plan.json')
            if plan and beam.get('rows') and len(beam['rows']) == 5:
                status['bcbeam-final'] = publish_group(ssh, plan['entries'], 'bcbeam-final',
                                                       '09160320', literature)
                if status['bcbeam-final']['status'] == 'complete':
                    completed.add('bcbeam-final')
        else:
            status['bcbeam-final'] = {'status': 'complete' if 'bcbeam-final' in completed else 'waiting_for_final'}
        publisher.atomic_text(STATUS, json.dumps(status, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({key: value['status'] for key, value in status.items()}), flush=True)
        if len(completed) == 4 or args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
