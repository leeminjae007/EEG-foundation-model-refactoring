"""Run pinned, unmodified CSBrain Siena functions with local paths only.

No filtering, channels, annotations, segments, labels, resampling or splits
are reimplemented here. Stop on upstream failure instead of silently skipping.
"""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'outputs/siena_preprocessing'
SOURCE = Path(__file__).resolve().parent / 'siena_official'
RAW = Path('/gpfs/data/oermannlab/users/ml10266/Data/raw/siena/physionet.org/files/siena-scalp-eeg/1.0.0')
OUTPUT = Path('/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model/processed/siena/processed')
HASHES = {'data_process.py': 'c078bc508b1ed7a394a38d7877939f64a09f0758f9662836e3eae6d6b910cca2',
          'json_generate.py': '4a7468c7cab25375b5c7988e9d2d1847a91ce21b42c94f55f0f796a82b2dc275'}
LOADER_SHA = 'a1f1e896714b2ff7a72ef637e5591b5c6fa7cb310ee2e6e1d7dd570fb966f4e7'


def verify_sources():
    for name, expected in HASHES.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Upstream source changed: {name}')
    if hashlib.sha256((ROOT / 'src/data/preprocessing/preprocessing_siena.py').read_bytes()).hexdigest() != LOADER_SHA:
        raise RuntimeError('Existing Siena loader changed')


def main():
    # Isolated upstream dependency; do not alter the shared training environment.
    AUDIT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    repair = json.loads((SOURCE / 'verified.json').read_text())
    if repair['status'] != 'verified':
        raise RuntimeError('Raw checksum repair is incomplete')
    records = [line.strip() for line in (RAW / 'RECORDS').read_text().splitlines() if line.strip().endswith('.edf')]
    subjects = sorted({Path(record).parent.name for record in records})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger_path = AUDIT / 'preprocessing_records.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for subject in subjects:
        sys.argv = [str(SOURCE / 'data_process.py'), '--patient_id', subject]
        namespace = runpy.run_path(sys.argv[0], run_name='preserved_csbrain_siena')
        process = namespace['process_edf']
        out = OUTPUT / 'processed_segments' / subject
        out.mkdir(parents=True, exist_ok=True)
        # Only paths are substituted; processing code and arguments are original.
        process.__globals__.update(base_dir=str(RAW), output_dir=str(out))
        files = sorted(f for f in os.listdir(RAW / subject) if f.endswith('.edf') and f.startswith(subject))
        for filename in files:
            key = f'{subject}/{filename}'
            if key in ledger:
                continue
            print(json.dumps({'processing': key}), flush=True)
            success, message = process(str(RAW / key), namespace['seizure_records'], str(out), sampling_rate=512)
            print(message, flush=True)
            if not success:
                raise RuntimeError(message)
            ledger[key] = message
            ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    if set(ledger) != set(records):
        raise RuntimeError('Processed record roster differs from official RECORDS')
    namespace = runpy.run_path(str(SOURCE / 'json_generate.py'), run_name='preserved_csbrain_siena_json')
    namespace['main'].__globals__.update(
        data_folder=str(OUTPUT / 'processed_segments'),
        save_folder_train=str(OUTPUT / 'train.json'),
        save_folder_val=str(OUTPUT / 'val.json'),
        save_folder_test=str(OUTPUT / 'test.json'))
    namespace['main']()
    verify_sources()
    summary = {'source_commit': '185aee55b24d0410a830df8dd08d03f675616998',
               'source_sha256': HASHES, 'existing_loader_sha256': LOADER_SHA,
               'path_overrides_only': True, 'records': len(ledger)}
    for split in ('train', 'val', 'test'):
        meta = json.loads((OUTPUT / f'{split}.json').read_text())
        summary[split] = {'samples': len(meta['subject_data']),
                         'subjects': sorted({d['subject_name'] for d in meta['subject_data']}),
                         'dataset_info': meta['dataset_info']}
    (AUDIT / 'preprocessing_complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
