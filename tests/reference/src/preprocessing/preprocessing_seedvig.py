"""Build the canonical numeric subject-disjoint SEED-VIG split."""

import json
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import scipy.io


DATA_DIR = Path(
    "/gpfs/data/oermannlab/users/ml10266/Data/raw/SEED-VIG/Raw_Data"
)
LABELS_DIR = Path(
    "/gpfs/data/oermannlab/users/ml10266/Data/raw/SEED-VIG/perclos_labels"
)
OUTPUT_DIR = Path(
    "/gpfs/home/ml10266/eegfm_data/SEED-VIG/"
    "processed_cbramod_subject_split"
)

PROTOCOL_ID = "gr5_subject_disjoint"

SPLIT_FILES = {
    "train": (
        "1_20151124_noon_2.mat",
        "2_20151106_noon.mat",
        "3_20151024_noon.mat",
        "4_20151105_noon.mat",
        "4_20151107_noon.mat",
        "5_20141108_noon.mat",
        "5_20151012_night.mat",
        "6_20151121_noon.mat",
        "7_20151015_night.mat",
        "8_20151022_noon.mat",
        "9_20151017_night.mat",
        "10_20151125_noon.mat",
        "11_20151024_night.mat",
        "12_20150928_noon.mat",
        "13_20150929_noon.mat",
    ),
    "val": (
        "14_20151014_night.mat",
        "15_20151126_night.mat",
        "16_20151128_night.mat",
        "17_20150925_noon.mat",
    ),
    "test": (
        "18_20150926_noon.mat",
        "19_20151114_noon.mat",
        "20_20151129_night.mat",
        "21_20151016_noon.mat",
    ),
}


def _validate_raw_files():
    expected = {
        filename for filenames in SPLIT_FILES.values() for filename in filenames
    }
    data_files = {path.name for path in DATA_DIR.glob("*.mat")}
    label_files = {path.name for path in LABELS_DIR.glob("*.mat")}
    if data_files != expected or label_files != expected:
        raise RuntimeError(
            "SEED-VIG raw EEG and label files must exactly match the canonical numeric split manifest"
        )


def _load_recording(filename):
    eeg = scipy.io.loadmat(DATA_DIR / filename)["EEG"][0][0][0]
    labels = scipy.io.loadmat(LABELS_DIR / filename)["perclos"]
    if eeg.shape != (1_416_000, 17) or labels.shape != (885, 1):
        raise RuntimeError(f"unexpected SEED-VIG shape for {filename}")
    if not np.isfinite(eeg).all() or not np.isfinite(labels).all():
        raise RuntimeError(f"non-finite SEED-VIG value in {filename}")
    if labels.min() < 0 or labels.max() > 1:
        raise RuntimeError(f"PERCLOS label outside [0,1] in {filename}")
    samples = eeg.reshape(885, 8, 200, 17).transpose(0, 3, 1, 2)
    return samples, labels[:, 0]


def main():
    _validate_raw_files()
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging_dir = OUTPUT_DIR.with_name(f".{OUTPUT_DIR.name}.build-{job_id}")
    if staging_dir.exists():
        raise FileExistsError(f"staging path already exists: {staging_dir}")

    split_keys = {split: [] for split in SPLIT_FILES}
    database = lmdb.open(str(staging_dir), map_size=6_000_000_000)
    try:
        for split, filenames in SPLIT_FILES.items():
            for filename in filenames:
                samples, labels = _load_recording(filename)
                print(split, filename, samples.shape, labels.shape, flush=True)
                with database.begin(write=True) as transaction:
                    for index, (sample, label) in enumerate(zip(samples, labels)):
                        key = f"{filename[:-4]}-{index}"
                        transaction.put(
                            key.encode(),
                            pickle.dumps({"sample": sample, "label": label}),
                        )
                        split_keys[split].append(key)
        with database.begin(write=True) as transaction:
            transaction.put(b"__keys__", pickle.dumps(split_keys))
        database.sync()
    finally:
        database.close()

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol": "numeric subject-disjoint SEED-VIG split: 1-13 / 14-17 / 18-21",
        "train_subject_ids": list(range(1, 14)),
        "val_subject_ids": list(range(14, 18)),
        "test_subject_ids": list(range(18, 22)),
        "files": {key: list(value) for key, value in SPLIT_FILES.items()},
        "sample_counts": {key: len(value) for key, value in split_keys.items()},
    }
    with (staging_dir / "split_manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    staging_dir.rename(OUTPUT_DIR)
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
