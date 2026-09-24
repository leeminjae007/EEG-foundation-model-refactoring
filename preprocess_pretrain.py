"""Convert TUH/TUEG EDF recordings into the pretraining LMDB format.

The signal operations and window rules reproduce the training-data pipeline.
Input/output locations are provided at runtime and are never stored in this
source tree.  Use --edf for a bounded end-to-end check of one recording.
"""

import argparse
import pickle
import random
from pathlib import Path

import lmdb
import mne
import numpy as np


CHANNELS = {
    "01_tcp_ar": [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
        "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF",
        "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
        "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
        "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
    ],
    "02_tcp_le": [
        "EEG FP1-LE", "EEG FP2-LE", "EEG F3-LE", "EEG F4-LE",
        "EEG C3-LE", "EEG C4-LE", "EEG P3-LE", "EEG P4-LE",
        "EEG O1-LE", "EEG O2-LE", "EEG F7-LE", "EEG F8-LE",
        "EEG T3-LE", "EEG T4-LE", "EEG T5-LE", "EEG T6-LE",
        "EEG FZ-LE", "EEG CZ-LE", "EEG PZ-LE",
    ],
    "03_tcp_ar_a": [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
        "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF",
        "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
        "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
        "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
    ],
}


def montage_for(path):
    parts = set(path.parts)
    for montage, channels in CHANNELS.items():
        if montage in parts:
            return channels
    return None


def recording_samples(path):
    """Yield the same (19, 30, 200) microvolt windows as the source pipeline."""
    channels = montage_for(path)
    if channels is None:
        return
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if any(name not in raw.ch_names for name in channels):
        return
    raw.pick_channels(channels, ordered=True, verbose="ERROR")
    raw.resample(200, verbose="ERROR")
    raw.filter(l_freq=0.3, h_freq=75, verbose="ERROR")
    raw.notch_filter(60, verbose="ERROR")
    signal = raw.to_data_frame().values[:, 1:]
    points, count = signal.shape
    if points < 300 * 200 or count != 19:
        return
    remainder = points % (30 * 200)
    signal = signal[60 * 200:-(remainder + 60 * 200), :]
    if not signal.size:
        return
    samples = signal.reshape(-1, 30, 200, count).transpose(0, 3, 1, 2)
    for index, sample in enumerate(samples):
        if np.max(np.abs(sample)) < 100:
            yield f"{path.stem}_{index}", sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-root", type=Path, help="Directory containing EDF files")
    source.add_argument("--edf", type=Path, help="One EDF for a bounded pipeline check")
    parser.add_argument("--output", required=True, type=Path, help="New LMDB directory")
    parser.add_argument("--map-size-gb", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    files = ([args.edf] if args.edf else sorted(args.raw_root.rglob("*.edf")))
    if not files or any(not path.is_file() for path in files):
        raise ValueError("No readable EDF files")
    if args.raw_root:
        random.Random(args.seed).shuffle(files)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output LMDB directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    keys = []
    db = lmdb.open(str(args.output), map_size=args.map_size_gb * 1024**3)
    try:
        for path in files:
            for key, sample in recording_samples(path):
                with db.begin(write=True) as transaction:
                    if not transaction.put(key.encode(), pickle.dumps(sample), overwrite=False):
                        raise ValueError("Duplicate recording/window key")
                keys.append(key)
        if not keys:
            raise ValueError("No valid 30-second windows were produced")
        with db.begin(write=True) as transaction:
            transaction.put(b"__keys__", pickle.dumps(keys))
        db.sync()
    finally:
        db.close()
    print(f"recordings={len(files)} windows={len(keys)} shape=(19, 30, 200)")


if __name__ == "__main__":
    main()
