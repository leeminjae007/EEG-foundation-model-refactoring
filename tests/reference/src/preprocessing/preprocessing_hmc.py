"""Prepare the HMC sleep-staging dataset for downstream training.

The preprocessing contract follows the public REVE/NeuroLM pipeline: four EEG
derivations, 0.1--75 Hz band-pass filtering, 50 Hz notch filtering, 200 Hz
resampling, and 30-second AASM sleep-stage epochs. Subjects stay disjoint in a
deterministic 100/25/26 train/validation/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mne
import numpy as np


SAMPLE_RATE = 200
EPOCH_SECONDS = 30
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_SECONDS
CHANNELS = ("EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2")
CHANNEL_NAMES = ("F4", "C4", "O2", "C3")
STAGE_TO_ID = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, required=True,
        help="HMC recordings directory containing SN*.edf files",
    )
    parser.add_argument(
        "--processed", type=Path, required=True, help="Output dataset root",
    )
    parser.add_argument(
        "--n-jobs", type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
        help="MNE filtering/resampling worker count",
    )
    return parser.parse_args()


def _recording_files(raw_root: Path) -> List[Path]:
    files = sorted(
        path for path in raw_root.glob("SN*.edf")
        if not path.stem.endswith("_sleepscoring")
    )
    if len(files) != 151:
        raise RuntimeError(
            f"Expected 151 HMC recordings in {raw_root}, found {len(files)}"
        )
    missing = [
        path.with_name(f"{path.stem}_sleepscoring.txt")
        for path in files
        if not path.with_name(f"{path.stem}_sleepscoring.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} scoring files: {missing[:3]}")
    return files


def _split_recordings(files: List[Path]) -> Dict[str, List[Path]]:
    return {"train": files[:100], "val": files[100:125], "test": files[125:]}


def _read_stages(scoring_path: Path) -> Iterable[Tuple[float, int]]:
    with scoring_path.open(newline="") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            clean = {key.strip(): value.strip() for key, value in row.items()}
            annotation = clean.get("Annotation", "")
            try:
                onset = float(clean.get("Recording onset", "nan"))
                duration = float(clean.get("Duration", "nan"))
            except ValueError:
                continue
            if np.isclose(duration, EPOCH_SECONDS) and annotation in STAGE_TO_ID:
                yield onset, STAGE_TO_ID[annotation]


def _load_recording(recording_path: Path, n_jobs: int) -> np.ndarray:
    raw = mne.io.read_raw_edf(recording_path, preload=True, verbose="ERROR")
    missing = [channel for channel in CHANNELS if channel not in raw.ch_names]
    if missing:
        raw.close()
        raise ValueError(f"{recording_path.name} is missing channels {missing}")
    raw.pick(list(CHANNELS))
    raw.reorder_channels(list(CHANNELS))
    raw.filter(0.1, 75.0, n_jobs=n_jobs, verbose="ERROR")
    raw.notch_filter(50.0, n_jobs=n_jobs, verbose="ERROR")
    raw.resample(SAMPLE_RATE, n_jobs=n_jobs, verbose="ERROR")
    signal = raw.get_data(units="uV").astype(np.float32, copy=False)
    raw.close()
    return signal


def _write_subject(
    recording_path: Path, output_dir: Path, n_jobs: int,
) -> Dict[str, object]:
    subject = recording_path.stem
    marker = output_dir / f".{subject}.complete.json"
    if marker.is_file():
        return json.loads(marker.read_text())

    signal = _load_recording(recording_path, n_jobs)
    scoring_path = recording_path.with_name(f"{subject}_sleepscoring.txt")
    class_counts = np.zeros(len(STAGE_TO_ID), dtype=np.int64)
    written = 0
    skipped_bounds = 0
    for onset, label in _read_stages(scoring_path):
        start = int(round(onset * SAMPLE_RATE))
        end = start + EPOCH_SAMPLES
        if start < 0 or end > signal.shape[1]:
            skipped_bounds += 1
            continue
        sample = {
            "X": np.ascontiguousarray(signal[:, start:end]),
            "ch_names": list(CHANNEL_NAMES),
            "y": int(label),
        }
        output_path = output_dir / f"{subject}-{written:04d}.pkl"
        temporary_path = output_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output_path)
        class_counts[label] += 1
        written += 1

    stats = {
        "subject": subject,
        "epochs": written,
        "class_counts": class_counts.tolist(),
        "skipped_out_of_bounds": skipped_bounds,
    }
    marker.write_text(json.dumps(stats, sort_keys=True) + "\n")
    return stats


def main() -> None:
    args = _parse_args()
    recordings = _recording_files(args.raw.resolve())
    splits = _split_recordings(recordings)
    manifest: Dict[str, object] = {
        "dataset": "HMC sleep staging 1.1",
        "sample_rate": SAMPLE_RATE,
        "epoch_seconds": EPOCH_SECONDS,
        "channels": list(CHANNELS),
        "label_mapping": STAGE_TO_ID,
        "split_policy": "sorted subjects: first 100 train, next 25 val, last 26 test",
        "splits": {},
    }

    for split, split_files in splits.items():
        output_dir = args.processed / split
        output_dir.mkdir(parents=True, exist_ok=True)
        subject_stats = []
        for index, recording_path in enumerate(split_files, start=1):
            stats = _write_subject(recording_path, output_dir, args.n_jobs)
            subject_stats.append(stats)
            print(
                f"{split} {index}/{len(split_files)} {stats['subject']}: "
                f"epochs={stats['epochs']}", flush=True,
            )
        manifest["splits"][split] = {
            "subjects": len(split_files),
            "epochs": int(sum(item["epochs"] for item in subject_stats)),
            "class_counts": np.sum(
                [item["class_counts"] for item in subject_stats], axis=0
            ).astype(int).tolist(),
        }

    args.processed.mkdir(parents=True, exist_ok=True)
    (args.processed / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    print(f"complete: {args.processed}", flush=True)


if __name__ == "__main__":
    main()
