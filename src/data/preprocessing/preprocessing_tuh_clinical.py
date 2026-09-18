"""Prepare TUSZ v2.0.6 and TUSL v2.0.1 for the GR9-1 downstream model.

The annotation handling follows EEG-FM-Bench's tusz.py and tusl.py (Apache-2.0),
adapted to this repository's fixed 200 Hz, 19-electrode input contract.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np


CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")
LABELS = {"tusz": {"bckg": 0, "seiz": 1},
          "tusl": {"seiz": 0, "slow": 1, "bckg": 2}}
RAW_NAMES = {"tusz": ("tuh_eeg_seizure", "v2.0.6"),
             "tusl": ("tuh_eeg_slowing", "v2.0.1")}
WINDOW_SAMPLES = 2000
FIELDS = ("split", "recording", "subject", "start", "label", "channel_mask")


def annotation_rows(path, allowed):
    """Drop low-confidence and duplicated channel annotations, as in EEG-FM-Bench."""
    events = set()
    with path.open(newline="") as source:
        reader = csv.DictReader(line for line in source if not line.startswith("#"))
        for row in reader:
            label = row["label"].strip().lower()
            if label not in allowed or float(row["confidence"]) < 0.5:
                continue
            start, stop = float(row["start_time"]), float(row["stop_time"])
            if np.isfinite(start) and np.isfinite(stop) and stop > start >= 0:
                events.add((round(start, 4), round(stop, 4), label))
    return sorted(events)


def signal_and_mask(edf):
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
    try:
        names = {}
        for index, name in enumerate(raw.ch_names):
            parts = name.strip().upper().split()
            electrode = parts[-1].split("-")[0]
            if electrode in CHANNELS and electrode not in names:
                names[electrode] = index
        if len(names) < 16:
            raise ValueError("fewer than 16 canonical electrodes")
        # This matches the existing TUAB/TUEV filters and physical-unit scale.
        raw.filter(0.3, min(75.0, raw.info["sfreq"] / 2.0 - 1.0), verbose="ERROR")
        if raw.info["sfreq"] > 120:
            raw.notch_filter(60.0, verbose="ERROR")
        raw.resample(200, verbose="ERROR")
        selected = np.zeros((len(CHANNELS), raw.n_times), dtype=np.float32)
        mask = 0
        for channel, electrode in enumerate(CHANNELS):
            if electrode in names:
                selected[channel] = raw.get_data(picks=[names[electrode]])[0] * 1e6
                mask |= 1 << channel
        return selected, mask
    finally:
        raw.close()


def tusz_windows(events, length):
    # TUH CSVs may annotate only short selected events, not an entire EDF.
    # Anchor windows on actual events rather than treating unannotated time as background.
    if length < WINDOW_SAMPLES:
        return
    candidates = defaultdict(set)
    seizure = [(round(a * 200), round(b * 200)) for a, b, label in events if label == "seiz"]
    for begin, end, label in events:
        begin, end = round(begin * 200), round(end * 200)
        if begin >= length or end <= begin:
            continue
        if end - begin < WINDOW_SAMPLES:
            starts = [max(0, min(length - WINDOW_SAMPLES,
                                 (begin + end) // 2 - WINDOW_SAMPLES // 2))]
        else:
            starts = list(range(begin, end - WINDOW_SAMPLES + 1, WINDOW_SAMPLES))
            starts.append(end - WINDOW_SAMPLES)
        for start in starts:
            if 0 <= start <= length - WINDOW_SAMPLES:
                candidates[start].add(label)
    for start, labels in sorted(candidates.items()):
        overlap = any(max(start, a) < min(start + WINDOW_SAMPLES, b)
                      for a, b in seizure)
        if "seiz" in labels:
            yield start, 1
        elif "bckg" in labels and not overlap:
            yield start, 0


def tusl_windows(events, length):
    seen = {}
    for a, b, label in events:
        start = round(a * 200)
        stop = round(b * 200)
        if start < 0 or start + WINDOW_SAMPLES > length or stop - start < WINDOW_SAMPLES - 2:
            continue
        value = LABELS["tusl"][label]
        if start in seen and seen[start] != value:
            seen[start] = None
        else:
            seen.setdefault(start, value)
    for start, label in sorted(seen.items()):
        if label is not None:
            yield start, label


def process_recording(job):
    dataset, edf_name, edf_root, output_root = job
    edf = Path(edf_name)
    subject = edf.relative_to(edf_root).parts[-4] if dataset == "tusz" else edf.relative_to(edf_root).parts[0]
    source_split = edf.relative_to(edf_root).parts[0] if dataset == "tusz" else "unsplit"
    recording = "__".join(edf.relative_to(edf_root).with_suffix("").parts)
    output = output_root / "signals" / (recording + ".npy")
    try:
        if dataset == "tusz":
            annotation = edf.with_suffix(".csv_bi")
            if not annotation.exists():
                raise FileNotFoundError(annotation)
            events = annotation_rows(annotation, LABELS[dataset])
        else:
            annotations = sorted(edf.parent.glob(edf.stem + "_*.csv"))
            if not annotations:
                raise FileNotFoundError("event CSVs for " + str(edf))
            events = sorted(set(event for path in annotations
                                for event in annotation_rows(path, LABELS[dataset])))
        signal, channel_mask = signal_and_mask(edf)
        windows = (tusz_windows(events, signal.shape[1]) if dataset == "tusz"
                   else tusl_windows(events, signal.shape[1]))
        rows = [{"split": source_split, "recording": recording, "subject": subject,
                 "start": start, "label": label, "channel_mask": channel_mask}
                for start, label in windows]
        if not rows:
            raise ValueError("no valid annotated windows")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".npy.partial")
        with temp.open("wb") as destination:
            np.save(destination, signal)
        os.replace(str(temp), str(output))
        return rows, None
    except Exception as error:
        return [], {"edf": str(edf), "reason": repr(error)}


def subject_split(rows):
    """Deterministic 80/10/10 subject split for TUSL, balanced by event labels."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(row)
    subjects = sorted(groups)
    totals = Counter(int(row["label"]) for row in rows)
    holdout = max(1, round(len(subjects) * .1))
    best = None
    assignments = None
    for trial in range(1000):
        order = subjects.copy()
        random.Random(20260916 + trial).shuffle(order)
        proposal = {subject: ("test" if i < holdout else
                              "val" if i < 2 * holdout else "train")
                    for i, subject in enumerate(order)}
        counts = {split: Counter(int(row["label"]) for subject in subjects
                                 if proposal[subject] == split for row in groups[subject])
                  for split in ("train", "val", "test")}
        missing = sum(counts[split][label] == 0 for split in counts for label in totals)
        imbalance = sum(((counts[split][label] / totals[label]) - target) ** 2
                        for split, target in (("train", .8), ("val", .1), ("test", .1))
                        for label in totals)
        score = (missing, imbalance)
        if best is None or score < best:
            best, assignments = score, proposal
    for row in rows:
        row["split"] = assignments[row["subject"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("tusz", "tusl"))
    parser.add_argument("--raw-base", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    name, version = RAW_NAMES[args.dataset]
    edf_root = args.raw_base / "tuh_eeg" / name / version / "edf"
    output_root = args.output_base / args.dataset
    edfs = sorted(edf_root.rglob("*.edf"))
    if args.limit:
        edfs = edfs[:args.limit]
    if not edfs:
        raise RuntimeError("No EDFs found under " + str(edf_root))
    jobs = [(args.dataset, str(edf), edf_root, output_root) for edf in edfs]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_recording, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            batch, error = future.result()
            rows.extend(batch)
            if error:
                errors.append(error)
            if index % 100 == 0:
                print("processed", index, "/", len(jobs), "windows", len(rows), "errors", len(errors), flush=True)
    if args.dataset == "tusz":
        for row in rows:
            row["split"] = {"train": "train", "dev": "val", "eval": "test"}[row["split"]]
        # Some historical TUSZ releases repeat patients across official folders.
        # Keep the official evaluation set and remove lower-priority duplicates.
        test_subjects = {row["subject"] for row in rows if row["split"] == "test"}
        val_subjects = {row["subject"] for row in rows if row["split"] == "val"}
        original_count = len(rows)
        rows = [row for row in rows if row["split"] == "test" or
                (row["split"] == "val" and row["subject"] not in test_subjects) or
                (row["split"] == "train" and row["subject"] not in test_subjects | val_subjects)]
        excluded_for_overlap = original_count - len(rows)
    else:
        subject_split(rows)
        excluded_for_overlap = 0
    partitions = {split: {row["subject"] for row in rows if row["split"] == split}
                  for split in ("train", "val", "test")}
    overlaps = {"train_val": sorted(partitions["train"] & partitions["val"]),
                "train_test": sorted(partitions["train"] & partitions["test"]),
                "val_test": sorted(partitions["val"] & partitions["test"])}
    if any(overlaps.values()):
        raise RuntimeError("Patient leakage across splits: " + repr(overlaps))
    if not all(partitions.values()):
        raise RuntimeError("One or more splits are empty")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.csv"
    temp = manifest.with_suffix(".csv.partial")
    with temp.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["recording"], row["start"])))
    os.replace(str(temp), str(manifest))
    audit = {"dataset": args.dataset, "raw_version": version, "recordings_found": len(edfs),
             "recordings_failed": len(errors), "errors": errors,
             "windows": len(rows), "label_counts": dict(Counter(str(row["label"]) for row in rows)),
             "split_counts": dict(Counter(row["split"] for row in rows)),
             "split_label_counts": {split: dict(Counter(str(row["label"]) for row in rows
                                                      if row["split"] == split))
                                    for split in partitions},
             "windows_excluded_for_subject_overlap": excluded_for_overlap,
             "subjects": {split: len(subjects) for split, subjects in partitions.items()},
             "subject_overlap": overlaps}
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: value for key, value in audit.items() if key != "errors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
