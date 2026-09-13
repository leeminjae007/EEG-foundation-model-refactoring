"""Preprocess the official BCI Competition IV 2a GDF release into LMDB."""

from pathlib import Path
import pickle

import lmdb
import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter, resample


RAW_ROOT = Path(
    "/gpfs/data/oermannlab/users/ml10266/Data/raw/bcic-iv-2a"
)
OUTPUT_PATH = Path(
    "/gpfs/data/oermannlab/users/ml10266/Data/eeg_foundation_downstream/"
    "bcic-iv-2a/processed_inde_avg_filter"
)

FILES_BY_SPLIT = {
    "train": [
        "A01E", "A01T", "A02E", "A02T", "A03E",
        "A03T", "A04E", "A04T", "A05E", "A05T",
    ],
    "val": ["A06E", "A06T", "A07E", "A07T"],
    "test": ["A08E", "A08T", "A09E", "A09T"],
}


def butter_bandpass(low_cut, high_cut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    return butter(
        order,
        [low_cut / nyquist, high_cut / nyquist],
        btype="band",
    )


def load_session(session_name):
    gdf_path = RAW_ROOT / f"{session_name}.gdf"
    label_path = RAW_ROOT / "true_labels" / f"{session_name}.mat"
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")

    if raw.info["sfreq"] != 250.0:
        raise ValueError(f"{gdf_path}: expected 250 Hz, got {raw.info['sfreq']}")
    if len(raw.ch_names) != 25:
        raise ValueError(f"{gdf_path}: expected 22 EEG + 3 EOG channels")

    trial_onsets = np.asarray([
        int(round(onset * raw.info["sfreq"]))
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description
        )
        if description == "768"
    ])
    labels = loadmat(label_path)["classlabel"].reshape(-1).astype(np.int64) - 1
    if len(trial_onsets) != 288 or len(labels) != 288:
        raise ValueError(
            f"{session_name}: expected 288 trials, got "
            f"{len(trial_onsets)} onsets and {len(labels)} labels"
        )

    # MNE returns SI units. The source MATLAB conversion used microvolts.
    eeg = raw.get_data(picks=np.arange(22)) * 1e6
    return eeg, trial_onsets, labels


def preprocess_trial(eeg, start, end, filter_coefficients):
    sample = eeg[:, start:end]
    sample = sample - np.mean(sample, axis=0, keepdims=True)
    sample = lfilter(*filter_coefficients, sample, axis=-1)
    sample = sample[:, 2 * 250:6 * 250]
    if sample.shape != (22, 1000):
        raise ValueError(f"expected trial crop (22, 1000), got {sample.shape}")
    sample = resample(sample, 800, axis=-1)
    return sample.reshape(22, 4, 200)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(OUTPUT_PATH), map_size=2 * 1024**3)
    keys_by_split = {split: [] for split in FILES_BY_SPLIT}
    filter_coefficients = butter_bandpass(0.3, 40.0, 250.0, order=5)

    with db.begin(write=True) as transaction:
        for split, sessions in FILES_BY_SPLIT.items():
            for session_name in sessions:
                eeg, onsets, labels = load_session(session_name)
                ends = np.concatenate([onsets[1:], [eeg.shape[1]]])
                for trial_index, (start, end, label) in enumerate(
                    zip(onsets, ends, labels)
                ):
                    sample = preprocess_trial(
                        eeg, int(start), int(end), filter_coefficients
                    )
                    run_index = 3 + trial_index // 48
                    index_in_run = trial_index % 48
                    key = f"{session_name}-{run_index}-{index_in_run}"
                    transaction.put(
                        key.encode(),
                        pickle.dumps({"sample": sample, "label": int(label)}),
                    )
                    keys_by_split[split].append(key)
                print(
                    f"{split} {session_name}: {len(labels)} trials",
                    flush=True,
                )

        transaction.put(b"__keys__", pickle.dumps(keys_by_split))

    db.sync()
    db.close()
    counts = {split: len(keys) for split, keys in keys_by_split.items()}
    print(f"complete: {counts}, total={sum(counts.values())}", flush=True)
    print(f"output: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
