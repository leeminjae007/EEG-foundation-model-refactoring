"""CBraMod ISRUC-Sleep Subgroup I preprocessing."""

from pathlib import Path

import numpy as np
from edf_ import read_raw_edf


RAW_ROOT = Path(
    "/gpfs/data/oermannlab/users/ml10266/Data/raw/isruc-original-group1"
)
OUTPUT_ROOT = Path(
    "/gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model/processed/"
    "isruc/processed_filter_35"
)
LABEL_TO_ID = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}


def main():
    sequence_index = 0
    label_index = 0
    total_epochs = 0

    for subject in range(1, 101):
        eeg_path = RAW_ROOT / str(subject) / f"{subject}.rec"
        label_path = RAW_ROOT / str(subject) / f"{subject}_1.txt"

        labels = np.asarray(
            [
                LABEL_TO_ID[line.strip()]
                for line in label_path.read_text().splitlines()
                if line.strip()
            ],
            dtype=np.int64,
        )

        raw = read_raw_edf(eeg_path, preload=True, verbose="ERROR")
        raw.filter(0.3, 35, fir_design="firwin", verbose="ERROR")
        raw.notch_filter(50, verbose="ERROR")
        signal = raw.to_data_frame().values[:, 1:]
        signal = signal[:, 2:8].T

        epoch_count = min(signal.shape[1] // 6000, len(labels))
        epoch_count -= epoch_count % 20
        signal = signal[:, :epoch_count * 6000]
        signal = signal.reshape(6, epoch_count, 6000).transpose(1, 0, 2)
        labels = labels[:epoch_count]
        signal = signal.reshape(-1, 20, 6, 6000)
        labels = labels.reshape(-1, 20)

        name = f"ISRUC-group1-{subject}"
        sequence_dir = OUTPUT_ROOT / "seq" / name
        label_dir = OUTPUT_ROOT / "labels" / name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sequence in signal:
            np.save(
                sequence_dir / f"{name}-{sequence_index}.npy",
                sequence,
            )
            sequence_index += 1
        for label_sequence in labels:
            np.save(
                label_dir / f"{name}-{label_index}.npy",
                label_sequence,
            )
            label_index += 1

        total_epochs += epoch_count
        print(
            f"{name}: epochs={epoch_count}, sequences={len(signal)}",
            flush=True,
        )

    print(
        f"complete: epochs={total_epochs}, sequences={sequence_index}",
        flush=True,
    )
    print(f"output: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
