"""Sequence loader for processed ISRUC sequence/label NPY files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.electrode_geometry import (
    resolve_channel_coordinates,
    resolve_channel_regions,
)


ISRUC_CHANNELS = (
    "F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1",
)
ISRUC_SUBJECTS = {
    "train": range(1, 81),
    "val": range(81, 91),
    "test": range(91, 101),
}


class ISRUCDataset(Dataset):
    channel_names = ISRUC_CHANNELS
    sample_rate = 200
    num_outputs = 5
    task = "multiclass"

    def __init__(self, data_dir, split):
        root = Path(data_dir)
        self.samples = []
        for subject in ISRUC_SUBJECTS[split]:
            name = f"ISRUC-group1-{subject}"
            sequence_files = sorted((root / "seq" / name).glob("*.npy"))
            label_files = sorted((root / "labels" / name).glob("*.npy"))
            if len(sequence_files) != len(label_files):
                raise ValueError(f"{name} sequence/label file count differs")
            for sequence_path, label_path in zip(
                sequence_files, label_files
            ):
                self.samples.append(
                    (sequence_path, label_path, name)
                )
        self.subject_ids = tuple(sample[2] for sample in self.samples)
        self.recording_ids = tuple(
            sample[0].stem for sample in self.samples
        )
        self.channel_coordinates, _ = resolve_channel_coordinates(
            self.channel_names
        )
        self.channel_region_ids = resolve_channel_regions(
            self.channel_names
        )
        self.channel_validity = torch.ones(6, dtype=torch.bool)

    def __len__(self):
        return len(self.samples)

    def enable_coordinate_only_channels(self):
        """ISRUC already enables all six coordinate-resolved channels."""

    def __getitem__(self, index):
        sequence_path, label_path, subject_id = self.samples[index]
        signal = np.asarray(np.load(sequence_path), dtype=np.float32)
        labels = np.asarray(np.load(label_path), dtype=np.int64).reshape(-1)
        if signal.shape != (20, 6, 6000):
            raise ValueError(
                f"{sequence_path} has shape {signal.shape}, "
                "expected (20, 6, 6000)"
            )
        if labels.shape != (20,) or np.any(
            (labels < 0) | (labels >= self.num_outputs)
        ):
            raise ValueError(f"{label_path} has invalid sleep-stage labels")
        return {
            "x": torch.from_numpy(signal.copy()) / 100.0,
            "channel_coordinates": self.channel_coordinates,
            "channel_region_ids": self.channel_region_ids,
            "channel_validity": self.channel_validity,
            "label": torch.from_numpy(labels.copy()),
            "subject_id": subject_id,
            "recording_id": self.recording_ids[index],
            "sample_id": sequence_path.name,
        }
