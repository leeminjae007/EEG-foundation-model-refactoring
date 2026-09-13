"""Direct loader for processed HMC 30-second sleep-stage epochs."""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.electrode_geometry import (
    resolve_channel_coordinates,
    resolve_channel_regions,
)


class HMCDataset(Dataset):
    """Load the subject-disjoint HMC pickle splits produced by preprocessing."""

    channel_names = ("F4", "C4", "O2", "C3")
    sample_rate = 200
    num_outputs = 5
    task = "multiclass"

    def __init__(self, data_dir, split):
        self.files = sorted((Path(data_dir) / split).glob("SN*.pkl"))
        self.subject_ids = tuple(path.stem.split("-", 1)[0] for path in self.files)
        self.recording_ids = self.subject_ids
        self.channel_coordinates, coordinate_validity = (
            resolve_channel_coordinates(self.channel_names)
        )
        self.channel_region_ids = resolve_channel_regions(self.channel_names)
        self.channel_validity = torch.ones(len(self.channel_names), dtype=torch.bool)
        unresolved = ~coordinate_validity | self.channel_region_ids.lt(0)
        if unresolved.any():
            missing = [
                name for name, invalid in zip(
                    self.channel_names, unresolved.tolist()
                ) if invalid
            ]
            raise ValueError(f"HMC channels lack canonical geometry: {missing}")

    def __len__(self):
        return len(self.files)

    def enable_coordinate_only_channels(self):
        """All four HMC channels are already coordinate-enabled."""

    def __getitem__(self, index):
        path = self.files[index]
        with path.open("rb") as source:
            record = pickle.load(source)
        signal = record["X"]
        if tuple(signal.shape) != (4, 6000):
            raise ValueError(
                f"{path} has signal shape {signal.shape}, expected (4, 6000)"
            )
        channel_names = tuple(record.get("ch_names", ()))
        if channel_names != self.channel_names:
            raise ValueError(
                f"{path} has channels {channel_names}, expected {self.channel_names}"
            )
        label = int(record["y"])
        if not 0 <= label < self.num_outputs:
            raise ValueError(f"{path} has sleep-stage label {label}")
        return {
            "x": torch.as_tensor(signal, dtype=torch.float32) / 100.0,
            "channel_coordinates": self.channel_coordinates,
            "channel_region_ids": self.channel_region_ids,
            "channel_validity": self.channel_validity,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": self.subject_ids[index],
            "recording_id": self.recording_ids[index],
            "sample_id": path.name,
        }
