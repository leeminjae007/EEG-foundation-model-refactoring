"""Direct loader for processed TUAB 10-second pickle segments."""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.electrode_geometry import (
    CHB_MIT_16_CHANNELS,
    resolve_channel_coordinates,
    resolve_channel_regions,
)


class TUABDataset(Dataset):
    channel_names = CHB_MIT_16_CHANNELS
    sample_rate = 200
    num_outputs = 2
    task = "binary"

    def __init__(self, data_dir, split):
        self.files = sorted((Path(data_dir) / split).glob("*.pkl"))
        self.subject_ids = tuple(
            path.name.split("_", 1)[0] for path in self.files
        )
        self.recording_ids = tuple(
            path.stem.rsplit("_", 1)[0] for path in self.files
        )
        self.channel_coordinates, _ = resolve_channel_coordinates(
            self.channel_names
        )
        self.channel_region_ids = resolve_channel_regions(
            self.channel_names
        )
        self.channel_validity = torch.ones(16, dtype=torch.bool)

    def __len__(self):
        return len(self.files)

    def enable_coordinate_only_channels(self):
        """All stored TUAB channels are already coordinate-enabled."""

    def __getitem__(self, index):
        path = self.files[index]
        with path.open("rb") as source:
            record = pickle.load(source)
        signal = record["X"]
        if signal.shape != (16, 2000):
            raise ValueError(
                f"{path} has signal shape {signal.shape}, expected (16, 2000)"
            )
        label = int(record["y"])
        if label not in (0, 1):
            raise ValueError(f"{path} has binary label {label}")
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
