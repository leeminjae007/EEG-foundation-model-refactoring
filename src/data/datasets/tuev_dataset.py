"""Direct loader for processed TUEV event pickle samples."""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.electrode_geometry import (
    CHB_MIT_16_CHANNELS,
    resolve_channel_coordinates,
    resolve_channel_regions,
)


TUEV_SPLIT_DIRECTORIES = {
    "train": "processed_train",
    "val": "processed_eval",
    "test": "processed_test",
}


class TUEVDataset(Dataset):
    channel_names = CHB_MIT_16_CHANNELS
    sample_rate = 200
    num_outputs = 6
    task = "multiclass"

    def __init__(self, data_dir, split):
        self.split = str(split)
        directory = Path(data_dir) / TUEV_SPLIT_DIRECTORIES[split]
        self.files = sorted(directory.glob("*.pkl"))
        self.subject_ids = tuple(
            (
                path.name.split("_", 1)[0]
                if self.split != "test"
                else "unavailable"
            )
            for path in self.files
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
        """All stored TUEV channels are already coordinate-enabled."""

    def __getitem__(self, index):
        path = self.files[index]
        with path.open("rb") as source:
            record = pickle.load(source)
        signal = record["signal"]
        if signal.shape != (16, 1000):
            raise ValueError(
                f"{path} has signal shape {signal.shape}, expected (16, 1000)"
            )
        label = int(record["label"][0]) - 1
        if not 0 <= label < self.num_outputs:
            raise ValueError(f"{path} has six-class label {label}")
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
