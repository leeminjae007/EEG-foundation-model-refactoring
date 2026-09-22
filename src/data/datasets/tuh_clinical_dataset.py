"""Window reader for processed TUH seizure and slowing corpora."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.electrode_geometry import resolve_channel_coordinates, resolve_channel_regions


CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")


@lru_cache(maxsize=8)
def _signal(path):
    return np.load(path, mmap_mode="r")


class _TUHClinicalDataset(Dataset):
    channel_names = CHANNELS
    sample_rate = 200
    signal_length = 2000

    def __init__(self, data_dir, split):
        self.root = Path(data_dir)
        with (self.root / "manifest.csv").open(newline="") as source:
            self.rows = [row for row in csv.DictReader(source) if row["split"] == split]
        if not self.rows:
            raise ValueError("No " + split + " rows in " + str(self.root / "manifest.csv"))
        self.subject_ids = tuple(row["subject"] for row in self.rows)
        self.recording_ids = tuple(row["recording"] for row in self.rows)
        self.channel_coordinates, _ = resolve_channel_coordinates(self.channel_names)
        self.channel_region_ids = resolve_channel_regions(self.channel_names)

    def __len__(self):
        return len(self.rows)

    def enable_coordinate_only_channels(self):
        pass

    def __getitem__(self, index):
        row = self.rows[index]
        start = int(row["start"])
        path = self.root / "signals" / (row["recording"] + ".npy")
        signal = _signal(str(path))[:, start:start + self.signal_length].copy()
        if signal.shape != (19, self.signal_length):
            raise ValueError("Invalid TUH window at " + str(path) + ": " + str(signal.shape))
        bits = int(row["channel_mask"])
        validity = torch.tensor([(bits & (1 << i)) != 0 for i in range(19)], dtype=torch.bool)
        return {"x": torch.from_numpy(signal) / 100.0,
                "channel_coordinates": self.channel_coordinates,
                "channel_region_ids": self.channel_region_ids,
                "channel_validity": validity,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "subject_id": row["subject"],
                "recording_id": row["recording"],
                "sample_id": row["recording"] + "-" + row["start"]}


class TUSZDataset(_TUHClinicalDataset):
    num_outputs = 1
    task = "binary"


class TUSLDataset(_TUHClinicalDataset):
    num_outputs = 3
    task = "multiclass"
