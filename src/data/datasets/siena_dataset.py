"""Adapter for the preserved CSBrain Siena JSON/pickle preprocessing output.

The official preprocessing remains untouched: it writes 29-channel, 512 Hz,
10-second pickles and split JSON indices.  This adapter applies exactly the
original Siena loader's model-side conversion (Fourier resample to 200 Hz and
``X * 10000``) and exposes the canonical downstream sample dictionary.
"""

import json
import pickle
from pathlib import Path

from scipy.signal import resample
import torch
from torch.utils.data import Dataset

from src.data.electrode_geometry import (
    resolve_channel_coordinates,
    resolve_channel_regions,
)


SIENA_CHANNELS = (
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fc1", "Fc5",
    "Cp1", "Cp5", "F9", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4",
    "O2", "F8", "T4", "T6", "Fc2", "Fc6", "Cp2", "Cp6", "F10",
)


class SienaDataset(Dataset):
    """Read official Siena splits without changing their preprocessing contract."""

    channel_names = SIENA_CHANNELS
    source_sample_rate = 512
    sample_rate = 200
    segment_seconds = 10
    task = "binary"
    num_outputs = 1

    def __init__(self, data_dir, split):
        index_path = Path(data_dir) / f"{split}.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        info = payload["dataset_info"]
        if int(info["sampling_rate"]) != self.source_sample_rate:
            raise ValueError(f"{index_path} has unexpected sample rate")
        if tuple(info["ch_names"]) != self.channel_names:
            raise ValueError(f"{index_path} has unexpected channel order")
        self.samples = tuple(payload["subject_data"])
        self.subject_ids = tuple(str(sample["subject_name"]) for sample in self.samples)
        self.recording_ids = tuple(
            Path(sample["file"]).parent.name for sample in self.samples
        )
        self.channel_coordinates, coordinate_validity = resolve_channel_coordinates(
            self.channel_names
        )
        self.channel_region_ids = resolve_channel_regions(self.channel_names)
        # The current backbone consumes coordinates/validity; region ids are
        # retained as metadata. All official channels with valid coordinates,
        # including F9/F10, are kept rather than dropping source electrodes.
        self.channel_validity = coordinate_validity.clone()
        if not bool(self.channel_validity.all()):
            missing = [
                name for name, valid in zip(self.channel_names, self.channel_validity.tolist())
                if not valid
            ]
            raise ValueError(f"Siena channels lack canonical coordinates: {missing}")

    def __len__(self):
        return len(self.samples)

    def enable_coordinate_only_channels(self):
        """All original Siena electrodes with canonical coordinates are active."""

    def __getitem__(self, index):
        sample = self.samples[index]
        path = Path(sample["file"])
        with path.open("rb") as source:
            record = pickle.load(source)
        signal = record["X"]
        expected = (len(self.channel_names), self.source_sample_rate * self.segment_seconds)
        if tuple(signal.shape) != expected:
            raise ValueError(f"{path} has {signal.shape}, expected {expected}")
        label = int(record["Y"])
        if label not in (0, 1):
            raise ValueError(f"{path} has invalid binary label {label}")
        signal = resample(signal, self.sample_rate * self.segment_seconds, axis=-1)
        # Matches preserved preprocessing_siena.CustomDataset.normalize.
        signal = torch.as_tensor(signal * 10000.0, dtype=torch.float32)
        return {
            "x": signal,
            "channel_coordinates": self.channel_coordinates,
            "channel_region_ids": self.channel_region_ids,
            "channel_validity": self.channel_validity,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": self.subject_ids[index],
            "recording_id": self.recording_ids[index],
            "sample_id": path.name,
        }
