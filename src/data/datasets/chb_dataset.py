"""Direct loader for the processed CHB-MIT seizure segments."""

import pickle
from pathlib import Path

from scipy.signal import resample
import torch
from torch.utils.data import Dataset

from src.data.electrode_geometry import (
    CHB_MIT_16_CHANNELS,
    resolve_channel_coordinates,
    resolve_channel_regions,
)


CHB_SOURCE_SAMPLE_RATE = 256
CHB_MODEL_SAMPLE_RATE = 200
CHB_SEGMENT_SECONDS = 10


class CHBDataset(Dataset):
    """Read CBraMod-format ``{"X": [16,L], "y": 0|1}`` pickles.

    The stored segments are already preprocessed.  The sole signal conversion
    is the original CBraMod loader's local conversion of every stored segment
    to 2,000 samples so the result can be divided into one-second, 200-sample
    model patches. This also preserves the reference behavior for augmented
    seizure segments clipped by the end of a recording.
    """

    channel_names = CHB_MIT_16_CHANNELS
    sample_rate = CHB_MODEL_SAMPLE_RATE
    task = "binary"
    num_outputs = 2

    def __init__(self, data_root, split):
        self.split = str(split)
        split_dir = Path(data_root) / self.split
        self.files = sorted(split_dir.glob("*.pkl"))
        self.subject_ids = tuple(
            path.name.split("_", 1)[0] for path in self.files
        )
        self.recording_ids = tuple(
            path.stem.rsplit("_", 1)[0] for path in self.files
        )
        self.channel_coordinates, _ = (
            resolve_channel_coordinates(self.channel_names)
        )
        self.channel_region_ids = resolve_channel_regions(
            self.channel_names
        )
        self.channel_validity = torch.ones(
            len(self.channel_names), dtype=torch.bool
        )

    def __len__(self):
        return len(self.files)

    def binary_label_counts(self):
        """Count train labels without applying signal resampling."""
        counts = [0, 0]
        for path in self.files:
            with path.open("rb") as source:
                label = int(pickle.load(source)["y"])
            if label not in (0, 1):
                raise ValueError(f"{path} has binary label {label}")
            counts[label] += 1
        return tuple(counts)

    def enable_coordinate_only_channels(self):
        """All stored CHB-MIT channels are already coordinate-enabled."""

    def __getitem__(self, index):
        path = self.files[index]
        with path.open("rb") as source:
            record = pickle.load(source)
        signal = record["X"]
        if signal.ndim != 2 or signal.shape[0] != len(self.channel_names):
            raise ValueError(
                f"{path} has signal shape {signal.shape}; expected 16 channels"
            )
        label = int(record["y"])
        if label not in (0, 1):
            raise ValueError(f"{path} has binary label {label}")
        model_length = CHB_MODEL_SAMPLE_RATE * CHB_SEGMENT_SECONDS
        signal = resample(signal, model_length, axis=-1)
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
