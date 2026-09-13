"""Shared direct-reader for the confirmed CBraMod split-indexed LMDB format."""

import io
import math
import pickle
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset

from src.utils.electrode_geometry import (
    canonicalize_channel_name,
    resolve_channel_coordinates,
    resolve_channel_regions,
)


def unknown_channel_names(count):
    return tuple(f"UNKNOWN-{index + 1:02d}" for index in range(count))


class _NumpyPickleReader(pickle.Unpickler):
    """Read NumPy-2-created arrays in the cluster's NumPy 1.24 runtime."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def _loads(payload):
    return _NumpyPickleReader(io.BytesIO(payload)).load()


class SplitLMDBDataset(Dataset):
    """Read ``__keys__[split]`` and ``{"sample", "label"}`` records."""

    channel_names = ()
    stored_channel_indices = ()
    ignored_channel_names = ()
    coordinate_only_channel_names = ()
    stored_shape = ()
    sample_rate = 200
    task = "multiclass"
    num_outputs = None

    def __init__(self, data_dir, split):
        self.path = Path(data_dir)
        self.split = str(split)
        database = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with database.begin(write=False) as transaction:
            split_keys = _loads(transaction.get(b"__keys__"))
        database.close()
        self.keys = tuple(split_keys[self.split])
        self.subject_ids = tuple(
            self.subject_id(key) for key in self.keys
        )
        self.recording_ids = tuple(
            key.rsplit("-", 1)[0] for key in self.keys
        )
        self.database = None
        self.sample_scale_correction = None
        stored_channels = int(self.stored_shape[0])
        selected_indices = (
            tuple(self.stored_channel_indices)
            if self.stored_channel_indices
            else tuple(range(stored_channels))
        )
        if len(selected_indices) != len(self.channel_names):
            raise ValueError(
                f'{type(self).__name__} selects {len(selected_indices)} '
                f'channels but defines {len(self.channel_names)} names'
            )
        if (
            len(set(selected_indices)) != len(selected_indices)
            or any(index < 0 or index >= stored_channels
                   for index in selected_indices)
        ):
            raise ValueError(
                f'{type(self).__name__} has invalid stored channel indices'
            )
        self.selected_channel_indices = selected_indices
        self.channel_coordinates, coordinate_validity = (
            resolve_channel_coordinates(self.channel_names)
        )
        self.channel_coordinate_validity = coordinate_validity
        self.channel_region_ids = resolve_channel_regions(
            self.channel_names
        )
        ignored = {
            canonicalize_channel_name(name)
            for name in self.ignored_channel_names
        }
        self.channel_validity = torch.tensor(
            [
                canonicalize_channel_name(name) not in ignored
                for name in self.channel_names
            ],
            dtype=torch.bool,
        )
        unresolved = self.channel_validity & (
            ~coordinate_validity | self.channel_region_ids.lt(0)
        )
        if unresolved.any():
            names = [
                name for name, missing in zip(
                    self.channel_names, unresolved.tolist()
                ) if missing
            ]
            raise ValueError(
                f'{type(self).__name__} needs canonical channel metadata '
                f'for active channels: {names}'
            )

    def enable_coordinate_only_channels(self):
        """Enable valid-coordinate electrodes excluded only by region routing."""
        coordinate_only = {
            canonicalize_channel_name(name)
            for name in self.coordinate_only_channel_names
        }
        for index, name in enumerate(self.channel_names):
            if canonicalize_channel_name(name) not in coordinate_only:
                continue
            if not self.channel_coordinate_validity[index]:
                raise ValueError(
                    f'{type(self).__name__} has no coordinate for {name}'
                )
            self.channel_validity[index] = True

    def _database(self):
        if self.database is None:
            self.database = lmdb.open(
                str(self.path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
        return self.database

    def __len__(self):
        return len(self.keys)

    def subject_id(self, key):
        return key.split("-", 1)[0]

    def __getitem__(self, index):
        key = self.keys[index]
        with self._database().begin(write=False) as transaction:
            record = _loads(transaction.get(key.encode()))
        signal = record["sample"]
        if tuple(signal.shape) != tuple(self.stored_shape):
            raise ValueError(
                f"{key} has signal shape {signal.shape}, expected "
                f"{self.stored_shape}"
            )
        signal = torch.as_tensor(signal, dtype=torch.float32).reshape(
            self.stored_shape[0], -1
        )
        signal = signal[list(self.selected_channel_indices)]
        if self.sample_scale_correction is not None:
            reference_std = self.sample_scale_correction['reference_std']
            ratio = float(signal.std()) / reference_std
            if ratio >= self.sample_scale_correction['ratio_threshold']:
                exponent = max(1, math.floor(math.log10(ratio) + 0.5))
                signal = signal / float(10 ** exponent)
        if self.task == "regression":
            label = torch.tensor(float(record["label"]), dtype=torch.float32)
        else:
            value = int(record["label"])
            if not 0 <= value < self.num_outputs:
                raise ValueError(
                    f"{key} has label {value}, expected "
                    f"[0,{self.num_outputs})"
                )
            label = torch.tensor(value, dtype=torch.long)
        return {
            "x": signal / 100.0,
            "channel_coordinates": self.channel_coordinates,
            "channel_region_ids": self.channel_region_ids,
            "channel_validity": self.channel_validity,
            "label": label,
            "subject_id": self.subject_ids[index],
            "recording_id": self.recording_ids[index],
            "sample_id": key,
        }
