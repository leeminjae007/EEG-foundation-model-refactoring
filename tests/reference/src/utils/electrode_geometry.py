"""Stateless EEG channel-name and standard-montage geometry utilities."""

from functools import lru_cache
import re

import mne
import torch


TUEG_19_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)
CHB_MIT_16_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
SEED_V_62_CHANNELS = (
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
)

_REGIONS = (
    ("FP1", "AF7", "AF3", "F7", "F5", "F3", "F1", "FC5", "FC3", "FC1"),
    ("FP2", "AF8", "AF4", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6"),
    ("FPZ", "AFZ", "FZ", "FCZ"),
    ("C5", "C3", "C1"),
    ("C2", "C4", "C6"),
    ("CZ",),
    ("FT7", "T9", "T7", "TP7", "P7"),
    ("FT8", "T10", "T8", "TP8", "P8"),
    ("CP5", "CP3", "CP1", "P5", "P3", "P1"),
    ("CP2", "CP4", "CP6", "P2", "P4", "P6"),
    ("CPZ", "PZ", "POZ", "OZ", "IZ"),
    ("PO7", "PO5", "PO3", "CB1", "O1"),
    ("PO4", "PO6", "PO8", "O2", "CB2"),
)
_REGION_BY_CHANNEL = {
    name: region for region, names in enumerate(_REGIONS) for name in names
}
_BIPOLAR_REGIONS = {
    "FP1-F7": 0, "F7-T7": 6, "T7-P7": 6, "P7-O1": 11,
    "FP2-F8": 1, "F8-T8": 7, "T8-P8": 7, "P8-O2": 12,
    "FP1-F3": 0, "F3-C3": 3, "C3-P3": 8, "P3-O1": 11,
    "FP2-F4": 1, "F4-C4": 4, "C4-P4": 9, "P4-O2": 12,
}
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_COORDINATE_ALIASES = {"CB1": "PO9", "CB2": "PO10"}
_REFERENCES = {"A1", "A2", "M1", "M2"}


def canonicalize_channel_name(name):
    name = str(name).strip().upper()
    name = re.sub(r"^EEG\s+", "", name)
    name = re.sub(r"-(REF|LE|AVG|AR)$", "", name)
    name = name.replace("–", "-").replace("—", "-")
    return "-".join(_ALIASES.get(part, part) for part in name.split("-"))


def channel_region_index(channel_name):
    name = canonicalize_channel_name(channel_name)
    if name in _REGION_BY_CHANNEL:
        return _REGION_BY_CHANNEL[name]
    if name in _BIPOLAR_REGIONS:
        return _BIPOLAR_REGIONS[name]
    parts = name.split("-")
    if len(parts) == 2 and parts[1] in _REFERENCES:
        return _REGION_BY_CHANNEL.get(parts[0], -1)
    if len(parts) == 2 and all(part in _REGION_BY_CHANNEL for part in parts):
        first, second = (_REGION_BY_CHANNEL[part] for part in parts)
        return first if first == second else -1
    return -1


def resolve_channel_regions(channel_names):
    return torch.tensor(
        [channel_region_index(name) for name in channel_names],
        dtype=torch.long,
    )


@lru_cache(maxsize=1)
def _standard_geometry():
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = {
        name.upper(): torch.tensor(position, dtype=torch.float32)
        for name, position in montage.get_positions()["ch_pos"].items()
    }
    radius = max(float(position.norm()) for position in positions.values())
    return {name: position / radius for name, position in positions.items()}


def _coordinate(channel_name, montage):
    name = canonicalize_channel_name(channel_name)
    name = _COORDINATE_ALIASES.get(name, name)
    if name in montage:
        return montage[name]
    parts = name.split("-")
    if len(parts) == 2 and parts[1] in _REFERENCES:
        return montage.get(parts[0])
    if len(parts) == 2 and all(part in montage for part in parts):
        return (montage[parts[0]] + montage[parts[1]]) / 2.0
    return None


def resolve_channel_coordinates(channel_names):
    montage = _standard_geometry()
    coordinates = [_coordinate(name, montage) for name in channel_names]
    validity = torch.tensor(
        [coordinate is not None for coordinate in coordinates],
        dtype=torch.bool,
    )
    coordinates = [
        coordinate if coordinate is not None else torch.zeros(3)
        for coordinate in coordinates
    ]
    return torch.stack(coordinates), validity
