"""Direct loader for processed Mumtaz MDD LMDB samples."""

import re

from src.datasets.processed_dataset import SplitLMDBDataset
from src.utils.electrode_geometry import TUEG_19_CHANNELS


class MumtazDataset(SplitLMDBDataset):
    channel_names = TUEG_19_CHANNELS
    stored_shape = (19, 5, 200)
    num_outputs = 2

    def subject_id(self, key):
        return re.search(r"(?:MDD|H) S\d+", key).group(0)
