"""CBraMod-protocol Mumtaz MDD/healthy processed LMDB reader.

The channel order follows the source preprocessing's ``selected_channels``;
the original train/val/test split is intentionally retained for comparability.
https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_mumtaz.py
"""

from src.data.datasets.processed_dataset import SplitLMDBDataset


MUMTAZ_CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


class MumtazDataset(SplitLMDBDataset):
    channel_names = MUMTAZ_CHANNELS
    stored_shape = (19, 5, 200)
    task = "binary"
    num_outputs = 2

    def __init__(self, data_dir, split):
        super().__init__(data_dir, split)
        self.recording_ids = tuple(key.rsplit("_", 1)[0] for key in self.keys)

    def subject_id(self, key):
        # e.g. "H S27 EO_17" or "MDD S34 EO_25"
        return key.rsplit("_", 1)[0].rsplit(" ", 1)[0]
