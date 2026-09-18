"""Direct loader for processed SEED-V LMDB samples."""

from src.data.datasets.processed_dataset import SplitLMDBDataset
from src.data.electrode_geometry import SEED_V_62_CHANNELS


class SEEDVDataset(SplitLMDBDataset):
    channel_names = SEED_V_62_CHANNELS
    stored_shape = (62, 1, 200)
    num_outputs = 5

    def subject_id(self, key):
        return key.split("_", 1)[0]
