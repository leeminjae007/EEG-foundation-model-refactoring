"""CBraMod-protocol BCI Competition 2020 Track 3 LMDB reader."""

from src.data.datasets.bcic20203_metadata import BCIC2020_3_CHANNELS
from src.data.datasets.processed_dataset import SplitLMDBDataset


class BCIC20203Dataset(SplitLMDBDataset):
    channel_names = BCIC2020_3_CHANNELS
    stored_shape = (64, 3, 200)
    num_outputs = 5
    # The backbone consumes coordinates/validity, not the auxiliary region ID.
    # Peripheral FT9/FT10, TP9/TP10 and PO9/PO10 have montage coordinates,
    # but no region bucket in the current optional region mapping.
    allow_coordinate_only_channels = True

    def subject_id(self, key):
        # CBraMod writes "train-Data_Sample01-0" (analogous val/test keys).
        parts = key.split("-")
        return parts[1] if len(parts) >= 3 else key
