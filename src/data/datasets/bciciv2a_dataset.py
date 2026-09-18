"""Direct loader for processed BCI Competition IV 2a LMDB samples."""

from src.data.datasets.processed_dataset import SplitLMDBDataset


BCICIV2A_CHANNELS = (
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ",
    "P2", "POZ",
)


class BCICIV2ADataset(SplitLMDBDataset):
    channel_names = BCICIV2A_CHANNELS
    stored_shape = (22, 4, 200)
    num_outputs = 4

    def subject_id(self, key):
        return key[:3]
