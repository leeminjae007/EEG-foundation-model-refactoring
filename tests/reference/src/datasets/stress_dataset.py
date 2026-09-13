"""Direct loader for processed mental-arithmetic stress LMDB samples."""

from src.datasets.processed_dataset import SplitLMDBDataset


STRESS_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "FZ", "CZ", "PZ", "A2-A1",
)


class StressDataset(SplitLMDBDataset):
    channel_names = STRESS_CHANNELS
    ignored_channel_names = ('A2-A1',)
    stored_shape = (20, 5, 200)
    num_outputs = 2
