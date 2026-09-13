"""Direct loader for processed BCIC2020-3 imagined-speech samples."""

from src.datasets.processed_dataset import SplitLMDBDataset


BCIC2020_3_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3",
    "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
    "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1",
    "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
    "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8",
)


class SpeechDataset(SplitLMDBDataset):
    channel_names = BCIC2020_3_CHANNELS
    ignored_channel_names = ("TP9", "TP10", "PO9", "PO10", "FT9", "FT10")
    # Actual-channel models use the valid 10-10 coordinates for these leads.
    coordinate_only_channel_names = ignored_channel_names
    stored_shape = (64, 3, 200)
    num_outputs = 5

    def subject_id(self, key):
        return key.split("-", 2)[1]
