"""Direct loader for processed PhysioNet motor-imagery LMDB samples."""

from src.data.datasets.processed_dataset import SplitLMDBDataset


PHYSIO_CHANNELS = (
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ",
)


class PhysioDataset(SplitLMDBDataset):
    channel_names = PHYSIO_CHANNELS
    stored_shape = (64, 4, 200)
    num_outputs = 4

    def subject_id(self, key):
        return key[:4]
