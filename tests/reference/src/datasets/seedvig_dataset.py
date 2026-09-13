"""Direct loader for processed SEED-VIG regression LMDB samples."""

from src.datasets.processed_dataset import (
    SplitLMDBDataset,
)


class SEEDVIGDataset(SplitLMDBDataset):
    channel_names = (
        'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
        'CP1', 'CP2', 'P1', 'PZ', 'P2',
        'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2',
    )
    stored_shape = (17, 8, 200)
    task = "regression"
    num_outputs = 1
