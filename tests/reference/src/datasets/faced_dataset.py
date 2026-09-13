"""Direct loader for processed FACED LMDB samples."""

from src.datasets.processed_dataset import SplitLMDBDataset


# FACED stores 30 scalp EEG channels followed by the A2/A1 mastoids. The
# published acquisition order is retained exactly; mastoids are references,
# not scalp signals routed to a latent region.
FACED_SCALP_CHANNELS = (
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
    'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2',
)


class FACEDDataset(SplitLMDBDataset):
    channel_names = FACED_SCALP_CHANNELS
    stored_channel_indices = tuple(range(30))
    stored_shape = (32, 10, 200)
    num_outputs = 9
