"""BCI Competition 2020 Track 3 channel metadata, in stored channel order.

Provenance:
* Original MATLAB ``epo_{train,validation,test}.clab`` is checked against
  ``mnt.clab`` by the NEMAR conversion script, then used without reordering:
  https://github.com/nemarDatasets/nm000113/blob/main/code/bcic2020-3.py
* All 45 NEMAR channel sidecars (15 subjects x 3 runs) have this same order:
  https://github.com/nemarDatasets/nm000113/tree/main/sub-01/eeg
* CBraMod's benchmark preprocessing transposes the signal to
  (trials, channels, time) but does not permute channels:
  https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_speech.py

The legacy processed LMDB stores no channel names. This list describes the
published preprocessing protocol; verify against original MATLAB clab before
claiming provenance for an independently generated LMDB.
"""

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

BCIC2020_3_SPLIT_TRIALS = {"train": 4500, "val": 750, "test": 750}
BCIC2020_3_STORED_SHAPE = (64, 3, 200)
BCIC2020_3_NUM_CLASSES = 5
