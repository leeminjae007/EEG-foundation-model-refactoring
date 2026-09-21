"""Published BCIC2020-3 channel order must stay complete and stable."""

from src.data.datasets.bcic20203_metadata import (
    BCIC2020_3_CHANNELS,
    BCIC2020_3_NUM_CLASSES,
    BCIC2020_3_SPLIT_TRIALS,
    BCIC2020_3_STORED_SHAPE,
)


def test_bcic20203_metadata_contract():
    assert len(BCIC2020_3_CHANNELS) == 64
    assert len(set(BCIC2020_3_CHANNELS)) == 64
    assert BCIC2020_3_CHANNELS[:4] == ("Fp1", "Fp2", "F7", "F3")
    assert BCIC2020_3_CHANNELS[-4:] == ("PO3", "POz", "PO4", "PO8")
    assert BCIC2020_3_STORED_SHAPE == (64, 3, 200)
    assert BCIC2020_3_SPLIT_TRIALS == {"train": 4500, "val": 750, "test": 750}
    assert BCIC2020_3_NUM_CLASSES == 5
