"""Authoritative processed-dataset contracts used by downstream training."""

from dataclasses import dataclass

from src.datasets.bciciv2a_dataset import BCICIV2ADataset
from src.datasets.chb_dataset import CHBDataset
from src.datasets.faced_dataset import FACEDDataset
from src.datasets.hmc_dataset import HMCDataset
from src.datasets.isruc_dataset import ISRUCDataset
from src.datasets.physio_dataset import PhysioDataset
from src.datasets.seedv_dataset import SEEDVDataset
from src.datasets.siena_dataset import SienaDataset
from src.datasets.stress_dataset import StressDataset
from src.datasets.tuab_dataset import TUABDataset
from src.datasets.tuev_dataset import TUEVDataset


@dataclass(frozen=True)
class DatasetSpec:
    dataset_class: type
    task: str
    num_outputs: int
    num_channels: int
    signal_length: int


DATASET_SPECS = {
    "bciciv2a": DatasetSpec(BCICIV2ADataset, "multiclass", 4, 22, 800),
    "chb": DatasetSpec(CHBDataset, "binary", 1, 16, 2000),
    "faced": DatasetSpec(FACEDDataset, "multiclass", 9, 30, 2000),
    "hmc": DatasetSpec(HMCDataset, "multiclass", 5, 4, 6000),
    "isruc": DatasetSpec(ISRUCDataset, "multiclass", 5, 6, 6000),
    "physio": DatasetSpec(PhysioDataset, "multiclass", 4, 64, 800),
    "seed-v": DatasetSpec(SEEDVDataset, "multiclass", 5, 62, 200),
    "siena": DatasetSpec(SienaDataset, "binary", 1, 29, 2000),
    "stress": DatasetSpec(StressDataset, "binary", 1, 20, 1000),
    "tuab": DatasetSpec(TUABDataset, "binary", 1, 16, 2000),
    "tuev": DatasetSpec(TUEVDataset, "multiclass", 6, 16, 1000),
}


def get_dataset_spec(dataset_name):
    return DATASET_SPECS[dataset_name]
