"""Connect the checked-in reference data package when src/data is absent."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, module)
    return module


def ensure_data_imports():
    import src

    if "src.data" in sys.modules:
        return "src.data (already connected)"
    if importlib.util.find_spec("src.data") is not None:
        return "src/data"
    reference = ROOT / "tests/reference/src"
    if not (reference / "datasets/pretraining_dataset.py").is_file():
        raise FileNotFoundError("Need src/data or the checked-in tests/reference/src data sources")
    # These are namespace aliases, not copies or replacements of dataset code.
    for name, directory in (("src.utils", "utils"), ("src.datasets", "datasets")):
        if name not in sys.modules:
            _package(name, reference / directory)
    package = _package("src.data", reference)
    package.datasets = sys.modules["src.datasets"]
    sys.modules["src.data.datasets"] = package.datasets
    geometry = importlib.import_module("src.utils.electrode_geometry")
    package.electrode_geometry = geometry
    sys.modules["src.data.electrode_geometry"] = geometry
    return "tests/reference/src (namespace aliases; src/data absent)"
