"""Read CSBrain's published region assignments and within-region ordering."""

import ast
from functools import lru_cache

from ablation.sources import VENDOR

DATASETS = {"seed-v": "seedv", "seed-vig": "seedvig"}


def canonical(name):
    from src.data.electrode_geometry import canonicalize_channel_name
    return canonicalize_channel_name(name)


@lru_cache(maxsize=None)
def source_metadata(dataset):
    path = VENDOR / "csbrain" / ("pretrain_main.py" if dataset == "pretrain" else
                                "models/model_for_" + DATASETS.get(dataset, dataset) + ".py")
    values = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id.lower()
            for suffix in ("brain_regions", "electrode_labels", "signal_electrodes", "selected_channels", "topology"):
                if name.endswith(suffix):
                    values[suffix] = ast.literal_eval(node.value)
    labels = values.get("electrode_labels", values.get("signal_electrodes", values.get("selected_channels")))
    regions = values["brain_regions"]
    if labels is None or len(labels) != len(regions):
        raise ValueError("Invalid original CSBrain metadata: " + str(path))
    topology = {region: [canonical(name) for name in names]
                for region, names in values["topology"].items()}
    if dataset == "physio":
        # The published metadata assigns FT7/FT8 to region 2 but lists them only
        # under topology[0]. Keep its region IDs and place the omitted entries
        # first in region 2, in their original selected_channels order. This is
        # the sole explicit metadata repair; the vendor file stays byte-identical.
        topology[2] = ["FT7", "FT8"] + topology[2]
    return [canonical(name) for name in labels], regions, topology


def region_order(channel_names, dataset="pretrain"):
    labels, original_regions, topology = source_metadata(dataset)
    mapping = dict(zip(labels, original_regions))
    names = [canonical(name) for name in channel_names]
    ignored = set()
    if dataset != "pretrain":
        from src.data.datasets.registry import get_dataset_spec
        ignored = {canonical(name) for name in getattr(get_dataset_spec(dataset).dataset_class,
                                                       "ignored_channel_names", ())}
    active = [index for index, name in enumerate(names) if name not in ignored]
    # Original bipolar wrappers assign a pair by its first (signal) electrode.
    signals = [name if name in mapping else name.split("-")[0] for name in names]
    missing = [signals[index] for index in active if signals[index] not in mapping]
    if missing:
        raise ValueError("Channels absent from original CSBrain " + dataset + " metadata: " + str(missing))
    regions = [mapping[signals[index]] for index in active]
    if any(region not in range(5) for region in regions):
        raise ValueError("Original CSBrain supports region IDs 0..4 only")
    order = sorted(active, key=lambda i: (mapping[signals[i]], topology[mapping[signals[i]]].index(signals[i])))
    return regions, order
