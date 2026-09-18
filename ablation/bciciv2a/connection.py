"""Add a validation-only switch to an isolated copy of the baseline engine."""


def connected_source(original):
    changes = {
        '    for split in ("train", "val", "test"):\n':
        '    for split in (("train", "val", "test") if config["runtime"].get("evaluate_test", True) else ("train", "val")):\n',
        '            test = evaluate(model, loaders["test"], spec.task, data["dataset"], device, world)\n':
        '            test = (evaluate(model, loaders["test"], spec.task, data["dataset"], device, world)\n'
        '                    if config["runtime"].get("evaluate_test", True) else None)\n',
    }
    for old, new in changes.items():
        if original.count(old) != 1:
            raise ValueError("Baseline engine changed; inspect before connecting validation-only mode")
        original = original.replace(old, new)
    compile(original, "engine.py", "exec")
    return original


def without_removed_seedvig(original):
    """Keep the remaining dataset registry usable after external SEED-VIG removal."""
    for line in (
        "from src.data.datasets.seedvig_dataset import SEEDVIGDataset\n",
        '    "seed-vig": DatasetSpec(SEEDVIGDataset, "regression", 1, 17, 1600),\n',
    ):
        if original.count(line) != 1:
            raise ValueError("Inspect changed dataset registry before snapshotting")
        original = original.replace(line, "")
    return original
