"""Read one real distributed pretraining batch to validate worker IPC."""

import argparse
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.runtime import set_paths

set_paths()
import torch
import yaml
from src.data.datasets.pretraining_dataset import make_pretraining_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    multiprocessing.current_process()._config["tempdir"] = os.environ.get("TMPDIR", "/tmp")
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    config = yaml.safe_load(args.config.read_text())
    data = config["data"]
    dataset, loader, _ = make_pretraining_loader(
        data["dataset_dir"], 2, data["pin_memory"], data["num_workers"],
        1, 0, config["seed"], True, len(data["channel_names"]),
        data["num_patches"], config["patch_encoder"]["patch_samples"])
    signals, indices = next(iter(loader))
    expected = (2, len(data["channel_names"]),
                data["num_patches"] * config["patch_encoder"]["patch_samples"])
    if tuple(signals.shape) != expected or tuple(indices.shape) != (2,):
        raise RuntimeError(f"unexpected batch: {tuple(signals.shape)}, {tuple(indices.shape)}")
    dataset.close()
    print(f"DataLoader IPC verified: shape={expected}, workers={data['num_workers']}, "
          f"tempdir={multiprocessing.current_process()._config['tempdir']}", flush=True)
