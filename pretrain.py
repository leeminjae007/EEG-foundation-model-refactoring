"""Final mask-55 pretraining entry point."""

import argparse
import multiprocessing
import os
from src.training.runtime import ROOT, set_paths

set_paths()
import torch
import yaml
from src.training.engine import run_pretrain

# torchrun children can inherit multiprocessing's cached temp directory from a
# deeply nested immutable campaign path. Force that internal socket directory
# to the short node-local TMPDIR before DataLoader workers are created.
multiprocessing.current_process()._config["tempdir"] = os.environ.get("TMPDIR", "/tmp")
if "file_descriptor" in torch.multiprocessing.get_all_sharing_strategies():
    torch.multiprocessing.set_sharing_strategy("file_descriptor")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data-dir", required=True, help="Processed pretraining LMDB directory")
    parser.add_argument("--output", default="outputs/pretrain")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(2)
    config = yaml.safe_load((ROOT / args.config).read_text())
    config["data"]["dataset_dir"] = args.data_dir
    config["runtime"]["output"] = args.output
    run_pretrain(config, args)
