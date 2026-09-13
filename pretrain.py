"""GR9-1 모델 + geometry 마스킹 사전학습 진입점. --smoke는 소량 실제 batch 한 번만 학습한다."""

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
torch.multiprocessing.set_sharing_strategy("file_descriptor")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(2)
    config = yaml.safe_load((ROOT / args.config).read_text())
    run_pretrain(config, args)
