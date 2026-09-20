"""GR9-1 전체 미세조정. 분류는 validation BAcc와 AUROC/Kappa를 함께 보관한다."""

import argparse
from src.training.runtime import ROOT, set_paths

set_paths()
import torch
import yaml
from src.training.engine import run_finetune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="One bounded train/validation/test epoch at configured batch size; diagnostic output only")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.smoke_batches < 0 or (args.smoke_batches and (args.smoke or args.resume or args.distributed)):
        parser.error("--smoke-batches must be positive and cannot be combined with smoke/resume/distributed")
    if args.device == "cpu":
        torch.set_num_threads(2)
    config = yaml.safe_load((ROOT / args.config).read_text())
    run_finetune(config, args)
