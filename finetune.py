"""Final mask-55 downstream finetuning entry point."""

import argparse
from src.training.runtime import ROOT, set_paths

set_paths()
import torch
import yaml
from src.training.engine import run_finetune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--data-dir", required=True, help="Processed downstream LMDB directory")
    parser.add_argument("--checkpoint", required=True, help="Final pretraining checkpoint")
    parser.add_argument("--dataset", default="chb")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--class-counts", nargs=2, type=int, metavar=("NEGATIVE", "POSITIVE"),
                        help="Required when switching to a different binary dataset")
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
    config["seed"] = args.seed
    config["data"]["dataset"] = args.dataset
    config["data"]["dataset_dir"] = args.data_dir
    config["model"]["checkpoint"] = args.checkpoint
    config["runtime"]["output"] = args.output or f"outputs/downstream/{args.dataset}/seed{args.seed}"
    if args.learning_rate is not None:
        for name in ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate"):
            config["optimization"][name] = args.learning_rate
    if args.weight_decay is not None:
        config["optimization"]["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        config["model"]["head_dropout"] = args.dropout
    if args.class_counts:
        config["optimization"]["class_counts"] = args.class_counts
    elif args.dataset != "chb" and args.dataset in ("siena", "stress", "tuab", "tusz"):
        parser.error("Pass --class-counts NEGATIVE POSITIVE for this binary dataset")
    run_finetune(config, args)
