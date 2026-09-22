"""Small CPU check for the published architecture and checkpoint adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
import argparse

import torch
import yaml

from src.model import PretrainModel
from src.training.engine import build_finetune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-checkpoint", help="Optionally verify a private final checkpoint without copying it")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pretrain.yaml").read_text())
    assert config["masking"]["mask_ratio"] == 0.55
    assert config["encoder"]["fusion_gate"] == "patch_feature"
    assert config["mae"]["decoder_depth"] == 2
    model = PretrainModel(config, torch.device("cpu"))
    assert len(model.backbone.encoder.patch_gates) == 3
    assert len(model.decoder.blocks) == 2
    model.eval()
    with torch.no_grad():
        prediction, masks = model(torch.randn(1, 19, 6000))
    assert prediction.shape[0] == 1
    assert torch.isfinite(prediction).all()
    assert masks["target_mask"].any()

    # Simulate the canonical training wrapper's encoder.core namespace.
    state = {}
    for key, tensor in model.state_dict().items():
        if key.startswith("backbone.encoder."):
            key = key.replace("backbone.encoder.", "backbone.encoder.core.", 1)
        state[key] = tensor
    saved = dict(config)
    saved["ablation"] = {"encoder": "mjde", "position": "shpe"}
    downstream = yaml.safe_load(Path("configs/downstream.yaml").read_text())
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "pretrain.pth"
        torch.save({"config": saved, "model": state}, checkpoint)
        downstream["model"]["checkpoint"] = str(checkpoint)
        finetune = build_finetune(downstream)
    assert finetune.backbone.encoder.fusion_gate_mode == "patch_feature"
    if args.real_checkpoint:
        downstream["model"]["checkpoint"] = args.real_checkpoint
        real = build_finetune(downstream)
        assert real.backbone.encoder.fusion_gate_mode == "patch_feature"
    print("release verification passed")


if __name__ == "__main__":
    main()
