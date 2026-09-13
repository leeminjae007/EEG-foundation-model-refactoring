"""GR9-1 checkpoint → 새 이름. 누락/중복/shape는 strict loading으로 검증한다."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import PretrainModel


def encoder_key(key):
    if key.startswith("patch_encoder."):
        return key.replace("patch_encoder.", "tokenizer.", 1)
    if key == "positional_encoder.spatial.projection.weight":
        return "position.projection.weight"
    if key == "positional_encoder.post_fusion_norm.weight":
        return "position.norm.weight"
    if key.startswith("context_encoder."):
        key = key.replace("context_encoder.", "encoder.", 1)
        return key.replace(".block.", ".").replace(".mlp.layers.", ".mlp.")
    return key


def decoder_key(key):
    if key == "spherical_expansion.projection.weight":
        return "position.projection.weight"
    if key == "position_post_fusion_norm.weight":
        return "position.norm.weight"
    return key


def convert(checkpoint):
    state = {}
    mapping = {}
    for key, tensor in checkpoint["context_encoder"].items():
        target = "backbone." + encoder_key(key)
        assert target not in state, target
        state[target] = tensor
        mapping["context_encoder." + key] = target
    for key, tensor in checkpoint["decoder"].items():
        target = "decoder." + decoder_key(key)
        assert target not in state, target
        state[target] = tensor
        mapping["decoder." + key] = target
    return state, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu")
    state, mapping = convert(checkpoint)
    model = PretrainModel(checkpoint["resolved_config"], torch.device("cpu"))
    model.load_state_dict(state, strict=True)
    # 가중치만 변환한다. 기존 optimizer를 이어받는 resume 변환이 아니다.
    result = {"model": state, "config": checkpoint["resolved_config"],
              "epoch": checkpoint["epoch"], "key_mapping": mapping}
    with args.destination.open("xb") as output:
        torch.save(result, output)


if __name__ == "__main__":
    main()
